/*
 * imu.c — IMU detection, initialization, and raw data reads
 *
 * Supported chips:
 *   - MPU6000:   WHO_AM_I register 0x75, expected value 0x68
 *   - ICM42688P: WHO_AM_I register 0x75, expected value 0x47
 *   - BMI270:    CHIP_ID  register 0x00, expected value 0x24
 *
 * What the sensors actually measure:
 *   accel (ax/ay/az) — linear acceleration in g, including gravity.
 *                      When flat on a table: ax~0, ay~0, az~±1g.
 *   gyro  (gx/gy/gz) — angular RATE in °/s around each axis.
 *                      This is NOT yaw/pitch/roll; those are integrated
 *                      angles derived in the attitude estimation step.
 *   temp             — chip junction temperature (raw counts).
 *
 * Scaling at the configured full-scale ranges:
 *   ICM42688P / MPU6000 gyro  ±2000 dps  →  16.4 LSB per °/s
 *   ICM42688P accel ±16 g                →   2048 LSB per g
 *   MPU6000   accel ±8 g                 →   4096 LSB per g
 *
 * The GYRO_1_ALIGN CW180_DEG from the Betaflight config means
 * the IMU is mounted rotated 180° on the PCB. We apply axis
 * remapping in a later step when building attitude estimation.
 */

#include <stdint.h>
#include "imu.h"
#include "spi.h"

/* ============================================================
 * WHO_AM_I / CHIP_ID IDENTIFICATION
 * ============================================================ */

#define MPU6000_WHO_AM_I_REG    0x75
#define MPU6000_WHO_AM_I_VAL    0x68

#define ICM42688_WHO_AM_I_REG   0x75
#define ICM42688_WHO_AM_I_VAL   0x47

#define BMI270_CHIP_ID_REG      0x00
#define BMI270_CHIP_ID_VAL      0x24

/* ============================================================
 * SIMPLE DELAY (spin loop — calibrated for 168 MHz)
 * ============================================================ */
static void imu_delay_ms(uint32_t ms)
{
    volatile uint32_t count = ms * 33600U;
    while (count--) { __asm volatile ("nop"); }
}

/* ============================================================
 * IMU DETECTION
 * ============================================================ */
imu_type_t imu_detect(uint8_t *whoami)
{
    uint8_t id;

    id = spi1_read_register(MPU6000_WHO_AM_I_REG);  /* 0x75 — also ICM42688P */
    if (whoami) *whoami = id;

    if (id == MPU6000_WHO_AM_I_VAL)   return IMU_MPU6000;
    if (id == ICM42688_WHO_AM_I_VAL)  return IMU_ICM42688P;

    id = spi1_read_register(BMI270_CHIP_ID_REG);    /* 0x00 */
    if (whoami) *whoami = id;

    if (id == BMI270_CHIP_ID_VAL)     return IMU_BMI270;

    return IMU_NONE;
}


/* ============================================================
 * ICM-42688-P INITIALIZATION
 * ============================================================
 *
 * Register bank 0 (default after reset).
 *
 * Config chosen:
 *   Gyro:  ±2000 dps, 1 kHz ODR, low-noise mode
 *   Accel: ±16 g,     1 kHz ODR, low-noise mode
 *
 * No FIFO, no interrupts — we poll TEMP_DATA1 burst for
 * the freshest sample each loop iteration.
 *
 * Key register map:
 *   0x11  DEVICE_CONFIG   — soft reset (bit 0)
 *   0x16  FIFO_CONFIG     — 0x00 = bypass (stream to registers)
 *   0x1D  TEMP_DATA1      — first byte of 14-byte sensor burst
 *   0x4E  PWR_MGMT0       — accel + gyro power mode
 *   0x4F  GYRO_CONFIG0    — ODR + full-scale
 *   0x50  ACCEL_CONFIG0   — ODR + full-scale
 *   0x51  GYRO_CONFIG1    — UI filter bandwidth (LPF)
 *   0x76  REG_BANK_SEL    — bank select (we always stay on bank 0)
 *
 * Burst layout from 0x1D (14 bytes):
 *   [0..1]   TEMP  H,L
 *   [2..3]   ACCEL_X H,L
 *   [4..5]   ACCEL_Y H,L
 *   [6..7]   ACCEL_Z H,L
 *   [8..9]   GYRO_X H,L
 *   [10..11] GYRO_Y H,L
 *   [12..13] GYRO_Z H,L
 *
 * Temperature formula (from datasheet §4.13):
 *   T_degC = (TEMP_DATA / 132.48) + 25
 * ============================================================ */

#define ICM_REG_DEVICE_CONFIG   0x11
#define ICM_REG_FIFO_CONFIG     0x16
#define ICM_REG_TEMP_DATA1      0x1D
#define ICM_REG_PWR_MGMT0       0x4E
#define ICM_REG_GYRO_CONFIG0         0x4F
#define ICM_REG_ACCEL_CONFIG0        0x50
#define ICM_REG_GYRO_CONFIG1         0x51   /* filter order (bits [3:2]) */
#define ICM_REG_GYRO_ACCEL_CONFIG0   0x52   /* UI filter bandwidth for gyro [3:0] and accel [7:4] */
#define ICM_REG_BANK_SEL             0x76

/* GYRO_CONFIG0 [7:5] FS_SEL: 000=±2000, 001=±1000, 010=±500, 011=±250 dps
 * GYRO_CONFIG0 [3:0] ODR:    0110=1kHz, 0101=2kHz, 0100=4kHz, 0011=8kHz  */
#define ICM_GYRO_FS_2000DPS_ODR_1K  0x06   /* 000 | 0110 */

/* ACCEL_CONFIG0 [7:5] FS_SEL: 000=±16g, 001=±8g, 010=±4g, 011=±2g
 * ACCEL_CONFIG0 [3:0] ODR:    0110=1kHz                                   */
#define ICM_ACCEL_FS_16G_ODR_1K     0x06   /* 000 | 0110 */

/* PWR_MGMT0 [3:2] GYRO_MODE: 11=low-noise
 * PWR_MGMT0 [1:0] ACCEL_MODE: 11=low-noise                                */
#define ICM_PWR_GYRO_ACCEL_LN       0x0F   /* 0b00001111 */

/* GYRO_ACCEL_CONFIG0 (0x52) — UI filter bandwidth (same code table for both)
 * bits [7:4]: accel UI LPF BW
 * bits [3:0]: gyro  UI LPF BW
 *   0 = ODR/2  (Nyquist, no filtering — default after reset)
 *   1 = ODR/4
 *   2 = ODR/5
 *   3 = ODR/8
 *   4 = ODR/10
 *   5 = ODR/16
 *   6 = ODR/20  → 1000/20 = 50 Hz  ← chosen value
 *   7 = ODR/40
 *  15 = low-latency bypass (no filtering)
 *
 * Gyro: value 6 → 50 Hz, the Nyquist limit of the 100 Hz control loop.
 *
 * Accel: value 7 → ODR/40 = 25 Hz (added 2026-11-21). Before this the
 * accel had NO extra filtering (500 Hz bandwidth) while the firmware reads
 * it at only 100 Hz, so prop vibration (~100–200 Hz, near multiples of the
 * read rate) aliased into slow false swings. On the thrust-ramp bench
 * (TEST 16) az_std reached 0.6 g at 1250 µs, and a 0.12 g swing falsely
 * tripped the liftoff check. The accel only feeds the slow AHRS correction
 * (beta 0.05) and NN obs[3..5], so 25 Hz costs ~9 ms of lag and cuts a
 * 100 Hz vibration ~16x (2nd-order UI filter, the reset default).
 * It does not remove aliasing completely — only oversampling would. */
#define ICM_UI_BW_GYRO_50HZ_ACCEL_25HZ  0x76   /* accel [7:4] = 7 (ODR/40 = 25 Hz), gyro [3:0] = 6 (ODR/20 = 50 Hz) */

static void icm42688p_init(void)
{
    /* Stay on bank 0 */
    spi1_write_register(ICM_REG_BANK_SEL, 0x00);

    /* Soft reset — clears all registers */
    spi1_write_register(ICM_REG_DEVICE_CONFIG, 0x01);
    imu_delay_ms(2);   /* datasheet: 1 ms minimum after soft reset */

    /* Bypass FIFO — sensor data goes straight to registers */
    spi1_write_register(ICM_REG_FIFO_CONFIG, 0x00);

    /* Gyro: ±2000 dps, 1 kHz */
    spi1_write_register(ICM_REG_GYRO_CONFIG0, ICM_GYRO_FS_2000DPS_ODR_1K);

    /* UI low-pass filters via GYRO_ACCEL_CONFIG0 (0x52): gyro 50 Hz,
     * accel 25 Hz. Attenuates motor/prop vibration before it reaches the
     * output registers, so less of it aliases into the control band when
     * the firmware reads at 100 Hz. */
    spi1_write_register(ICM_REG_GYRO_ACCEL_CONFIG0, ICM_UI_BW_GYRO_50HZ_ACCEL_25HZ);

    /* Accel: ±16 g, 1 kHz */
    spi1_write_register(ICM_REG_ACCEL_CONFIG0, ICM_ACCEL_FS_16G_ODR_1K);

    /* Power on both in low-noise mode.
     * CRITICAL: datasheet §9.17 — do not access any other register
     * for ≥200 µs after writing PWR_MGMT0.  We use 5 ms for safety. */
    spi1_write_register(ICM_REG_PWR_MGMT0, ICM_PWR_GYRO_ACCEL_LN);
    imu_delay_ms(5);
}


/* ============================================================
 * MPU-6000 INITIALIZATION
 * ============================================================
 *
 * Config:
 *   Gyro:  ±2000 dps, DLPF off (8 kHz raw)
 *   Accel: ±8 g
 *
 * Burst layout from ACCEL_XOUT_H (0x3B), 14 bytes:
 *   [0..1]   ACCEL_X H,L
 *   [2..3]   ACCEL_Y H,L
 *   [4..5]   ACCEL_Z H,L
 *   [6..7]   TEMP    H,L
 *   [8..9]   GYRO_X  H,L
 *   [10..11] GYRO_Y  H,L
 *   [12..13] GYRO_Z  H,L
 *
 * Temperature formula: T_degC = TEMP_RAW/340 + 36.53
 * ============================================================ */

#define MPU_REG_SMPLRT_DIV      0x19
#define MPU_REG_CONFIG          0x1A
#define MPU_REG_GYRO_CONFIG     0x1B
#define MPU_REG_ACCEL_CONFIG    0x1C
#define MPU_REG_ACCEL_XOUT_H    0x3B
#define MPU_REG_SIGNAL_PATH_RST 0x68
#define MPU_REG_USER_CTRL       0x6A
#define MPU_REG_PWR_MGMT_1      0x6B

static void mpu6000_init(void)
{
    /* Hard reset */
    spi1_write_register(MPU_REG_PWR_MGMT_1, 0x80);
    imu_delay_ms(100);

    /* Reset signal paths */
    spi1_write_register(MPU_REG_SIGNAL_PATH_RST, 0x07);
    imu_delay_ms(5);

    /* PLL clock from X gyro (more stable than internal RC) */
    spi1_write_register(MPU_REG_PWR_MGMT_1, 0x01);
    imu_delay_ms(5);

    /* Disable I2C interface — SPI only */
    spi1_write_register(MPU_REG_USER_CTRL, 0x10);

    /* DLPF disabled → raw 8 kHz gyro */
    spi1_write_register(MPU_REG_CONFIG, 0x00);

    /* Gyro ±2000 dps [4:3]=11 */
    spi1_write_register(MPU_REG_GYRO_CONFIG,  0x18);

    /* Accel ±8 g [4:3]=10 */
    spi1_write_register(MPU_REG_ACCEL_CONFIG, 0x10);

    imu_delay_ms(10);
}


/* ============================================================
 * BMI270 INITIALIZATION (minimal — no config file)
 * ============================================================
 *
 * BMI270 requires loading an 8 kB config blob from Bosch before
 * it produces valid sensor data. Without it imu_read_raw() will
 * return zeroes for sensor fields.
 *
 * To enable full BMI270 support: download bmi270_config_file.h
 * from https://github.com/BoschSensortec/BMI270-Sensor-API and
 * define BMI270_CONFIG_LOADED.
 *
 * Burst layout from DATA_8 (0x0C), little-endian, needs 1 dummy byte:
 *   buf[0]      dummy
 *   buf[1..2]   ACCEL_X LSB, MSB
 *   buf[3..4]   ACCEL_Y LSB, MSB
 *   buf[5..6]   ACCEL_Z LSB, MSB
 *   buf[7..8]   GYRO_X LSB, MSB
 *   buf[9..10]  GYRO_Y LSB, MSB
 *   buf[11..12] GYRO_Z LSB, MSB
 * ============================================================ */

#define BMI270_REG_CMD          0x7E
#define BMI270_REG_PWR_CONF     0x7C
#define BMI270_REG_PWR_CTRL     0x7D
#define BMI270_REG_ACC_CONF     0x40
#define BMI270_REG_ACC_RANGE    0x41
#define BMI270_REG_GYR_CONF     0x42
#define BMI270_REG_GYR_RANGE    0x43
#define BMI270_REG_CHIP_ID      0x00
#define BMI270_REG_DATA_8       0x0C  /* ACC_X LSB */

static void bmi270_init(void)
{
    /* Soft reset */
    spi1_write_register(BMI270_REG_CMD, 0xB6);
    imu_delay_ms(2);

    /* Dummy read — first SPI read after reset returns garbage */
    (void)spi1_read_register(BMI270_REG_CHIP_ID);

    /* Disable advanced power save before any config writes */
    spi1_write_register(BMI270_REG_PWR_CONF, 0x00);
    imu_delay_ms(1);

    /* Accel: 100 Hz, normal mode, ±8 g */
    spi1_write_register(BMI270_REG_ACC_CONF,  0xA8);
    spi1_write_register(BMI270_REG_ACC_RANGE, 0x02);

    /* Gyro: 200 Hz, normal mode, ±2000 dps */
    spi1_write_register(BMI270_REG_GYR_CONF,  0xA9);
    spi1_write_register(BMI270_REG_GYR_RANGE, 0x00);

    /* Power on accel + gyro + temp */
    spi1_write_register(BMI270_REG_PWR_CTRL, 0x0E);
    imu_delay_ms(50);
}


/* ============================================================
 * imu_init — public entry point
 * ============================================================ */
void imu_init(imu_type_t imu)
{
    switch (imu) {
        case IMU_MPU6000:   mpu6000_init();   break;
        case IMU_ICM42688P: icm42688p_init(); break;
        case IMU_BMI270:    bmi270_init();    break;
        default: break;
    }
}


/* ============================================================
 * RAW DATA READS
 * ============================================================
 *
 * Single burst transaction per call — all 7 values come from
 * the same measurement cycle so there is no sample-tearing.
 * All chips are big-endian (high byte first) except BMI270
 * which is little-endian and requires a dummy lead byte.
 * ============================================================ */

static int icm42688p_read_raw(imu_raw_t *out)
{
    /* 14-byte burst from TEMP_DATA1 (0x1D):
     * [0..1]=TEMP  [2..3]=AX  [4..5]=AY  [6..7]=AZ
     * [8..9]=GX  [10..11]=GY  [12..13]=GZ
     * All big-endian (high byte first). */
    uint8_t buf[14];
    spi1_read_registers(ICM_REG_TEMP_DATA1, buf, 14);

    out->temp = (int16_t)((uint16_t)buf[0]  << 8 | buf[1]);
    out->ax   = (int16_t)((uint16_t)buf[2]  << 8 | buf[3]);
    out->ay   = (int16_t)((uint16_t)buf[4]  << 8 | buf[5]);
    out->az   = (int16_t)((uint16_t)buf[6]  << 8 | buf[7]);
    out->gx   = (int16_t)((uint16_t)buf[8]  << 8 | buf[9]);
    out->gy   = (int16_t)((uint16_t)buf[10] << 8 | buf[11]);
    out->gz   = (int16_t)((uint16_t)buf[12] << 8 | buf[13]);
    return 0;
}

static int mpu6000_read_raw(imu_raw_t *out)
{
    /* 14-byte burst from ACCEL_XOUT_H (0x3B):
     * [0..1]=AX  [2..3]=AY  [4..5]=AZ  [6..7]=TEMP
     * [8..9]=GX  [10..11]=GY  [12..13]=GZ
     * All big-endian. */
    uint8_t buf[14];
    spi1_read_registers(MPU_REG_ACCEL_XOUT_H, buf, 14);

    out->ax   = (int16_t)((uint16_t)buf[0]  << 8 | buf[1]);
    out->ay   = (int16_t)((uint16_t)buf[2]  << 8 | buf[3]);
    out->az   = (int16_t)((uint16_t)buf[4]  << 8 | buf[5]);
    out->temp = (int16_t)((uint16_t)buf[6]  << 8 | buf[7]);
    out->gx   = (int16_t)((uint16_t)buf[8]  << 8 | buf[9]);
    out->gy   = (int16_t)((uint16_t)buf[10] << 8 | buf[11]);
    out->gz   = (int16_t)((uint16_t)buf[12] << 8 | buf[13]);
    return 0;
}

static int bmi270_read_raw(imu_raw_t *out)
{
    /* 13-byte read from DATA_8 (0x0C): 1 dummy + 12 data.
     * Little-endian: LSB first.
     * buf[0]=dummy  buf[1..2]=AX  buf[3..4]=AY  buf[5..6]=AZ
     * buf[7..8]=GX  buf[9..10]=GY  buf[11..12]=GZ */
    uint8_t buf[13];
    spi1_read_registers(BMI270_REG_DATA_8, buf, 13);

    out->ax = (int16_t)((uint16_t)buf[2]  << 8 | buf[1]);
    out->ay = (int16_t)((uint16_t)buf[4]  << 8 | buf[3]);
    out->az = (int16_t)((uint16_t)buf[6]  << 8 | buf[5]);
    out->gx = (int16_t)((uint16_t)buf[8]  << 8 | buf[7]);
    out->gy = (int16_t)((uint16_t)buf[10] << 8 | buf[9]);
    out->gz = (int16_t)((uint16_t)buf[12] << 8 | buf[11]);

    /* BMI270 temp is at 0x22/0x23, little-endian, needs dummy byte */
    uint8_t tbuf[3];
    spi1_read_registers(0x22, tbuf, 3);
    out->temp = (int16_t)((uint16_t)tbuf[2] << 8 | tbuf[1]);
    return 0;
}


/* ============================================================
 * imu_read_raw — public entry point
 * ============================================================ */
int imu_read_raw(imu_type_t imu, imu_raw_t *out)
{
    switch (imu) {
        case IMU_MPU6000:   return mpu6000_read_raw(out);
        case IMU_ICM42688P: return icm42688p_read_raw(out);
        case IMU_BMI270:    return bmi270_read_raw(out);
        default:            return -1;
    }
}
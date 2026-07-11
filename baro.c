/*
 * baro.c — Barometer driver: DPS310/SPL06 + BMP280
 *
 * ── How a Barometer Measures Altitude ─────────────────────────
 *
 * Atmospheric pressure decreases with altitude. At sea level,
 * pressure is ~101325 Pa. For every 8.43 metres you go up,
 * pressure drops by ~100 Pa (1 hPa).
 *
 * We don't need absolute altitude — just height above our
 * takeoff point. So we capture the pressure at boot as our
 * "ground reference" and compute relative altitude from there:
 *
 *   altitude = (P_ground - P_current) / 12.015
 *
 * This linear approximation is accurate to <1% within ±100 m
 * of the reference point (plenty for a drone).
 *
 *
 * ── Vertical Velocity ─────────────────────────────────────────
 *
 * We differentiate altitude over time to get vertical velocity:
 *   v_raw = (alt_new - alt_old) / dt
 *
 * This is noisy, so we apply a simple IIR low-pass filter:
 *   v_filtered = α × v_raw + (1 - α) × v_filtered_prev
 *
 * With α = 0.3 and ~25 Hz update rate, the cutoff is ~1.5 Hz.
 *
 *
 * ── DPS310 (Infineon) / SPL06 (GoerTek) ──────────────────────
 *
 * The DPS310 is common on budget F405 boards (Sologood, etc.).
 * SPL06-001 is register-compatible (same driver works for both).
 *
 * Chip ID register 0x0D: product ID in upper nibble.
 *   0x10 = DPS310 rev 0, 0x11 = DPS310 rev 1
 *
 * Configuration:
 *   - Pressure: 32 measurements/sec, 16× oversampling
 *   - Temperature: 32 measurements/sec, 2× oversampling
 *   - Continuous pressure + temperature mode
 *
 * Calibration uses 9 coefficients (c0, c1, c00, c10, c01, c11,
 * c20, c21, c30) stored across registers 0x10–0x21. The
 * compensation formula is a second-order polynomial:
 *
 *   T_sc = raw_T / kT
 *   P_sc = raw_P / kP
 *   Temp(°C) = c0 * 0.5 + c1 * T_sc
 *   Press(Pa) = c00 + P_sc*(c10 + P_sc*(c20 + P_sc*c30))
 *                    + T_sc*c01 + T_sc*P_sc*(c11 + P_sc*c21)
 *
 * kT and kP are oversampling scale factors from the datasheet.
 *
 *
 * ── BMP280 (Bosch) ───────────────────────────────────────────
 *
 * Also supported if present. Uses the Bosch 32-bit integer
 * compensation algorithm from the datasheet.
 */

#include <stdint.h>
#include "baro.h"
#include "i2c.h"


/* ============================================================
 * COMMON CONSTANTS
 * ============================================================ */

/* I2C addresses (SDO pin selects) */
#define BARO_ADDR_LO    0x76
#define BARO_ADDR_HI    0x77

/* Velocity filter coefficient */
#define VVEL_ALPHA      0.1f

/* Read baro every N ticks (100 Hz / 4 = 25 Hz) */
#define BARO_DECIMATION 4

/* Pressure-to-altitude scale factor near sea level */
#define PA_TO_METERS    (1.0f / 12.015f)


/* ============================================================
 * COMMON STATE
 * ============================================================ */
static baro_type_t    chip_type;
static uint8_t        baro_addr;
static int            baro_detected;
static float          pressure_ref_pa;
static float          altitude_m;
static float          vvel_mps;
static float          prev_altitude_m;
static float          pressure_pa;
static float          temperature_c;
static int            ref_set;
static uint32_t       tick_counter;


/* ============================================================
 *
 *  ██████  ██████  ███████ ██████  ██  ██████
 *  ██   ██ ██   ██ ██           ██ ██ ██  ██ ██
 *  ██   ██ ██████  ███████  █████  ██ ██  ██
 *  ██   ██ ██           ██      ██ ██ ██  ██
 *  ██████  ██      ███████ ██████  ██  ██████
 *
 * ============================================================ */

/* ── DPS310 Register Map ──────────────────────────────────────── */
#define DPS_REG_PSR_B2      0x00    /* Pressure data [23:16] */
#define DPS_REG_PSR_B1      0x01
#define DPS_REG_PSR_B0      0x02
#define DPS_REG_TMP_B2      0x03    /* Temperature data [23:16] */
#define DPS_REG_TMP_B1      0x04
#define DPS_REG_TMP_B0      0x05
#define DPS_REG_PRS_CFG     0x06    /* Pressure rate + oversampling */
#define DPS_REG_TMP_CFG     0x07    /* Temp rate + oversampling + ext */
#define DPS_REG_MEAS_CFG    0x08    /* Operating mode + ready flags */
#define DPS_REG_CFG_REG     0x09    /* Bit shift, interrupt, FIFO */
#define DPS_REG_RESET       0x0C    /* Write 0x89 for soft reset */
#define DPS_REG_PRODUCT_ID  0x0D    /* Product + revision ID */
#define DPS_REG_COEF        0x10    /* Calibration coefficients (18 bytes) */
#define DPS_REG_COEF_SRCE   0x28    /* Temperature coefficient source */

/* Product ID: upper nibble = 0x1 for DPS310 */
#define DPS_PRODUCT_MASK    0xF0
#define DPS_PRODUCT_VAL     0x10

/* MEAS_CFG bits */
#define DPS_MEAS_COEF_RDY   (1U << 7)
#define DPS_MEAS_SENSOR_RDY (1U << 6)
#define DPS_MEAS_TMP_RDY    (1U << 5)
#define DPS_MEAS_PRS_RDY    (1U << 4)

/* Configuration values:
 *
 * PRS_CFG (0x06):
 *   PM_RATE [6:4] = 101 (32 meas/sec)
 *   PM_PRC  [3:0] = 0100 (16× oversampling)
 *   → 0b01010100 = 0x54
 *
 * TMP_CFG (0x07):
 *   TMP_EXT  [7]   = set from COEF_SRCE register
 *   TMP_RATE [6:4]  = 101 (32 meas/sec)
 *   TMP_PRC  [3:0]  = 0001 (2× oversampling)
 *   → 0b_1010001 = 0x51 | (ext << 7)
 *
 * CFG_REG (0x09):
 *   P_SHIFT [2] = 1 (required when pressure oversampling > 8×)
 *   T_SHIFT [3] = 0 (temperature oversampling ≤ 8×)
 *   → 0x04
 *
 * MEAS_CFG (0x08):
 *   MEAS_CTRL [2:0] = 111 (continuous pressure + temperature)
 *   → 0x07
 */
#define DPS_PRS_CFG_VAL     0x54
#define DPS_TMP_CFG_BASE    0x51    /* OR with (ext_flag << 7) */
#define DPS_CFG_REG_VAL     0x04
#define DPS_MEAS_CONTINUOUS 0x07

/* Oversampling scale factors (from DPS310 datasheet Table 9).
 *
 * Index = oversampling setting (0=1×, 1=2×, 2=4×, ... 7=128×).
 * These are used to convert raw ADC counts to scaled values
 * for the compensation formula.
 *
 * Our config: Pressure 16× → index 4, Temperature 2× → index 1
 */
#define DPS_KP  253952.0f       /* 16× oversampling scale factor */
#define DPS_KT  1572864.0f      /* 2× oversampling scale factor */


/* ── DPS310 Calibration Coefficients ──────────────────────────── */
static struct {
    int32_t c0, c1;             /* 12-bit signed */
    int32_t c00, c10;           /* 20-bit signed */
    int16_t c01, c11, c20, c21, c30;  /* 16-bit signed */
} dps_cal;


/* Sign-extend a value of 'bits' width to int32_t */
static int32_t sign_extend(uint32_t val, int bits)
{
    if (val & (1U << (bits - 1))) {
        val |= ~((1U << bits) - 1);
    }
    return (int32_t)val;
}


/* Parse DPS310 calibration from 18 raw bytes (reg 0x10–0x21).
 *
 * Byte layout (from datasheet):
 *   [0]  = c0[11:4]
 *   [1]  = c0[3:0]:c1[11:8]
 *   [2]  = c1[7:0]
 *   [3]  = c00[19:12]
 *   [4]  = c00[11:4]
 *   [5]  = c00[3:0]:c10[19:16]
 *   [6]  = c10[15:8]
 *   [7]  = c10[7:0]
 *   [8]  = c01[15:8]
 *   [9]  = c01[7:0]
 *   [10] = c11[15:8]
 *   [11] = c11[7:0]
 *   [12] = c20[15:8]
 *   [13] = c20[7:0]
 *   [14] = c21[15:8]
 *   [15] = c21[7:0]
 *   [16] = c30[15:8]
 *   [17] = c30[7:0]
 */
static void dps310_parse_coef(const uint8_t *c)
{
    dps_cal.c0  = sign_extend(((uint32_t)c[0] << 4) | (c[1] >> 4), 12);
    dps_cal.c1  = sign_extend(((uint32_t)(c[1] & 0x0F) << 8) | c[2], 12);
    dps_cal.c00 = sign_extend(((uint32_t)c[3] << 12) | ((uint32_t)c[4] << 4) | (c[5] >> 4), 20);
    dps_cal.c10 = sign_extend(((uint32_t)(c[5] & 0x0F) << 16) | ((uint32_t)c[6] << 8) | c[7], 20);
    dps_cal.c01 = (int16_t)(((uint16_t)c[8]  << 8) | c[9]);
    dps_cal.c11 = (int16_t)(((uint16_t)c[10] << 8) | c[11]);
    dps_cal.c20 = (int16_t)(((uint16_t)c[12] << 8) | c[13]);
    dps_cal.c21 = (int16_t)(((uint16_t)c[14] << 8) | c[15]);
    dps_cal.c30 = (int16_t)(((uint16_t)c[16] << 8) | c[17]);
}


/* DPS310 compensation: raw 24-bit values → pressure (Pa) and temperature (°C) */
static void dps310_compensate(int32_t raw_P, int32_t raw_T,
                              float *press_out, float *temp_out)
{
    float T_sc = (float)raw_T / DPS_KT;
    float P_sc = (float)raw_P / DPS_KP;

    *temp_out = (float)dps_cal.c0 * 0.5f + (float)dps_cal.c1 * T_sc;

    *press_out = (float)dps_cal.c00
               + P_sc * ((float)dps_cal.c10 + P_sc * ((float)dps_cal.c20 + P_sc * (float)dps_cal.c30))
               + T_sc * (float)dps_cal.c01
               + T_sc * P_sc * ((float)dps_cal.c11 + P_sc * (float)dps_cal.c21);
}


/* Detect and init DPS310. Returns 1 on success. */
static int dps310_detect_and_init(uint8_t addr, uint8_t *chip_id_out)
{
    uint8_t id = 0;
    if (i2c1_read_reg(addr, DPS_REG_PRODUCT_ID, &id) < 0) return 0;
    if ((id & DPS_PRODUCT_MASK) != DPS_PRODUCT_VAL) return 0;

    if (chip_id_out) *chip_id_out = id;
    baro_addr = addr;

    /* Soft reset */
    i2c1_write_reg(addr, DPS_REG_RESET, 0x89);
    for (volatile uint32_t d = 0; d < 200000; d++);  /* ~4 ms */

    /* Wait for sensor ready (coefficient + sensor) */
    for (int attempt = 0; attempt < 20; attempt++) {
        uint8_t status = 0;
        i2c1_read_reg(addr, DPS_REG_MEAS_CFG, &status);
        if ((status & (DPS_MEAS_COEF_RDY | DPS_MEAS_SENSOR_RDY)) ==
            (DPS_MEAS_COEF_RDY | DPS_MEAS_SENSOR_RDY)) {
            break;
        }
        for (volatile uint32_t d = 0; d < 50000; d++);  /* ~1 ms */
    }

    /* Read calibration coefficients (18 bytes from 0x10) */
    uint8_t cbuf[18];
    if (i2c1_read_burst(addr, DPS_REG_COEF, cbuf, 18) < 0) return 0;
    dps310_parse_coef(cbuf);

    /* Check which temperature sensor was used for calibration.
     * Bit 7 of COEF_SRCE (0x28) indicates external (1) vs internal (0).
     * We must use the same sensor for correct compensation. */
    uint8_t coef_srce = 0;
    i2c1_read_reg(addr, DPS_REG_COEF_SRCE, &coef_srce);
    uint8_t tmp_ext = (coef_srce >> 7) & 1;

    /* Configure measurement parameters */
    i2c1_write_reg(addr, DPS_REG_PRS_CFG, DPS_PRS_CFG_VAL);
    i2c1_write_reg(addr, DPS_REG_TMP_CFG, DPS_TMP_CFG_BASE | (tmp_ext << 7));
    i2c1_write_reg(addr, DPS_REG_CFG_REG, DPS_CFG_REG_VAL);

    /* Start continuous pressure + temperature measurement */
    i2c1_write_reg(addr, DPS_REG_MEAS_CFG, DPS_MEAS_CONTINUOUS);

    return 1;
}


/* Read DPS310 and compute pressure/temperature.
 * Returns 0 on success, -1 if data not ready or read failed. */
static int dps310_read(float *press_out, float *temp_out)
{
    /* Check if new data is available */
    uint8_t status = 0;
    i2c1_read_reg(baro_addr, DPS_REG_MEAS_CFG, &status);
    if (!(status & DPS_MEAS_PRS_RDY)) return -1;

    /* Read 6 bytes: pressure (3) + temperature (3) */
    uint8_t raw[6];
    if (i2c1_read_burst(baro_addr, DPS_REG_PSR_B2, raw, 6) < 0) return -1;

    /* Assemble 24-bit two's complement values */
    int32_t raw_P = sign_extend(
        ((uint32_t)raw[0] << 16) | ((uint32_t)raw[1] << 8) | raw[2], 24);
    int32_t raw_T = sign_extend(
        ((uint32_t)raw[3] << 16) | ((uint32_t)raw[4] << 8) | raw[5], 24);

    dps310_compensate(raw_P, raw_T, press_out, temp_out);
    return 0;
}


/* ============================================================
 *
 *  ██████  ███    ███ ██████  ██████   █████   ██████
 *  ██   ██ ████  ████ ██   ██      ██ ██   ██ ██  ██ ██
 *  ██████  ██ ████ ██ ██████   █████   █████  ██  ██
 *  ██   ██ ██  ██  ██ ██      ██      ██   ██ ██  ██
 *  ██████  ██      ██ ██      ███████  █████   ██████
 *
 * ============================================================ */

/* ── BMP280 Register Map ──────────────────────────────────────── */
#define BMP_REG_CALIB00     0x88    /* Calibration data (26 bytes) */
#define BMP_REG_CHIP_ID     0xD0    /* Should return 0x58 */
#define BMP_REG_RESET       0xE0    /* Write 0xB6 for soft reset */
#define BMP_REG_CTRL        0xF4    /* Oversampling + mode */
#define BMP_REG_CONFIG      0xF5    /* Standby, filter, SPI */
#define BMP_REG_DATA        0xF7    /* 6 bytes: press[3], temp[3] */

#define BMP_CHIP_ID_VAL     0x58

/* ctrl_meas: osrs_t=×2, osrs_p=×16, mode=normal → 0x57 */
#define BMP_CTRL_MEAS       0x57
/* config: t_sb=0.5ms, filter=4, no SPI → 0x08 */
#define BMP_CONFIG_VAL      0x08


/* ── BMP280 Calibration ──────────────────────────────────────── */
static struct {
    uint16_t T1;
    int16_t  T2, T3;
    uint16_t P1;
    int16_t  P2, P3, P4, P5, P6, P7, P8, P9;
} bmp_cal;

static int32_t bmp_t_fine;

static uint16_t le16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static int16_t  le16s(const uint8_t *p) { return (int16_t)le16(p); }


/* BMP280 compensation — Bosch datasheet 32-bit integer version */
static int32_t bmp280_compensate_temp(int32_t adc_T)
{
    int32_t var1 = ((((adc_T >> 3) - ((int32_t)bmp_cal.T1 << 1))) * ((int32_t)bmp_cal.T2)) >> 11;
    int32_t var2 = (((((adc_T >> 4) - ((int32_t)bmp_cal.T1)) *
                      ((adc_T >> 4) - ((int32_t)bmp_cal.T1))) >> 12) *
                    ((int32_t)bmp_cal.T3)) >> 14;
    bmp_t_fine = var1 + var2;
    return (bmp_t_fine * 5 + 128) >> 8;
}

static uint32_t bmp280_compensate_press(int32_t adc_P)
{
    int32_t var1 = ((int32_t)bmp_t_fine >> 1) - (int32_t)64000;
    int32_t var2 = (((var1 >> 2) * (var1 >> 2)) >> 11) * ((int32_t)bmp_cal.P6);
    var2 = var2 + ((var1 * ((int32_t)bmp_cal.P5)) << 1);
    var2 = (var2 >> 2) + (((int32_t)bmp_cal.P4) << 16);
    var1 = (((bmp_cal.P3 * (((var1 >> 2) * (var1 >> 2)) >> 13)) >> 3) +
            ((((int32_t)bmp_cal.P2) * var1) >> 1)) >> 18;
    var1 = ((((32768 + var1)) * ((int32_t)bmp_cal.P1)) >> 15);
    if (var1 == 0) return 0;

    uint32_t p = (uint32_t)((((int32_t)1048576 - adc_P) - (var2 >> 12))) * 3125;
    if (p < 0x80000000U)
        p = (p << 1) / (uint32_t)var1;
    else
        p = (p / (uint32_t)var1) * 2;

    var1 = (((int32_t)bmp_cal.P9) * ((int32_t)(((p >> 3) * (p >> 3)) >> 13))) >> 12;
    var2 = (((int32_t)(p >> 2)) * ((int32_t)bmp_cal.P8)) >> 13;
    p = (uint32_t)((int32_t)p + ((var1 + var2 + (int32_t)bmp_cal.P7) >> 4));
    return p;
}


/* Detect and init BMP280. Returns 1 on success. */
static int bmp280_detect_and_init(uint8_t addr, uint8_t *chip_id_out)
{
    uint8_t id = 0;
    if (i2c1_read_reg(addr, BMP_REG_CHIP_ID, &id) < 0) return 0;
    if (id != BMP_CHIP_ID_VAL) return 0;

    if (chip_id_out) *chip_id_out = id;
    baro_addr = addr;

    /* Soft reset */
    i2c1_write_reg(addr, BMP_REG_RESET, 0xB6);
    for (volatile uint32_t d = 0; d < 100000; d++);

    /* Read calibration (26 bytes) */
    uint8_t cbuf[26];
    if (i2c1_read_burst(addr, BMP_REG_CALIB00, cbuf, 26) < 0) return 0;

    bmp_cal.T1 = le16(&cbuf[0]);   bmp_cal.T2 = le16s(&cbuf[2]);
    bmp_cal.T3 = le16s(&cbuf[4]);  bmp_cal.P1 = le16(&cbuf[6]);
    bmp_cal.P2 = le16s(&cbuf[8]);  bmp_cal.P3 = le16s(&cbuf[10]);
    bmp_cal.P4 = le16s(&cbuf[12]); bmp_cal.P5 = le16s(&cbuf[14]);
    bmp_cal.P6 = le16s(&cbuf[16]); bmp_cal.P7 = le16s(&cbuf[18]);
    bmp_cal.P8 = le16s(&cbuf[20]); bmp_cal.P9 = le16s(&cbuf[22]);

    /* Configure (set config BEFORE ctrl_meas per datasheet §3.3) */
    i2c1_write_reg(addr, BMP_REG_CONFIG, BMP_CONFIG_VAL);
    i2c1_write_reg(addr, BMP_REG_CTRL, BMP_CTRL_MEAS);

    return 1;
}


/* Read BMP280, compute pressure/temperature.  Returns 0 or -1. */
static int bmp280_read(float *press_out, float *temp_out)
{
    uint8_t raw[6];
    if (i2c1_read_burst(baro_addr, BMP_REG_DATA, raw, 6) < 0) return -1;

    int32_t adc_P = ((int32_t)raw[0] << 12) | ((int32_t)raw[1] << 4) | (raw[2] >> 4);
    int32_t adc_T = ((int32_t)raw[3] << 12) | ((int32_t)raw[4] << 4) | (raw[5] >> 4);

    int32_t temp_raw  = bmp280_compensate_temp(adc_T);
    uint32_t press_raw = bmp280_compensate_press(adc_P);

    *temp_out  = (float)temp_raw / 100.0f;
    *press_out = (float)press_raw;
    return 0;
}


/* ============================================================
 *
 *   ██████  ██████  ███    ███ ███    ███  ██████  ███    ██
 *  ██      ██    ██ ████  ████ ████  ████ ██    ██ ████   ██
 *  ██      ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██
 *  ██      ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██
 *   ██████  ██████  ██      ██ ██      ██  ██████  ██   ████
 *
 * ============================================================ */


/* ============================================================
 * baro_init — Detect and configure whichever baro is present
 * ============================================================ */
baro_type_t baro_init(uint8_t *chip_id)
{
    baro_detected = 0;
    chip_type = BARO_NONE;

    /* Initialize state */
    altitude_m = 0.0f;
    prev_altitude_m = 0.0f;
    vvel_mps = 0.0f;
    pressure_pa = 0.0f;
    temperature_c = 0.0f;
    pressure_ref_pa = 0.0f;
    ref_set = 0;
    tick_counter = 0;

    /* Try DPS310 first (more common on budget FC boards) */
    if (dps310_detect_and_init(BARO_ADDR_LO, chip_id) ||
        dps310_detect_and_init(BARO_ADDR_HI, chip_id)) {
        chip_type = BARO_DPS310;
        baro_detected = 1;
        return BARO_DPS310;
    }

    /* Try BMP280 */
    if (bmp280_detect_and_init(BARO_ADDR_LO, chip_id) ||
        bmp280_detect_and_init(BARO_ADDR_HI, chip_id)) {
        chip_type = BARO_BMP280;
        baro_detected = 1;
        return BARO_BMP280;
    }

    return BARO_NONE;
}


/* ============================================================
 * baro_set_ground_reference — average 16 readings as baseline
 * ============================================================ */
void baro_set_ground_reference(void)
{
    if (!baro_detected) return;

    float sum = 0.0f;
    int good = 0;

    for (int i = 0; i < 16; i++) {
        float p = 0.0f, t = 0.0f;
        int ok;

        if (chip_type == BARO_DPS310)
            ok = dps310_read(&p, &t);
        else
            ok = bmp280_read(&p, &t);

        if (ok == 0 && p > 30000.0f && p < 120000.0f) {
            sum += p;
            good++;
        }

        /* Wait ~40 ms for next measurement */
        for (volatile uint32_t d = 0; d < 1500000; d++);
    }

    if (good > 0) {
        pressure_ref_pa = sum / (float)good;
        altitude_m = 0.0f;
        prev_altitude_m = 0.0f;
        vvel_mps = 0.0f;
        ref_set = 1;
    }
}


/* ============================================================
 * baro_update — Read sensor, update altitude + velocity
 * ============================================================ */
int baro_update(float dt)
{
    if (!baro_detected) return -1;

    tick_counter++;
    if (tick_counter < BARO_DECIMATION) return 0;
    tick_counter = 0;

    float baro_dt = dt * (float)BARO_DECIMATION;

    float p = 0.0f, t = 0.0f;
    int ok;

    if (chip_type == BARO_DPS310)
        ok = dps310_read(&p, &t);
    else
        ok = bmp280_read(&p, &t);

    if (ok < 0) return -1;

    temperature_c = t;
    pressure_pa = p;

    if (ref_set) {
        prev_altitude_m = altitude_m;
        altitude_m = (pressure_ref_pa - pressure_pa) * PA_TO_METERS;

        if (baro_dt > 0.001f) {
            float v_raw = (altitude_m - prev_altitude_m) / baro_dt;
            vvel_mps = VVEL_ALPHA * v_raw + (1.0f - VVEL_ALPHA) * vvel_mps;
        }
    }

    return 0;
}


/* ============================================================
 * GETTERS
 * ============================================================ */
float baro_get_altitude(void)    { return altitude_m; }
float baro_get_vvel(void)        { return vvel_mps; }
float baro_get_pressure(void)    { return pressure_pa; }
float baro_get_temperature(void) { return temperature_c; }

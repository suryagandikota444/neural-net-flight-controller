/*
 * main.c — STM32F405 Sologood FC firmware
 *
 * Revision 6: NN flight controller + BMP280 barometer + ESC calibration.
 *
 * ── Overview ────────────────────────────────────────────────────
 *
 * This firmware runs a trained neural network at 100 Hz to control
 * a quadrotor. The NN was trained in simulation (MuJoCo) using PPO
 * reinforcement learning. It takes 12 sensor observations as input
 * and outputs 4 motor commands.
 *
 * ── Sensors ─────────────────────────────────────────────────────
 *
 *   IMU (ICM-42688-P / MPU6000 / BMI270, SPI1):
 *     → gyro (rad/s), accel (g) → obs[0–5]
 *     → Madgwick AHRS → roll, pitch → obs[6–7]
 *
 *   Barometer (BMP280, I2C1 on PB8/PB9):
 *     → pressure → altitude error → obs[8]
 *     → filtered differentiation → vertical velocity → obs[9]
 *
 *   Not yet available:
 *     → obs[10–11] = horizontal velocity (needs optical flow / GPS)
 *
 * ── Control Loop Flow (every 10 ms) ────────────────────────────
 *
 *   ┌────────────┐    ┌────────────┐    ┌────────────┐
 *   │  1. READ   │───>│ 2. FILTER  │───>│ 3. BUILD   │
 *   │ IMU + BARO │    │  Madgwick  │    │   obs[12]  │
 *   │  raw data  │    │  AHRS +    │    │ normalized │
 *   │            │    │  altitude  │    │ sensor vec │
 *   └────────────┘    └────────────┘    └─────┬──────┘
 *                                             │
 *                                             ▼
 *   ┌────────────┐    ┌────────────┐    ┌────────────┐
 *   │  6. OUTPUT │<───│ 5. MAP TO  │<───│ 4. NN      │
 *   │    PWM     │    │   MOTORS   │    │  FORWARD   │
 *   │  1000-2000 │    │ hover+act  │    │  PASS      │
 *   │    µs      │    │ NN→BF map  │    │  ~15 µs    │
 *   └────────────┘    └────────────┘    └────────────┘
 *
 * ── Motor Mapping ───────────────────────────────────────────────
 *
 *      Front
 *   M4(FL)    M2(FR)       NN output mapping:
 *      \      /              nn[0] → M4(FL)
 *       \    /               nn[1] → M2(FR)
 *        ──                  nn[2] → M1(RR)
 *       /    \               nn[3] → M3(RL)
 *      /      \
 *   M3(RL)    M1(RR)
 *      Rear
 *
 * ── Axis Alignment ──────────────────────────────────────────────
 *
 * The Sologood F405 has the IMU mounted with a CW180° rotation.
 * CW180° about Z:
 *   Chip X → -Board X,  Chip Y → -Board Y,  Chip Z → +Board Z
 */

#include <stdint.h>
#include "clock.h"
#include "spi.h"
#include "imu.h"
#include "uart.h"
#include "motor.h"
#include "ahrs.h"
#include "nn.h"        /* Also provides NN_HOVER_THROTTLE, NN_ACTION_SCALE */
#include "i2c.h"
#include "baro.h"


/* ============================================================
 * BUILD MODES
 * ============================================================
 *
 * ESC_CAL_MODE: Set to 1 to run ESC throttle range calibration.
 *   Build with: make CFLAGS_EXTRA="-DESC_CAL_MODE=1"
 *   Or change the default below.
 *
 *   ** REMOVE ALL PROPELLERS BEFORE ESC CALIBRATION **
 *
 *   Sequence:
 *     1. Power off the drone
 *     2. Flash this firmware with ESC_CAL_MODE=1
 *     3. Power on — all ESCs get full throttle signal (2000 µs)
 *     4. Wait for ESC "high" beeps (~3 s)
 *     5. Firmware drops to idle (1000 µs)
 *     6. Wait for ESC "low" confirmation beeps (~3 s)
 *     7. Calibration done — LED stays on
 *     8. Power off, reflash with ESC_CAL_MODE=0 (normal firmware)
 */
#ifndef ESC_CAL_MODE
#define ESC_CAL_MODE 0
#endif


/* ============================================================
 * GPIO — LED on PC15 (directly driven, active-low on this board)
 * ============================================================ */
#define RCC_AHB1ENR          (*(volatile uint32_t *)0x40023830)
#define RCC_AHB1ENR_GPIOCEN  (1U << 2)
#define GPIOC_MODER          (*(volatile uint32_t *)0x40020800)
#define GPIOC_BSRR           (*(volatile uint32_t *)0x40020818)
#define LED_PIN              15

static void led_on(void)  { GPIOC_BSRR = (1U << (LED_PIN + 16)); }
static void led_off(void) { GPIOC_BSRR = (1U << LED_PIN); }


/* ============================================================
 * Delay (spin loop, ~5 cycles/iter at 168 MHz → 33600 iter/ms)
 * ============================================================ */
static void delay_ms(uint32_t ms)
{
    volatile uint32_t n = ms * 33600U;
    while (n--) { __asm volatile ("nop"); }
}


/* ============================================================
 * LED error pattern — rapid flash means something went wrong
 * ============================================================ */
static void blink_error(void)
{
    for (int i = 0; i < 10; i++) {
        led_on();  delay_ms(50);
        led_off(); delay_ms(50);
    }
    delay_ms(500);
}


/* ============================================================
 * SysTick — 100 Hz control loop timer (polling, no interrupts)
 * ============================================================ */
#define SYST_CSR    (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR    (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR    (*(volatile uint32_t *)0xE000E018)

#define SYST_CSR_ENABLE     (1U << 0)
#define SYST_CSR_CLKSOURCE  (1U << 2)
#define SYST_CSR_COUNTFLAG  (1U << 16)

static void systick_init_hz(uint32_t hz)
{
    SYST_RVR = (SYSCLK_HZ / hz) - 1U;
    SYST_CVR = 0;
    SYST_CSR = SYST_CSR_CLKSOURCE | SYST_CSR_ENABLE;
}

#ifndef LOOP_TIMING_DEBUG
#define LOOP_TIMING_DEBUG 1
#endif

#define LOOP_TIMING_INTERVAL 50

#define DEMCR                (*(volatile uint32_t *)0xE000EDFC)
#define DEMCR_TRCENA         (1U << 24)
#define DWT_CTRL             (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT           (*(volatile uint32_t *)0xE0001004)
#define DWT_CTRL_CYCCNTENA   (1U << 0)

static void dwt_cycle_counter_init(void)
{
    DEMCR |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

static inline uint32_t dwt_cycle_now(void)
{
    return DWT_CYCCNT;
}

static inline uint32_t dwt_cycle_delta(uint32_t t0, uint32_t t1)
{
    return t1 - t0;
}

static inline uint32_t cycles_to_us(uint32_t cycles)
{
    return (cycles + (SYSCLK_HZ / 2000000U)) / (SYSCLK_HZ / 1000000U);
}

static void print_loop_timing_us(uint32_t total_cycles,
                                 uint32_t imu_cycles,
                                 uint32_t conv_cycles,
                                 uint32_t ahrs_cycles,
                                 uint32_t baro_cycles,
                                 uint32_t obs_cycles,
                                 uint32_t nn_cycles,
                                 uint32_t map_cycles,
                                 uint32_t pwm_cycles)
{
    uart2_write_string("LT us tot:");
    uart2_write_uint32(cycles_to_us(total_cycles));
    uart2_write_string(" imu:");
    uart2_write_uint32(cycles_to_us(imu_cycles));
    uart2_write_string(" conv:");
    uart2_write_uint32(cycles_to_us(conv_cycles));
    uart2_write_string(" ahrs:");
    uart2_write_uint32(cycles_to_us(ahrs_cycles));
    uart2_write_string(" baro:");
    uart2_write_uint32(cycles_to_us(baro_cycles));
    uart2_write_string(" obs:");
    uart2_write_uint32(cycles_to_us(obs_cycles));
    uart2_write_string(" nn:");
    uart2_write_uint32(cycles_to_us(nn_cycles));
    uart2_write_string(" map:");
    uart2_write_uint32(cycles_to_us(map_cycles));
    uart2_write_string(" pwm:");
    uart2_write_uint32(cycles_to_us(pwm_cycles));
    uart2_write_string("\r\n");
}


/* ============================================================
 * SENSOR CONVERSION CONSTANTS
 * ============================================================ */
#define GYRO_RAW_TO_RADS   (1.0f / 939.65f)
#define MAX_GYRO_RADS      34.907f
#define ANGLE_OBS_SCALE    (1.0f / 45.0f)

/* Accel scale factor — set at runtime based on detected IMU:
 *   ICM42688P: ±16g → 2048 LSB/g → 1/2048
 *   MPU6000:   ±8g  → 4096 LSB/g → 1/4096
 *   BMI270:    ±8g  → 4096 LSB/g → 1/4096
 */
static float accel_raw_to_g;

/* Observation scaling for barometer (must match drone_env.py) */
#define ALT_SCALE          2.0f     /* ±2 m altitude error maps to ±1 */
#define VVEL_SCALE         5.0f     /* ±5 m/s vertical vel maps to ±1 */
#define HVEL_SCALE         3.0f     /* ±3 m/s horizontal vel maps to ±1 */


/* ============================================================
 * AXIS ALIGNMENT — Sologood F405 CW180° IMU rotation
 *
 * A 180° rotation around Z inverts BOTH X and Y axes.
 *   Chip X → -Board X  (forward becomes backward)
 *   Chip Y → -Board Y  (left becomes right)
 *   Chip Z → +Board Z  (up stays up)
 * ============================================================ */
#define ALIGN_AX(raw)  (-(raw))
#define ALIGN_AY(raw)  (-(raw))
#define ALIGN_AZ(raw)  ( (raw))
#define ALIGN_GX(raw)  (-(raw))
#define ALIGN_GY(raw)  (-(raw))
#define ALIGN_GZ(raw)  ( (raw))


/* ============================================================
 * NN-TO-MOTOR MAPPING (verified by motor identification test)
 * ============================================================ */
static const int NN_TO_MOTOR[4] = {
    3,   /* nn[0]=FL(CCW) → idx3 → PA9 → Front-Left  */
    2,   /* nn[1]=FR(CW)  → idx2 → PA8 → Front-Right */
    1,   /* nn[2]=RR(CCW) → idx1 → PC9 → Rear-Right  */
    0    /* nn[3]=RL(CW)  → idx0 → PC8 → Rear-Left   */
};


/* ============================================================
 * CONTROL LOOP CONFIGURATION
 * ============================================================ */
#define LOOP_HZ            100
#define LOOP_DT            (1.0f / LOOP_HZ)
#define SETTLE_TICKS       200        /* 2 seconds — AHRS + baro convergence */
#define TILT_LIMIT_DEG     60.0f      /* safety cutoff */
#define DEBUG_INTERVAL     50         /* telemetry at 2 Hz (0=off) */

/* Throttle ramp: gradually increase from idle to hover over RAMP_TICKS
 * after arming. This helps ESCs commit to motor startup. */
#define RAMP_TICKS         50         /* 0.5 seconds ramp after arming */


/* ============================================================
 * UTILITIES
 * ============================================================ */
static inline float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float my_fabsf(float x)
{
    return x < 0.0f ? -x : x;
}


/* ============================================================
 * DEBUG: Telemetry with motor PWM µs + baro data
 *
 * Output format (every 500 ms):
 *   R:-3.2 P:+1.5 | FL:1329 FR:1330 RR:1328 RL:1331 | A:+0.3 V:-0.1 P:101312
 *
 * A = altitude (m), V = vertical velocity (m/s), P = pressure (Pa)
 * ============================================================ */
static void print_fixed1(float val)
{
    if (val < 0.0f) { uart2_write_char('-'); val = -val; }
    else             { uart2_write_char('+'); }
    uint32_t whole = (uint32_t)val;
    uint32_t frac  = (uint32_t)((val - (float)whole) * 10.0f) % 10;
    uart2_write_uint32(whole);
    uart2_write_char('.');
    uart2_write_uint32(frac);
}

static void print_telemetry(float roll, float pitch,
                            const float motor_cmd[4],
                            float alt_m, float vvel_mps,
                            float pressure_pa, float temp_c,
                            int baro_ok)
{
    static const char *labels[4] = {"FL:", "FR:", "RR:", "RL:"};

    uart2_write_string("R:");
    print_fixed1(roll);
    uart2_write_string(" P:");
    print_fixed1(pitch);
    uart2_write_string(" | ");
    for (int i = 0; i < 4; i++) {
        uart2_write_string(labels[i]);
        uart2_write_uint32(1000 + (uint32_t)(motor_cmd[i] * 1000.0f));
        if (i < 3) uart2_write_char(' ');
    }

    if (baro_ok) {
        uart2_write_string(" | A:");
        print_fixed1(alt_m);
        uart2_write_string(" V:");
        print_fixed1(vvel_mps);
        uart2_write_string(" T:");
        print_fixed1(temp_c);
        uart2_write_string(" Pa:");
        uart2_write_uint32((uint32_t)pressure_pa);
    } else {
        uart2_write_string(" | BARO:none");
    }

    uart2_write_string("\r\n");
}


/* ============================================================
 * ESC CALIBRATION — Motor PWM register direct access
 * ============================================================
 *
 * For ESC calibration we need to set all 4 PWM outputs to
 * full throttle (2000 µs) before the ESCs boot. We write
 * the CCR registers directly for maximum speed at startup.
 */
#define TIM1_CCR1   (*(volatile uint32_t *)0x40010034)
#define TIM1_CCR2   (*(volatile uint32_t *)0x40010038)
#define TIM8_CCR3   (*(volatile uint32_t *)0x4001043C)
#define TIM8_CCR4   (*(volatile uint32_t *)0x40010440)

static void motors_full_throttle(void)
{
    TIM1_CCR1 = 2000;  /* PA8 — Front-Right */
    TIM1_CCR2 = 2000;  /* PA9 — Front-Left  */
    TIM8_CCR3 = 2000;  /* PC8 — Rear-Left   */
    TIM8_CCR4 = 2000;  /* PC9 — Rear-Right  */
}


/* ============================================================
 * MAIN
 * ============================================================ */
int main(void)
{
    /* ── 1. Clock (168 MHz from HSE) ────────────────────────── */
    int clock_ok = clock_init();

    /* ── 2. LED setup ───────────────────────────────────────── */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
    (void)RCC_AHB1ENR;
    GPIOC_MODER &= ~(3U << (LED_PIN * 2));
    GPIOC_MODER |=  (1U << (LED_PIN * 2));
    led_off();

    if (clock_ok != 0) {
        while (1) { blink_error(); }
    }

    /* ── 3. UART (debug output) ─────────────────────────────── */
    uart2_init(115200);
    uart2_write_string("\r\n=== FC BOOT (NN v2 + BARO) ===\r\n");
    uart2_write_string("Clock: 168MHz OK\r\n");

#if LOOP_TIMING_DEBUG
    dwt_cycle_counter_init();
    uart2_write_string("Loop timing: DWT cycle counter enabled\r\n");
#endif

    /* ── 4. SPI ─────────────────────────────────────────────── */
    delay_ms(100);
    spi1_init();
    uart2_write_string("SPI1: OK\r\n");
    delay_ms(10);

    /* ── 5. IMU detection & init ────────────────────────────── */
    uint8_t whoami = 0;
    imu_type_t imu = imu_detect(&whoami);

    uart2_write_string("IMU WHO_AM_I: ");
    uart2_write_hex8(whoami);
    uart2_write_string("\r\n");

    switch (imu) {
        case IMU_MPU6000:    uart2_write_string("IMU: MPU6000\r\n"); break;
        case IMU_ICM42688P:  uart2_write_string("IMU: ICM-42688-P\r\n"); break;
        case IMU_BMI270:     uart2_write_string("IMU: BMI270\r\n"); break;
        default:
            uart2_write_string("IMU: NOT DETECTED\r\n");
            while (1) { blink_error(); }
    }

    imu_init(imu);
    uart2_write_string("IMU: configured\r\n");

    /* Set accel scale factor based on detected IMU chip */
    switch (imu) {
        case IMU_ICM42688P:
            accel_raw_to_g = 1.0f / 2048.0f;   /* ±16g → 2048 LSB/g */
            uart2_write_string("Accel scale: 1/2048 (16g)\r\n");
            break;
        case IMU_MPU6000:
        case IMU_BMI270:
        default:
            accel_raw_to_g = 1.0f / 4096.0f;   /* ±8g → 4096 LSB/g */
            uart2_write_string("Accel scale: 1/4096 (8g)\r\n");
            break;
    }

    /* ── 6. Motor PWM init ──────────────────────────────────── */
    motor_init();
    uart2_write_string("Motors: PWM 400Hz, idle\r\n");

    /* ── 7. AHRS init ───────────────────────────────────────── */
    ahrs_init();
    uart2_write_string("AHRS: Madgwick filter ready\r\n");

    /* ── 8. I2C + Barometer ─────────────────────────────────── */
    i2c1_init();
    uart2_write_string("I2C1: 400kHz on PB8/PB9\r\n");

    /* I2C bus scan — find all devices on the bus */
    uart2_write_string("I2C scan: ");
    int i2c_found = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        if (i2c1_probe(addr) == 0) {
            uart2_write_string("0x");
            uart2_write_hex8(addr);

            /* Try to identify common baro chips */
            uint8_t id = 0;
            if (addr == 0x76 || addr == 0x77) {
                /* BMP280 chip ID register 0xD0 → 0x58 */
                /* BMP388 chip ID register 0x00 → 0x50 */
                /* DPS310 chip ID register 0x0D → 0x10 */
                i2c1_read_reg(addr, 0xD0, &id);
                if (id == 0x58) {
                    uart2_write_string("(BMP280) ");
                } else if (id == 0x50) {
                    uart2_write_string("(BMP388) ");
                } else {
                    i2c1_read_reg(addr, 0x0D, &id);
                    if (id == 0x10) {
                        uart2_write_string("(DPS310) ");
                    } else {
                        uart2_write_string("(id?0x");
                        uart2_write_hex8(id);
                        uart2_write_string(") ");
                    }
                }
            } else {
                uart2_write_string(" ");
            }
            i2c_found++;
        }
    }
    if (i2c_found == 0) {
        uart2_write_string("no devices — check wiring or try PB6/PB7");
    }
    uart2_write_string("\r\n");

    uint8_t baro_id = 0;
    baro_type_t baro = baro_init(&baro_id);
    int baro_ok = (baro != BARO_NONE);

    if (baro == BARO_DPS310) {
        uart2_write_string("BARO: DPS310 detected (ID ");
        uart2_write_hex8(baro_id);
        uart2_write_string(")\r\n");
    } else if (baro == BARO_BMP280) {
        uart2_write_string("BARO: BMP280 detected (ID ");
        uart2_write_hex8(baro_id);
        uart2_write_string(")\r\n");
    } else {
        uart2_write_string("BARO: not found (ID ");
        uart2_write_hex8(baro_id);
        uart2_write_string(") — obs[8:9] will be zero\r\n");
    }

#if ESC_CAL_MODE
    /* ============================================================
     * ESC THROTTLE RANGE CALIBRATION
     * ============================================================
     *
     * This teaches the ESCs what "full throttle" and "idle" mean.
     *
     * ** REMOVE ALL PROPELLERS BEFORE RUNNING THIS **
     *
     * Procedure:
     *   1. Firmware immediately outputs 2000 µs (full throttle)
     *   2. ESCs detect high signal on boot → enter cal mode
     *   3. Wait 4 s — ESCs beep to confirm max throttle captured
     *   4. Firmware drops to 1000 µs (idle)
     *   5. Wait 4 s — ESCs beep to confirm min throttle captured
     *   6. LED solid ON — calibration complete, power off
     *
     * After calibration, reflash with ESC_CAL_MODE=0.
     */
    uart2_write_string("\r\n");
    uart2_write_string("=====================================\r\n");
    uart2_write_string("  ESC CALIBRATION MODE\r\n");
    uart2_write_string("  REMOVE ALL PROPELLERS!\r\n");
    uart2_write_string("=====================================\r\n\r\n");

    /* Step 1: Full throttle — ESCs learn the maximum */
    uart2_write_string("Setting FULL THROTTLE (2000us)...\r\n");
    uart2_write_string("Listen for ESC high beeps.\r\n");
    motors_full_throttle();

    /* Blink LED rapidly while waiting — signals "active calibration" */
    for (int i = 0; i < 40; i++) {
        led_on();  delay_ms(50);
        led_off(); delay_ms(50);
    }

    /* Step 2: Drop to idle — ESCs learn the minimum */
    uart2_write_string("Setting IDLE (1000us)...\r\n");
    uart2_write_string("Listen for ESC confirmation beeps.\r\n");
    motor_all_stop();

    /* Slow blink while ESCs confirm */
    for (int i = 0; i < 8; i++) {
        led_on();  delay_ms(250);
        led_off(); delay_ms(250);
    }

    /* Done */
    uart2_write_string("\r\nESC CALIBRATION COMPLETE\r\n");
    uart2_write_string("Power off, reflash with ESC_CAL_MODE=0\r\n");
    led_on();
    while (1) { /* halt */ }

#else /* Normal flight mode */

    /* ── 9. ESC arming (3 seconds of idle PWM) ──────────────── */
    uart2_write_string("ESC arming: 3s idle signal...\r\n");
    delay_ms(3000);
    uart2_write_string("ESC arming: done\r\n");

    /* ── 10. Barometer ground reference ─────────────────────── */
    if (baro_ok) {
        uart2_write_string("BARO: settling 2s...\r\n");

        /* Let the BMP280 IIR filter settle by reading for 2 seconds */
        for (int i = 0; i < 50; i++) {
            baro_update(0.04f);
            delay_ms(40);
        }

        baro_set_ground_reference();
        uart2_write_string("BARO: ground reference set (P=");
        uart2_write_uint32((uint32_t)baro_get_pressure());
        uart2_write_string(" Pa, T=");
        print_fixed1(baro_get_temperature());
        uart2_write_string(" C)\r\n");
    }

    /* ── 11. Start SysTick for 100 Hz ───────────────────────── */
    systick_init_hz(LOOP_HZ);
    uart2_write_string("Control loop: 100Hz starting\r\n");
    uart2_write_string("AHRS settling 2s (keep still)...\r\n");
    led_on();

    /* ── 12. Main control loop ──────────────────────────────── */

    imu_raw_t  sample;
    float      motor_cmd[4] = {0};
    uint32_t   tick_count = 0;
    uint32_t   debug_counter = 0;
    int        armed = 0;
    uint32_t   ramp_counter = 0;
#if LOOP_TIMING_DEBUG
    uint32_t   timing_counter = 0;
#endif

    while (1) {

        /* Wait for next 100 Hz tick (poll SysTick COUNTFLAG) */
        if (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {
            continue;
        }

        tick_count++;

    #if LOOP_TIMING_DEBUG
        uint32_t t0 = dwt_cycle_now();
    #endif

        /* ── A: Read IMU (SPI burst, ~10 µs) ──────────────── */
        imu_read_raw(imu, &sample);

    #if LOOP_TIMING_DEBUG
        uint32_t t1 = dwt_cycle_now();
    #endif

        /* ── B: Axis alignment (CW180°) + unit conversion ──── */
        float gx = (float)ALIGN_GX(sample.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(sample.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(sample.gz) * GYRO_RAW_TO_RADS;

        float ax = (float)ALIGN_AX(sample.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(sample.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(sample.az) * accel_raw_to_g;

    #if LOOP_TIMING_DEBUG
        uint32_t t2 = dwt_cycle_now();
    #endif

        /* ── C: AHRS update (Madgwick, ~5 µs) ──────────────── */
        ahrs_update(gx, gy, gz, ax, ay, az, LOOP_DT);

    #if LOOP_TIMING_DEBUG
        uint32_t t3 = dwt_cycle_now();
    #endif

        float roll_deg  = ahrs_get_roll();
        float pitch_deg = ahrs_get_pitch();

        /* ── C2: Barometer update (~25 Hz, decimated internally) */
        if (baro_ok) {
            baro_update(LOOP_DT);
        }

    #if LOOP_TIMING_DEBUG
        uint32_t t4 = dwt_cycle_now();
    #endif

        /* ── D: Settling check ─────────────────────────────── */
        if (!armed) {
            if (tick_count >= SETTLE_TICKS) {
                armed = 1;
                ramp_counter = 0;
                uart2_write_string("ARMED — motors live (ramping)\r\n");
            }
            continue;
        }

        /* ── E: Safety — tilt limit ───────────────────────── */
        if (my_fabsf(roll_deg) > TILT_LIMIT_DEG ||
            my_fabsf(pitch_deg) > TILT_LIMIT_DEG)
        {
            motor_all_stop();
            continue;
        }

        /* ── F: Build observation vector (matches drone_env.py) */
        float obs[12];

        /* Gyro (normalized by MAX_GYRO) */
        obs[0] = clampf(gx / MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / MAX_GYRO_RADS, -1.0f, 1.0f);

        /* Accel (normalized by 2g) */
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);

        /* Attitude error (roll/pitch in degrees, scaled by 45°) */
        obs[6] = clampf(roll_deg  * ANGLE_OBS_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch_deg * ANGLE_OBS_SCALE, -1.0f, 1.0f);

        /* Altitude error and vertical velocity from barometer */
        if (baro_ok) {
            float alt_err = baro_get_altitude();   /* positive = above ground ref */
            float vvel    = baro_get_vvel();        /* positive = going up */
            obs[8]  = clampf(alt_err / ALT_SCALE, -1.0f, 1.0f);
            obs[9]  = clampf(vvel / VVEL_SCALE, -1.0f, 1.0f);
        } else {
            obs[8]  = 0.0f;
            obs[9]  = 0.0f;
        }

        /* Horizontal velocity — not available without optical flow / GPS */
        obs[10] = 0.0f;
        obs[11] = 0.0f;

    #if LOOP_TIMING_DEBUG
        uint32_t t5 = dwt_cycle_now();
    #endif

        /* ── G: NN forward pass (~15 µs) ──────────────────── */
        float nn_out[4];
        nn_forward(obs, nn_out);

    #if LOOP_TIMING_DEBUG
        uint32_t t6 = dwt_cycle_now();
    #endif

        /* ── H: Residual action → motor command ──────────── */
        for (int i = 0; i < 4; i++) {
            motor_cmd[i] = clampf(
                NN_HOVER_THROTTLE + nn_out[i] * NN_ACTION_SCALE,
                0.0f, 1.0f
            );
        }

        /* ── H2: Throttle ramp after arming ─────────────────
         *
         * Gradually ramp from idle to full NN output over RAMP_TICKS.
         * This helps ESCs commit to motor startup instead of
         * jumping from 1000 µs to ~1322 µs instantly.
         *
         * ramp_frac goes from 0.0 → 1.0 over RAMP_TICKS ticks.
         * motor_cmd = ramp_frac × nn_cmd
         */
        if (ramp_counter < RAMP_TICKS) {
            ramp_counter++;
            float ramp_frac = (float)ramp_counter / (float)RAMP_TICKS;
            for (int i = 0; i < 4; i++) {
                motor_cmd[i] *= ramp_frac;
            }
        }

#if LOOP_TIMING_DEBUG
        uint32_t t7 = dwt_cycle_now();
#endif

        /* ── I: Write to motors (NN→Betaflight mapping) ─── */
        for (int i = 0; i < 4; i++) {
            motor_set(NN_TO_MOTOR[i], motor_cmd[i]);
        }

#if LOOP_TIMING_DEBUG
        uint32_t t8 = dwt_cycle_now();
        if (++timing_counter >= LOOP_TIMING_INTERVAL) {
            timing_counter = 0;
            print_loop_timing_us(
                dwt_cycle_delta(t0, t8),
                dwt_cycle_delta(t0, t1),
                dwt_cycle_delta(t1, t2),
                dwt_cycle_delta(t2, t3),
                dwt_cycle_delta(t3, t4),
                dwt_cycle_delta(t4, t5),
                dwt_cycle_delta(t5, t6),
                dwt_cycle_delta(t6, t7),
                dwt_cycle_delta(t7, t8)
            );
        }
#endif

        /* ── J: Debug telemetry at 2 Hz ────────────────────── */
        if (DEBUG_INTERVAL > 0 && ++debug_counter >= DEBUG_INTERVAL) {
            debug_counter = 0;
            print_telemetry(roll_deg, pitch_deg, motor_cmd,
                            baro_get_altitude(), baro_get_vvel(),
                            baro_get_pressure(), baro_get_temperature(),
                            baro_ok);
            led_off(); delay_ms(2); led_on();
        }
    }

#endif /* ESC_CAL_MODE */

    return 0;
}

/*
 * TODO: Future improvements
 * 1. Optical flow / GPS → obs[10], obs[11]
 * 2. DMA SPI reads + IMU data-ready interrupt
 * 3. DShot motor protocol
 * 4. UART DMA for non-blocking debug output
 * 5. Proper arm/disarm via UART command or RC receiver
 * 6. In-flight tuning of NN_HOVER_THROTTLE
 * 7. Complementary filter for vvel (fuse baro + accel Z)
 */

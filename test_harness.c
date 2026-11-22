/*
 * test_harness.c — Systematic hardware validation for drone FC
 *
 * ── BUILD / FLASH ──────────────────────────────────────────────
 *
 *   make flash-test                 passive tests only (default)
 *   make flash-test TESTS=3,8       just tests 3 and 8
 *   make flash-test TESTS=all       everything, INCLUDING motor tests
 *
 *   "passive" = 1-9 and 11-13. Tests 10 and 14 drive the motors and
 *   only run when you name them explicitly (TESTS=10, TESTS=14 or
 *   TESTS=all). TEST 15 (RC link, interactive, needs the radio on) is
 *   also opt-in: make flash-test TESTS=15. Even so: bench-test with the battery unplugged unless
 *   you are deliberately running a motor test on a secured rig.
 *
 * ── USAGE ──────────────────────────────────────────────────────
 *
 *   1. Open serial first: screen -L /dev/cu.SLAB_USBtoUART 115200
 *   2. Flash (BOOT + USB, then make flash-test ...). Tests start at once.
 *   3. Follow the prompts. Tilt during the "Preparing in 3...2...1"
 *      countdown — the AHRS keeps tracking throughout.
 *
 *      TEST 1:  SPI register readback      (immediate)
 *      TEST 2:  Static IMU readings        (5 s, FLAT and STILL)
 *      TEST 3:  Axis signs                 (nose down, right down, CW)
 *      TEST 4:  AHRS convergence           (right down, then flat)
 *      TEST 5:  Observation vector         (FLAT and STILL)
 *      TEST 6:  NN inference vs golden     (immediate)
 *      TEST 7:  Full control loop, motors off (10 s, FLAT and STILL)
 *      TEST 8:  Tilt → motor response      (right down, nose down, nose up)
 *      TEST 9:  Action filter lag          (right down)
 *      TEST 10: Motor wiring  ** SPINS MOTORS ** (opt-in)
 *      TEST 11: Action filter maths        (immediate)
 *      TEST 12: Loop timing                (5 s, FLAT)
 *      TEST 13: Baro → throttle noise      (8 s, FLAT)
 *      TEST 14: Throttle staircase ** SPINS MOTORS, PROPS ON ** (opt-in)
 *      TEST 15: RC link + arming logic, motors OFF  (opt-in, 60 s, radio ON)
 *      TEST 16: Thrust ramp on a scale ** PROPS ON, ARM switch = dead-man ** (opt-in)
 *      TEST 17: RC channel finder — which switch is which channel, motors OFF (opt-in, 90 s)
 *      TEST 18: Level calibration → flash (opt-in; flat surface, rotate 180° when asked)
 *
 * ── CONVENTIONS ────────────────────────────────────────────────
 *
 *   Body frame = training (drone_env.py) NWU: X forward, Y left, Z up.
 *     nose DOWN  → pitch > 0, ax < 0     right DOWN → roll > 0, ay > 0
 *     CW from above → gz < 0
 *   Correcting a tilt means raising the LOW side:
 *     nose down → FRONT motors up; right down → RIGHT motors up.
 *
 *   Axis alignment, observation scaling, altitude gains and the action
 *   filter come from fc_config.h — the same header main.c uses.
 */

#include <stdint.h>
#include <stddef.h>    /* NULL */
#include "clock.h"
#include "spi.h"
#include "imu.h"
#include "uart.h"
#include "ahrs.h"
#include "nn.h"        /* NN_HOVER_THROTTLE, NN_ACTION_SCALE, nn_forward() */
#include "i2c.h"
#include "baro.h"
#include "motor.h"     /* motor_init, motor_set, motor_all_stop for tests 10/14 */
#include "fc_config.h" /* shared with main.c: ALIGN_xx, obs scales, altitude gains, filter */
#include "host_test/golden_vectors.h"  /* test 6 reference outputs, generated from weights.h */
#include "crsf.h"      /* test 15 */
#include "rc.h"
#include "rc_input.h"
#include "level_cal.h" /* test 18 */

/* ============================================================
 * Test selection — set at build time by the Makefile:
 *   make flash-test TESTS=3,8     → HARNESS_TESTS="3,8"
 * "passive" (default) = every test except the motor tests 10 and 14.
 * "all" = every test. Motor tests never run unless explicitly listed.
 * ============================================================ */
#ifndef HARNESS_TESTS
#define HARNESS_TESTS "passive"
#endif

static int str_eq(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}

static int test_selected(int n)
{
    const char *s = HARNESS_TESTS;
    if (str_eq(s, "all"))     return 1;
    if (str_eq(s, "passive")) return n != 10 && n != 14 && n != 15 && n != 16 && n != 17 && n != 18;
    /* comma-separated list of numbers, e.g. "3,7,8" */
    while (*s) {
        int v = 0, digits = 0;
        while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; digits++; }
        if (digits && v == n) return 1;
        while (*s && (*s < '0' || *s > '9')) s++;
    }
    return 0;
}


/* ============================================================
 * LED on PC15 (same as main.c)
 * ============================================================ */
#define RCC_AHB1ENR          (*(volatile uint32_t *)0x40023830)
#define RCC_AHB1ENR_GPIOCEN  (1U << 2)
#define GPIOC_MODER          (*(volatile uint32_t *)0x40020800)
#define GPIOC_BSRR           (*(volatile uint32_t *)0x40020818)
#define LED_PIN              15

static void led_on(void)  { GPIOC_BSRR = (1U << (LED_PIN + 16)); }
static void led_off(void) { GPIOC_BSRR = (1U << LED_PIN); }


/* ============================================================
 * Delay (same as main.c — spin loop at 168 MHz)
 * ============================================================ */
static void delay_ms(uint32_t ms)
{
    volatile uint32_t n = ms * 33600U;
    while (n--) { __asm volatile ("nop"); }
}


/* ============================================================
 * SysTick — 100 Hz tick for AHRS timing
 * ============================================================ */
#define SYST_CSR    (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR    (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR    (*(volatile uint32_t *)0xE000E018)

#define SYST_CSR_ENABLE     (1U << 0)
#define SYST_CSR_CLKSOURCE  (1U << 2)
#define SYST_CSR_COUNTFLAG  (1U << 16)

static void systick_init_100hz(void)
{
    SYST_RVR = (SYSCLK_HZ / 100) - 1U;
    SYST_CVR = 0;
    SYST_CSR = SYST_CSR_CLKSOURCE | SYST_CSR_ENABLE;
}


/* ============================================================
 * DWT cycle counter — used by test 12 for sub-microsecond timing
 * ============================================================ */
#define DEMCR              (*(volatile uint32_t *)0xE000EDFC)
#define DEMCR_TRCENA       (1U << 24)
#define DWT_CTRL           (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT         (*(volatile uint32_t *)0xE0001004)
#define DWT_CTRL_CYCCNTENA (1U << 0)

static void dwt_init(void)
{
    DEMCR    |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

static inline uint32_t dwt_now(void)
{
    return DWT_CYCCNT;
}

/* Convert DWT cycle delta to microseconds (168 MHz clock) */
static inline uint32_t cycles_to_us(uint32_t cycles)
{
    /* (cycles + half-divisor) / divisor  — rounds to nearest µs */
    return (cycles + (SYSCLK_HZ / 2000000U)) / (SYSCLK_HZ / 1000000U);
}


/* ============================================================
 * Float printer — 3 decimal places with sign
 * (copied from main.c for self-containment)
 * ============================================================ */
static void print_float3(float val)
{
    if (val < 0.0f) { uart_write_char('-'); val = -val; }
    else             { uart_write_char('+'); }
    uint32_t whole = (uint32_t)val;
    uint32_t frac  = (uint32_t)((val - (float)whole) * 1000.0f + 0.5f);
    if (frac >= 1000) { whole++; frac -= 1000; }
    uart_write_uint32(whole);
    uart_write_char('.');
    if (frac < 100) uart_write_char('0');
    if (frac < 10)  uart_write_char('0');
    uart_write_uint32(frac);
}


/* ============================================================
 * Axis alignment (ALIGN_xx), GYRO_RAW_TO_RADS and MAX_GYRO_RADS come
 * from fc_config.h — the same definitions main.c compiles with.
 * ============================================================ */

/* Accel scale: set at runtime based on detected IMU */
static float accel_raw_to_g;

/* ============================================================
 * ICM-42688-P register addresses for readback
 * ============================================================ */
#define ICM_REG_GYRO_CONFIG0         0x4F
#define ICM_REG_ACCEL_CONFIG0        0x50
#define ICM_REG_PWR_MGMT0            0x4E
#define ICM_REG_GYRO_ACCEL_CONFIG0   0x52
#define ICM_REG_BANK_SEL             0x76

/* ============================================================
 * Separator / banner helpers
 * ============================================================ */
static void print_banner(const char *text)
{
    uart_write_string("\r\n");
    uart_write_string("========================================\r\n");
    uart_write_string(text);
    uart_write_string("\r\n");
    uart_write_string("========================================\r\n");
}

static void print_pass(const char *label)
{
    uart_write_string("  [PASS] ");
    uart_write_string(label);
    uart_write_string("\r\n");
}

static void print_fail(const char *label)
{
    uart_write_string("  [FAIL] ");
    uart_write_string(label);
    uart_write_string("\r\n");
}

static void print_regcheck(const char *name, uint8_t addr,
                           uint8_t got, uint8_t expected)
{
    uart_write_string("  ");
    uart_write_string(name);
    uart_write_string(" (");            /* uart_write_hex8 prints its own "0x" */
    uart_write_hex8(addr);
    uart_write_string(") = ");
    uart_write_hex8(got);
    uart_write_string("  expected ");
    uart_write_hex8(expected);
    if (got == expected) {
        uart_write_string("  OK\r\n");
    } else {
        uart_write_string("  ** MISMATCH **\r\n");
    }
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██
 *     ██    ██      ██         ██    ███
 *     ██    █████   ███████    ██     ██
 *     ██    ██           ██    ██     ██
 *     ██    ███████ ███████    ██     ██
 *
 *  SPI REGISTER READBACK
 *
 *  Purpose: Verify the IMU accepted our configuration writes.
 *           If ANY register mismatches, the SPI bus or init
 *           sequence is broken and nothing else matters.
 *
 *  Expected output for ICM-42688-P:
 *    GYRO_CONFIG0  (0x4F) = 0x06  (±2000 dps, 1 kHz ODR)
 *    ACCEL_CONFIG0 (0x50) = 0x06  (±16g, 1 kHz ODR)
 *    PWR_MGMT0     (0x4E) = 0x0F  (gyro+accel low-noise)
 *    GYRO_ACCEL_CFG(0x52) = 0x06  (50 Hz gyro UI LPF)
 *
 *  Expected output for MPU6000:
 *    GYRO_CONFIG   (0x1B) = 0x18  (±2000 dps)
 *    ACCEL_CONFIG  (0x1C) = 0x10  (±8g)
 *    PWR_MGMT_1    (0x6B) = 0x01  (PLL with X gyro ref)
 *    CONFIG        (0x1A) = 0x00  (DLPF off)
 *
 * ============================================================ */
static void test1_spi_readback(imu_type_t imu)
{
    print_banner("TEST 1: SPI Register Readback");
    uart_write_string("Verifying IMU accepted configuration writes.\r\n");
    uart_write_string("Any MISMATCH means SPI bus or init is broken.\r\n\r\n");

    int pass_count = 0;
    int total = 0;

    if (imu == IMU_ICM42688P) {
        uart_write_string("Chip: ICM-42688-P\r\n\r\n");

        uint8_t v;

        v = spi1_read_register(ICM_REG_GYRO_CONFIG0);
        print_regcheck("GYRO_CONFIG0 ", ICM_REG_GYRO_CONFIG0, v, 0x06);
        if (v == 0x06) pass_count++;
        total++;

        v = spi1_read_register(ICM_REG_ACCEL_CONFIG0);
        print_regcheck("ACCEL_CONFIG0", ICM_REG_ACCEL_CONFIG0, v, 0x06);
        if (v == 0x06) pass_count++;
        total++;

        v = spi1_read_register(ICM_REG_PWR_MGMT0);
        print_regcheck("PWR_MGMT0    ", ICM_REG_PWR_MGMT0, v, 0x0F);
        if (v == 0x0F) pass_count++;
        total++;

        v = spi1_read_register(ICM_REG_GYRO_ACCEL_CONFIG0);
        print_regcheck("GYRO_ACCEL_CF", ICM_REG_GYRO_ACCEL_CONFIG0, v, 0x76);  /* gyro 50 Hz, accel 25 Hz (imu.c) */
        if (v == 0x06) pass_count++;
        total++;

    } else {
        uart_write_string("  NO IMU DETECTED — SPI bus is dead.\r\n");
        return;
    }

    uart_write_string("\r\n  Result: ");
    uart_write_uint32(pass_count);
    uart_write_string("/");
    uart_write_uint32(total);
    uart_write_string(" registers match.\r\n");
    if (pass_count == total)
        print_pass("All config registers verified");
    else
        print_fail("Register mismatch — SPI write or init broken");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████ ██████
 *     ██    ██      ██         ██         ██
 *     ██    █████   ███████    ██     █████
 *     ██    ██           ██    ██    ██
 *     ██    ███████ ███████    ██    ███████
 *
 *  STATIC IMU CHARACTERIZATION
 *
 *  Purpose: Board flat and still. Verify:
 *    - Accel Z reads +1g (not -1g, not 0.5g)
 *    - Accel X,Y read ~0
 *    - Gyro X,Y,Z read ~0
 *    - Scale factors and axis alignment are correct
 *
 *  Procedure: Place board flat on table, press reset, don't touch.
 *
 *  Expected output (NWU convention, board flat):
 *    ax_avg ≈  0.00g   (±0.05)
 *    ay_avg ≈  0.00g   (±0.05)
 *    az_avg ≈ +1.00g   (±0.05)    ← THIS IS THE BIG ONE
 *    gx_avg ≈  0.00    (±0.02 rad/s)
 *    gy_avg ≈  0.00    (±0.02 rad/s)
 *    gz_avg ≈  0.00    (±0.02 rad/s)
 *
 *  If az ≈ -1.0:  Z axis is flipped (ALIGN_AZ sign wrong)
 *  If az ≈ +0.5:  Scale factor is wrong (using ±16g divisor
 *                 on a chip configured for ±8g, or vice versa)
 *  If ax or ay ≈ ±1.0: Axes are swapped (X/Y/Z assignment wrong)
 *
 * ============================================================ */
static void test2_static_imu(imu_type_t imu)
{
    print_banner("TEST 2: Static IMU Characterization");
    uart_write_string("Board must be FLAT and STILL for 5 seconds.\r\n");
    uart_write_string("Starting in 3 seconds...\r\n");
    delay_ms(3000);
    uart_write_string("Sampling...\r\n\r\n");

    /* Accumulate 500 samples (5 seconds at 100 Hz) */
    #define STATIC_SAMPLES 500

    float sum_ax = 0, sum_ay = 0, sum_az = 0;
    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    float min_az = 999.0f, max_az = -999.0f;

    /* Also track raw int16 values for sanity */
    int32_t sum_raw_az = 0;

    systick_init_100hz();

    for (int i = 0; i < STATIC_SAMPLES; i++) {
        /* Wait for tick */
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        imu_raw_t sample;
        imu_read_raw(imu, &sample);

        /* Apply axis alignment + scale (same as main.c) */
        float gx = (float)ALIGN_GX(sample.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(sample.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(sample.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(sample.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(sample.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(sample.az) * accel_raw_to_g;

        sum_ax += ax;  sum_ay += ay;  sum_az += az;
        sum_gx += gx;  sum_gy += gy;  sum_gz += gz;
        sum_raw_az += (int32_t)ALIGN_AZ(sample.az);

        if (az < min_az) min_az = az;
        if (az > max_az) max_az = az;

        /* Print a sample every 50 ticks (2 Hz) so you can watch */
        if (i % 50 == 0) {
            uart_write_string("  [");
            uart_write_uint32(i);
            uart_write_string("] ax=");
            print_float3(ax);
            uart_write_string(" ay=");
            print_float3(ay);
            uart_write_string(" az=");
            print_float3(az);
            uart_write_string(" | gx=");
            print_float3(gx);
            uart_write_string(" gy=");
            print_float3(gy);
            uart_write_string(" gz=");
            print_float3(gz);
            uart_write_string(" | raw_az=");
            uart_write_int32(ALIGN_AZ(sample.az));
            uart_write_string("\r\n");
        }
    }

    /* Compute averages */
    float n = (float)STATIC_SAMPLES;
    float avg_ax = sum_ax / n;
    float avg_ay = sum_ay / n;
    float avg_az = sum_az / n;
    float avg_gx = sum_gx / n;
    float avg_gy = sum_gy / n;
    float avg_gz = sum_gz / n;
    int32_t avg_raw_az = sum_raw_az / STATIC_SAMPLES;

    uart_write_string("\r\n---- STATIC RESULTS (500 samples, 5 sec) ----\r\n\r\n");

    uart_write_string("  Accel averages (g):\r\n");
    uart_write_string("    ax_avg = "); print_float3(avg_ax);
    uart_write_string("   (expect ~0.00)\r\n");
    uart_write_string("    ay_avg = "); print_float3(avg_ay);
    uart_write_string("   (expect ~0.00)\r\n");
    uart_write_string("    az_avg = "); print_float3(avg_az);
    uart_write_string("   (expect ~+1.00)\r\n");
    uart_write_string("    az_min = "); print_float3(min_az);
    uart_write_string("   az_max = "); print_float3(max_az);
    uart_write_string("\r\n");
    uart_write_string("    raw_az_avg = ");
    uart_write_int32(avg_raw_az);
    uart_write_string("   (ICM42688: expect ~+2048, MPU6000: expect ~+4096)\r\n");

    uart_write_string("\r\n  Gyro averages (rad/s):\r\n");
    uart_write_string("    gx_avg = "); print_float3(avg_gx);
    uart_write_string("   (expect ~0.00)\r\n");
    uart_write_string("    gy_avg = "); print_float3(avg_gy);
    uart_write_string("   (expect ~0.00)\r\n");
    uart_write_string("    gz_avg = "); print_float3(avg_gz);
    uart_write_string("   (expect ~0.00)\r\n");

    /* ---- Auto-check results ---- */
    uart_write_string("\r\n---- DIAGNOSTICS ----\r\n\r\n");

    /* Helper: fabsf without libm */
    #define FABSF(x) ((x) < 0.0f ? -(x) : (x))

    /* Check az ≈ +1.0 */
    if (FABSF(avg_az - 1.0f) < 0.10f)
        print_pass("az ≈ +1.0g — gravity direction correct");
    else if (FABSF(avg_az + 1.0f) < 0.10f)
        print_fail("az ≈ -1.0g — Z axis INVERTED (flip ALIGN_AZ sign)");
    else if (FABSF(avg_az - 0.5f) < 0.10f)
        print_fail("az ≈ +0.5g — WRONG SCALE (using ±16g divisor on ±8g chip?)");
    else if (FABSF(avg_az - 2.0f) < 0.20f)
        print_fail("az ≈ +2.0g — WRONG SCALE (using ±8g divisor on ±16g chip?)");
    else {
        print_fail("az is unexpected — check scale factor and alignment");
        uart_write_string("         Got az=");
        print_float3(avg_az);
        uart_write_string("\r\n");
    }

    /* Check ax, ay ≈ 0 */
    if (FABSF(avg_ax) < 0.08f)
        print_pass("ax ≈ 0 — X-axis not seeing gravity");
    else {
        print_fail("ax is not near zero — axis may be swapped or board tilted");
        uart_write_string("         Got ax=");
        print_float3(avg_ax);
        uart_write_string("\r\n");
    }

    if (FABSF(avg_ay) < 0.08f)
        print_pass("ay ≈ 0 — Y-axis not seeing gravity");
    else {
        print_fail("ay is not near zero — axis may be swapped or board tilted");
        uart_write_string("         Got ay=");
        print_float3(avg_ay);
        uart_write_string("\r\n");
    }

    /* Check gyros ≈ 0 */
    if (FABSF(avg_gx) < 0.03f && FABSF(avg_gy) < 0.03f && FABSF(avg_gz) < 0.03f)
        print_pass("Gyros ≈ 0 — no unexpected rotation bias");
    else
        print_fail("Gyro bias too large — board may be vibrating or gyro needs offset cal");

    /* Check noise spread on az */
    float az_spread = max_az - min_az;
    if (az_spread < 0.10f)
        print_pass("az noise spread < 0.10g — clean signal");
    else {
        uart_write_string("  [WARN] az noise spread = ");
        print_float3(az_spread);
        uart_write_string("g — may be vibration or EMI\r\n");
    }
}


/* ============================================================
 * Shared helpers for the interactive tests
 *
 * Previously every "Preparing in 3...2...1" countdown and every
 * "Return to FLAT" pause used delay_ms(), which stops calling
 * ahrs_update(). Any rotation made during those pauses was never seen
 * by the gyro, so the AHRS had to crawl to the new angle using the
 * accelerometer alone — at most 2*beta = 0.1 rad/s ≈ 5.7 °/s. That
 * produced the slow linear "creep" and the leftover angles that leaked
 * from one tilt into the next. These helpers keep the AHRS running at
 * 100 Hz through every pause.
 * ============================================================ */

/* AHRS helpers: get degrees */
extern float ahrs_get_roll(void);
extern float ahrs_get_pitch(void);
extern float ahrs_get_yaw(void);

/* Wait for the next 100 Hz SysTick period (SysTick is started in main). */
static void wait_tick(void)
{
    while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
}

/* Read one IMU sample in the body frame (fc_config.h alignment), with
 * optional bias correction (pass NULL to skip). */
static void read_body(imu_type_t imu, const float *gb, const float *ab,
                      float g[3], float a[3])
{
    imu_raw_t s;
    imu_read_raw(imu, &s);
    g[0] = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
    g[1] = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
    g[2] = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
    a[0] = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
    a[1] = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
    a[2] = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
    for (int k = 0; k < 3; k++) {
        if (gb) g[k] -= gb[k];
        if (ab) a[k] -= ab[k];
    }
}

/* Keep the AHRS updating for `ticks` × 10 ms. */
static void track_ticks(imu_type_t imu, const float *gb, const float *ab, int ticks)
{
    for (int i = 0; i < ticks; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, gb, ab, g, a);
        ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);
    }
}

/* "Preparing in 3...2...1..." with the AHRS tracking the whole time.
 * Move the board during the countdown; the caller prints the GO word. */
static void countdown_tracking(imu_type_t imu, const float *gb, const float *ab)
{
    uart_write_string("  Preparing in 3...");
    track_ticks(imu, gb, ab, 100);
    uart_write_string("2...");
    track_ticks(imu, gb, ab, 100);
    uart_write_string("1...");
    track_ticks(imu, gb, ab, 100);
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████ ██████
 *     ██    ██      ██         ██         ██
 *     ██    █████   ███████    ██     █████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██    ██████
 *
 *  DYNAMIC AXIS ALIGNMENT
 *
 *  Purpose: Verify that physical tilts produce the sign convention
 *           the NN was TRAINED with (fc-nn/drone_env.py, MuJoCo NWU:
 *           X forward, Y left, Z up). This is the #1 sim-to-real
 *           failure mode — it is how the pitch inversion was found.
 *
 *  Procedure: 3 tilts. Tilt during the countdown, hold while it samples.
 *
 *  Expected (training convention):
 *    TILT 1 nose DOWN   → ax < 0  (gravity shows up on -X)
 *                         AHRS pitch > 0
 *    TILT 2 right DOWN  → ay > 0, AHRS roll > 0
 *    TILT 3 rotate CW   → gz < 0 (right-hand rule about +Z up)
 *
 *  Auto-verdicts (averaged over the sampling window):
 *    nose down:  PASS if ax < -0.10 g and pitch > +5°
 *                FAIL "pitch inverted" if ax > +0.10 g and pitch < -5°
 *    right down: PASS if ay > +0.10 g and roll > +5°
 *    rotate CW:  PASS if gz < -0.05 rad/s
 *
 * ============================================================ */

typedef struct {
    float ax, ay, az, gx, gy, gz, roll, pitch, yaw;
} tilt_avg_t;

static void run_tilt_phase(imu_type_t imu, const char *label,
                           const char *instruction, int samples,
                           tilt_avg_t *out)
{
    uart_write_string("\r\n  >> ");
    uart_write_string(instruction);
    uart_write_string(" <<\r\n");
    uart_write_string("  (Move during the countdown, then hold for ~");
    uart_write_uint32(samples / 100);
    uart_write_string(" seconds)\r\n\r\n");

    /* Let user get into position — AHRS keeps tracking the motion */
    countdown_tracking(imu, NULL, NULL);
    uart_write_string("SAMPLING\r\n\r\n");

    float sum_ax = 0, sum_ay = 0, sum_az = 0;
    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    float sum_roll = 0, sum_pitch = 0, sum_yaw = 0;

    for (int i = 0; i < samples; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, NULL, NULL, g, a);
        ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);

        float roll  = ahrs_get_roll();
        float pitch = ahrs_get_pitch();
        float yaw   = ahrs_get_yaw();

        sum_ax += a[0];  sum_ay += a[1];  sum_az += a[2];
        sum_gx += g[0];  sum_gy += g[1];  sum_gz += g[2];
        sum_roll += roll;  sum_pitch += pitch;  sum_yaw += yaw;

        /* Print at 5 Hz (every 20 ticks) */
        if (i % 20 == 0) {
            uart_write_string("  ax=");  print_float3(a[0]);
            uart_write_string(" ay=");   print_float3(a[1]);
            uart_write_string(" az=");   print_float3(a[2]);
            uart_write_string(" gz=");   print_float3(g[2]);
            uart_write_string(" | R=");  print_float3(roll);
            uart_write_string(" P=");    print_float3(pitch);
            uart_write_string(" Y=");    print_float3(yaw);
            uart_write_string("\r\n");
        }
    }

    float n = (float)samples;
    out->ax = sum_ax / n;  out->ay = sum_ay / n;  out->az = sum_az / n;
    out->gx = sum_gx / n;  out->gy = sum_gy / n;  out->gz = sum_gz / n;
    out->roll = sum_roll / n;  out->pitch = sum_pitch / n;  out->yaw = sum_yaw / n;

    uart_write_string("\r\n  ---- ");
    uart_write_string(label);
    uart_write_string(" AVERAGES ----\r\n");
    uart_write_string("    ax="); print_float3(out->ax);
    uart_write_string("  ay="); print_float3(out->ay);
    uart_write_string("  az="); print_float3(out->az);
    uart_write_string("\r\n");
    uart_write_string("    gx="); print_float3(out->gx);
    uart_write_string("  gy="); print_float3(out->gy);
    uart_write_string("  gz="); print_float3(out->gz);
    uart_write_string("\r\n");
    uart_write_string("    AHRS roll="); print_float3(out->roll);
    uart_write_string("  pitch="); print_float3(out->pitch);
    uart_write_string("  yaw="); print_float3(out->yaw);
    uart_write_string("\r\n");
}


static void test3_dynamic_axis(imu_type_t imu)
{
    print_banner("TEST 3: Dynamic Axis Alignment");
    uart_write_string("Checks that tilts produce the SAME signs the NN was\r\n");
    uart_write_string("trained with (drone_env.py, X fwd / Y left / Z up).\r\n\r\n");
    uart_write_string("You will tilt the board 3 ways, ~30 deg each.\r\n");
    uart_write_string("Follow the prompts.\r\n");

    /* First, let AHRS settle for 2 seconds flat on table */
    uart_write_string("\r\nStep 0: Place board FLAT. AHRS settling (2 sec)...\r\n");
    ahrs_init();
    track_ticks(imu, NULL, NULL, 200);

    uart_write_string("  AHRS settled. Roll=");
    print_float3(ahrs_get_roll());
    uart_write_string(" Pitch=");
    print_float3(ahrs_get_pitch());
    uart_write_string("\r\n  (Should be near 0,0 if board is flat)\r\n");

    tilt_avg_t nd, rd, cw;

    /* ---- Tilt 1: Nose down ---- */
    uart_write_string("\r\n");
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 1: NOSE DOWN (~30 degrees)\r\n");
    uart_write_string("  Lift the REAR of the board so the nose dips.\r\n");
    uart_write_string("  Expected: ax < 0, AHRS pitch > 0\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "NOSE-DOWN", "Tilt NOSE DOWN and hold steady", 400, &nd);

    /* ---- Tilt 2: Right side down ---- */
    uart_write_string("\r\n");
    uart_write_string("  Return to FLAT (3 seconds, AHRS tracking)...\r\n");
    track_ticks(imu, NULL, NULL, 300);
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 2: RIGHT SIDE DOWN (~30 degrees)\r\n");
    uart_write_string("  Lift the LEFT side so the RIGHT side dips.\r\n");
    uart_write_string("  Expected: ay > 0, AHRS roll > 0\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "RIGHT-DOWN", "Tilt RIGHT SIDE DOWN and hold steady", 400, &rd);

    /* ---- Tilt 3: Yaw CW ---- */
    uart_write_string("\r\n");
    uart_write_string("  Return to FLAT (3 seconds, AHRS tracking)...\r\n");
    track_ticks(imu, NULL, NULL, 300);
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 3: ROTATE CW (viewed from above)\r\n");
    uart_write_string("  Keep board flat, rotate clockwise slowly while sampling.\r\n");
    uart_write_string("  Expected: gz < 0 (CW about +Z = negative), yaw decreasing\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "YAW-CW", "Rotate CLOCKWISE (keep flat)", 400, &cw);

    /* ---- Verdicts ---- */
    print_banner("TEST 3 RESULTS");

    if (nd.ax < -0.10f && nd.pitch > 5.0f)
        print_pass("NOSE DOWN: ax < 0 and pitch > 0 — matches training");
    else if (nd.ax > 0.10f && nd.pitch < -5.0f)
        print_fail("NOSE DOWN: ax > 0 and pitch < 0 — PITCH INVERTED vs training (check ALIGN_AX/ALIGN_GY in fc_config.h)");
    else
        uart_write_string("  [WARN] NOSE DOWN: inconclusive — tilt further (>15 deg) and hold still\r\n");

    if (rd.ay > 0.10f && rd.roll > 5.0f)
        print_pass("RIGHT DOWN: ay > 0 and roll > 0 — matches training");
    else if (rd.ay < -0.10f && rd.roll < -5.0f)
        print_fail("RIGHT DOWN: ay < 0 and roll < 0 — ROLL INVERTED vs training (check ALIGN_AY/ALIGN_GX)");
    else
        uart_write_string("  [WARN] RIGHT DOWN: inconclusive — tilt further (>15 deg) and hold still\r\n");

    if (cw.gz < -0.05f)
        print_pass("ROTATE CW: gz < 0 — matches training");
    else if (cw.gz > 0.05f)
        print_fail("ROTATE CW: gz > 0 — YAW INVERTED vs training (check ALIGN_GZ)");
    else
        uart_write_string("  [WARN] ROTATE CW: inconclusive — rotate steadily during sampling\r\n");

    uart_write_string("\r\nReference (training convention):\r\n");
    uart_write_string("  Nose down:     ax<0, pitch>0\r\n");
    uart_write_string("  Right down:    ay>0, roll>0\r\n");
    uart_write_string("  CW rotation:   gz<0, yaw decreasing\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████ ██   ██
 *     ██    ██      ██         ██    ██   ██
 *     ██    █████   ███████    ██    ███████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██         ██
 *
 *  AHRS CONVERGENCE TEST
 *
 *  Purpose: Verify the Madgwick filter converges to the correct
 *           angle within a reasonable time, and returns to zero.
 *
 *  Procedure:
 *    Phase A: Board flat, AHRS settles → should read ~0°, ~0°.
 *    Phase B: Tilt to ~30° roll and hold → AHRS roll should
 *             converge to ~25-35° within 1-2 seconds.
 *    Phase C: Return flat and hold → AHRS roll should return
 *             to ~0° within 1-2 seconds.
 *
 *  What we're checking:
 *    - Correct sign (positive roll = right-wing-down)
 *    - Correct magnitude (reads ~30° when tilted 30°, not 15°)
 *    - Convergence speed (settles in <2 seconds with beta=0.05)
 *    - Return to zero (no accumulated drift)
 *
 *  Auto-validation:
 *    Phase A: |roll| < 3° and |pitch| < 3°
 *    Phase B: roll > 15° (confirms convergence in correct direction)
 *    Phase C: |roll| < 5° (confirms return to level)
 *
 * ============================================================ */
static void test4_ahrs_convergence(imu_type_t imu)
{
    print_banner("TEST 4: AHRS Convergence");
    uart_write_string("Tests that the Madgwick filter converges to correct\r\n");
    uart_write_string("angles and returns to zero after tilting.\r\n\r\n");

    /* ---- Phase A: Settle flat (3 seconds) ---- */
    uart_write_string("Phase A: Place board FLAT and STILL.\r\n");
    uart_write_string("  Resetting AHRS and settling for 3 seconds...\r\n");
    ahrs_init();

    float roll_a = 0.0f, pitch_a = 0.0f;
    for (int i = 0; i < 300; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        /* Print at 2 Hz for last second only */
        if (i >= 200 && i % 50 == 0) {
            uart_write_string("    R=");
            print_float3(ahrs_get_roll());
            uart_write_string(" P=");
            print_float3(ahrs_get_pitch());
            uart_write_string("\r\n");
        }
    }
    roll_a  = ahrs_get_roll();
    pitch_a = ahrs_get_pitch();

    uart_write_string("  Final flat: R=");
    print_float3(roll_a);
    uart_write_string(" P=");
    print_float3(pitch_a);
    uart_write_string("\r\n");

    if (FABSF(roll_a) < 3.0f && FABSF(pitch_a) < 3.0f)
        print_pass("Phase A: AHRS settled near 0,0 when flat");
    else
        print_fail("Phase A: AHRS not near 0,0 when flat — check accel alignment");

    /* ---- Phase B: Tilt to ~30° roll right and hold (5 seconds) ---- */
    uart_write_string("\r\nPhase B: Tilt RIGHT SIDE DOWN ~30 degrees.\r\n");
    countdown_tracking(imu, NULL, NULL);   /* AHRS keeps tracking */
    uart_write_string("HOLD TILT NOW\r\n\r\n");

    float roll_b = 0.0f;
    for (int i = 0; i < 500; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        /* Print at 5 Hz */
        if (i % 20 == 0) {
            uart_write_string("  [");
            uart_write_uint32(i);
            uart_write_string("] R=");
            print_float3(ahrs_get_roll());
            uart_write_string(" P=");
            print_float3(ahrs_get_pitch());
            uart_write_string("\r\n");
        }
    }
    roll_b = ahrs_get_roll();

    uart_write_string("  Final tilted: R=");
    print_float3(roll_b);
    uart_write_string("\r\n");

    /* Check that roll converged to a reasonable tilt */
    if (roll_b > 15.0f)
        print_pass("Phase B: Roll > +15 deg — converged in correct direction");
    else if (roll_b < -15.0f)
        print_fail("Phase B: Roll < -15 deg — SIGN IS WRONG (roll axis inverted)");
    else {
        uart_write_string("  [WARN] Roll magnitude only ");
        print_float3(roll_b);
        uart_write_string(" deg — did you tilt far enough? Or filter is sluggish.\r\n");
    }

    /* ---- Phase C: Return flat and hold (4 seconds) ---- */
    uart_write_string("\r\nPhase C: Return board to FLAT.\r\n");
    countdown_tracking(imu, NULL, NULL);   /* AHRS keeps tracking */
    uart_write_string("HOLD FLAT NOW\r\n\r\n");

    float roll_c = 0.0f;
    for (int i = 0; i < 400; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        if (i % 20 == 0) {
            uart_write_string("  [");
            uart_write_uint32(i);
            uart_write_string("] R=");
            print_float3(ahrs_get_roll());
            uart_write_string(" P=");
            print_float3(ahrs_get_pitch());
            uart_write_string("\r\n");
        }
    }
    roll_c = ahrs_get_roll();

    uart_write_string("  Final return-flat: R=");
    print_float3(roll_c);
    uart_write_string("\r\n");

    if (FABSF(roll_c) < 5.0f)
        print_pass("Phase C: Roll returned near 0 — no accumulated drift");
    else
        print_fail("Phase C: Roll did not return to 0 — drift or slow convergence");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████ ███████
 *     ██    ██      ██         ██    ██
 *     ██    █████   ███████    ██    ███████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██    ███████
 *
 *  OBSERVATION VECTOR GROUND TRUTH
 *
 *  Purpose: Build the exact obs[12] vector that gets fed to the NN,
 *           using the same code path as main.c, and verify each
 *           element matches what the MuJoCo sim would produce for
 *           a flat, stationary drone.
 *
 *  Procedure: Board flat and still on table for 5 seconds.
 *
 *  Expected obs vector at rest (NWU, flat, at baro ground ref):
 *    obs[0]  =  0.000   gyro X  (no rotation)
 *    obs[1]  =  0.000   gyro Y
 *    obs[2]  =  0.000   gyro Z
 *    obs[3]  =  0.000   accel X / 2g  (no lateral accel)
 *    obs[4]  =  0.000   accel Y / 2g
 *    obs[5]  = +0.500   accel Z / 2g  ← THE BIG ONE (1g / 2 = 0.5)
 *    obs[6]  =  0.000   roll error / 45°
 *    obs[7]  =  0.000   pitch error / 45°
 *    obs[8]  = -0.050   altitude error: on the table alt≈0, target is
 *                       0.3 m → (0-0.3)/2 = -0.15, clamped to -0.05
 *                       (0.000 if no barometer)
 *    obs[9]  =  0.000   vertical velocity
 *    obs[10] =  0.000   hvel X (hardcoded 0)
 *    obs[11] =  0.000   hvel Y (hardcoded 0)
 *
 *  Key failure modes this catches:
 *    obs[5] = -0.5 → Z-axis inverted, NN thinks drone is upside down
 *    obs[5] = +0.25 or +1.0 → wrong accel scale factor
 *    obs[3] or obs[4] ≈ ±0.5 → accel axes swapped
 *    obs[6] or obs[7] != 0 → AHRS not settled or sign error
 *
 * ============================================================ */

/* Clamp utility (same as main.c) */
static inline float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Obs vector scaling — aliases for the shared fc_config.h values so the
 * harness can never drift from main.c again. */
#define OBS_MAX_GYRO_RADS   MAX_GYRO_RADS
#define OBS_ANGLE_SCALE     ANGLE_OBS_SCALE
#define OBS_ALT_SCALE       ALT_SCALE
#define OBS_VVEL_SCALE      VVEL_SCALE
#define OBS_ALT_OBS_CLAMP   ALT_OBS_CLAMP
#define OBS_TARGET_ALT      TARGET_ALT

static void test5_obs_vector(imu_type_t imu, int baro_ok)
{
    print_banner("TEST 5: Observation Vector Ground Truth");
    uart_write_string("Builds the exact obs[12] fed to the NN and verifies\r\n");
    uart_write_string("each element matches the sim's 'flat on table' values.\r\n\r\n");
    uart_write_string("Board must be FLAT and STILL.\r\n");

    /* Re-init AHRS and settle for 3 seconds */
    uart_write_string("  Resetting AHRS, settling 3 sec...\r\n");
    ahrs_init();
    for (int i = 0; i < 300; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        if (baro_ok) baro_update(0.01f);
    }

    /* If baro available, set ground reference now (we're on the table) */
    if (baro_ok) {
        uart_write_string("  Setting baro ground reference...\r\n");
        baro_set_ground_reference();
        /* Let it settle a bit more */
        for (int i = 0; i < 100; i++) {
            while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
            imu_raw_t s;
            imu_read_raw(imu, &s);
            float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
            float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
            float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
            float axf = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
            float ayf = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
            float azf = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
            ahrs_update(gx, gy, gz, axf, ayf, azf, 0.01f);
            baro_update(0.01f);
        }
    }

    /* Now sample 200 ticks (2 seconds) and average the obs vector */
    uart_write_string("  Sampling obs vector for 2 seconds...\r\n\r\n");

    float obs_sum[12] = {0};
    #define OBS_SAMPLES 200

    for (int i = 0; i < OBS_SAMPLES; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        imu_raw_t s;
        imu_read_raw(imu, &s);

        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;

        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        if (baro_ok) baro_update(0.01f);

        /* Build obs exactly like main.c */
        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);

        float roll_deg  = ahrs_get_roll();
        float pitch_deg = ahrs_get_pitch();
        obs[6] = clampf(roll_deg  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch_deg * OBS_ANGLE_SCALE, -1.0f, 1.0f);

        if (baro_ok) {
            float alt     = baro_get_altitude();
            float alt_err = alt - OBS_TARGET_ALT;
            float vvel    = baro_get_vvel();
            obs[8]  = clampf(alt_err / OBS_ALT_SCALE,
                             -OBS_ALT_OBS_CLAMP, OBS_ALT_OBS_CLAMP);
            obs[9]  = clampf(vvel / OBS_VVEL_SCALE, -1.0f, 1.0f);
        } else {
            obs[8]  = 0.0f;
            obs[9]  = 0.0f;
        }
        obs[10] = 0.0f;
        obs[11] = 0.0f;

        for (int k = 0; k < 12; k++) obs_sum[k] += obs[k];
    }

    /* Average */
    float obs_avg[12];
    for (int k = 0; k < 12; k++) obs_avg[k] = obs_sum[k] / (float)OBS_SAMPLES;

    /* Expected values.
     * obs[8]: on the table the baro reads alt≈0 m but the target is
     * TARGET_ALT (0.3 m), so (0 - 0.3)/ALT_SCALE = -0.15, which main.c
     * clamps to -ALT_OBS_CLAMP (-0.05). That clamped value IS the correct
     * firmware output here — it used to be reported as a FAIL. */
    float expected[12] = {
        0.0f, 0.0f, 0.0f,       /* gyro XYZ */
        0.0f, 0.0f, 0.5f,       /* accel XYZ / 2g */
        0.0f, 0.0f,             /* roll, pitch error */
        0.0f, 0.0f,             /* alt error (set below), vvel */
        0.0f, 0.0f              /* hvel XY */
    };
    if (baro_ok)
        expected[8] = clampf((0.0f - OBS_TARGET_ALT) / OBS_ALT_SCALE,
                             -OBS_ALT_OBS_CLAMP, OBS_ALT_OBS_CLAMP);
    /* Tolerances per element */
    static const float tol[12] = {
        0.01f, 0.01f, 0.01f,    /* gyro: tight, should be near zero */
        0.05f, 0.05f, 0.08f,    /* accel: moderate (noise) */
        0.04f, 0.04f,           /* AHRS angles: should be near 0 if flat */
        0.01f, 0.02f,           /* baro: clamped value is exact; vvel near 0 */
        0.001f, 0.001f          /* hvel: hardcoded 0 */
    };
    static const char *names[12] = {
        "obs[0]  gyro_x  ",
        "obs[1]  gyro_y  ",
        "obs[2]  gyro_z  ",
        "obs[3]  accel_x ",
        "obs[4]  accel_y ",
        "obs[5]  accel_z ",
        "obs[6]  roll_err",
        "obs[7]  pitch_er",
        "obs[8]  alt_err ",
        "obs[9]  vvel    ",
        "obs[10] hvel_x  ",
        "obs[11] hvel_y  "
    };

    /* Print and check each element */
    int obs_pass = 0;
    for (int k = 0; k < 12; k++) {
        uart_write_string("  ");
        uart_write_string(names[k]);
        uart_write_string(" = ");
        print_float3(obs_avg[k]);
        uart_write_string("   expect ");
        print_float3(expected[k]);
        uart_write_string("  tol ±");
        print_float3(tol[k]);

        float err = obs_avg[k] - expected[k];
        if (err < 0) err = -err;
        if (err <= tol[k]) {
            uart_write_string("  OK\r\n");
            obs_pass++;
        } else {
            uart_write_string("  ** FAIL **\r\n");
        }
    }

    uart_write_string("\r\n  Result: ");
    uart_write_uint32(obs_pass);
    uart_write_string("/12 elements within tolerance.\r\n");

    if (obs_pass == 12)
        print_pass("Obs vector matches sim ground truth");
    else
        print_fail("Obs vector mismatch — NN is seeing a world it wasn't trained on");

    /* Specific diagnostic for obs[5] */
    if (FABSF(obs_avg[5] - 0.5f) > 0.08f) {
        uart_write_string("\r\n  ** obs[5] DIAGNOSTIC **\r\n");
        if (FABSF(obs_avg[5] + 0.5f) < 0.08f)
            uart_write_string("  obs[5] ≈ -0.5: Z-accel INVERTED. NN thinks you're upside down.\r\n");
        else if (FABSF(obs_avg[5] - 0.25f) < 0.08f)
            uart_write_string("  obs[5] ≈ 0.25: Accel scale is 2x too small (wrong full-scale?).\r\n");
        else if (FABSF(obs_avg[5] - 1.0f) < 0.15f)
            uart_write_string("  obs[5] ≈ 1.0: Accel scale is 2x too large (wrong full-scale?).\r\n");
        else {
            uart_write_string("  obs[5] = ");
            print_float3(obs_avg[5]);
            uart_write_string(" — unexpected. Check ALIGN_AZ and accel_raw_to_g.\r\n");
        }
    }

    /* Also print what baro contributed */
    if (!baro_ok) {
        uart_write_string("\r\n  Note: No barometer detected. obs[8,9] forced to 0.\r\n");
        uart_write_string("  This matches what main.c does when baro_ok=0.\r\n");
    } else {
        /* obs[8] with target alt = 0.3m will be slightly nonzero
           since we just set ground ref and target is 0.3m above it */
        uart_write_string("\r\n  Note: Baro present. obs[8] = (alt - 0.3m) / 2.0,\r\n");
        uart_write_string("  clamped to ±0.05. On table: alt≈0, so obs[8]≈-0.05 (clamped).\r\n");
        uart_write_string("  This is expected — the P-altitude controller handles the rest.\r\n");
    }
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██████
 *     ██    ██      ██         ██    ██
 *     ██    █████   ███████    ██    ██████
 *     ██    ██           ██    ██    ██    ██
 *     ██    ███████ ███████    ██     ██████
 *
 *  NN INFERENCE VALIDATION
 *
 *  Purpose: Run nn_forward() ON THE STM32 for every reference vector in
 *           host_test/golden_vectors.h and compare against the outputs
 *           Python computed from the SAME weights.h. Catches: wrong or
 *           stale weights flashed, float/compiler issues on the target.
 *
 *  The reference values are generated, never typed by hand:
 *    fc-nn/generate_golden.py  (run with:  make golden)
 *  The golden file records the SHA256 of the weights.h it came from, and
 *  the Makefile refuses to build the harness if that hash does not match
 *  the current weights.h — so this test can no longer go stale silently
 *  (it previously compared against numbers from an older network and
 *  reported 0/8 even though the maths was correct).
 *
 *  Tolerance: |err| <= 1e-3 * max(1, |expected|). Float32 on host and
 *  target typically agree to ~1e-6.
 *
 * ============================================================ */
static void test6_nn_inference(void)
{
    print_banner("TEST 6: NN Inference Validation");
    uart_write_string("Runs nn_forward() on the chip for every golden vector and\r\n");
    uart_write_string("compares with Python's output from the same weights.h.\r\n\r\n");

    static const char *motor_names[4] = { "FL", "FR", "RR", "RL" };
    int pass = 0, total = 0;
    float worst = 0.0f;

    for (int v = 0; v < N_GOLDEN; v++) {
        float obs[12], out[4];
        for (int k = 0; k < 12; k++) obs[k] = golden_obs[v][k];
        nn_forward(obs, out);

        int vec_ok = 1;
        for (int k = 0; k < 4; k++) {
            float ref = golden_out[v][k];
            float err = FABSF(out[k] - ref);
            float tol = 1e-3f * (FABSF(ref) > 1.0f ? FABSF(ref) : 1.0f);
            if (err > worst) worst = err;
            total++;
            if (err <= tol) pass++; else vec_ok = 0;
        }

        uart_write_string(vec_ok ? "  OK   " : "  FAIL ");
        uart_write_string(golden_names[v]);
        uart_write_string(": ");
        for (int k = 0; k < 4; k++) {
            uart_write_string(motor_names[k]);
            uart_write_char('=');
            print_float3(out[k]);
            uart_write_char(' ');
        }
        if (!vec_ok) {
            uart_write_string(" expect ");
            for (int k = 0; k < 4; k++) { print_float3(golden_out[v][k]); uart_write_char(' '); }
        }
        uart_write_string("\r\n");
    }

    /* Context: what the level "hover" vector would command (v = 0) */
    {
        float out[4], obs[12];
        for (int k = 0; k < 12; k++) obs[k] = golden_obs[0][k];
        nn_forward(obs, out);
        uart_write_string("\r\n  Level-hover vector → motor PWM (hover_thr=");
        print_float3(NN_HOVER_THROTTLE);
        uart_write_string(", scale=");
        print_float3(NN_ACTION_SCALE);
        uart_write_string("):\r\n   ");
        for (int k = 0; k < 4; k++) {
            float cmd = clampf(NN_HOVER_THROTTLE + clampf(out[k], -1.0f, 1.0f) * NN_ACTION_SCALE, 0.0f, 1.0f);
            uart_write_char(' ');
            uart_write_string(motor_names[k]);
            uart_write_char('=');
            uart_write_uint32(1000U + (uint32_t)(cmd * 1000.0f + 0.5f));
        }
        uart_write_string(" us\r\n");
    }

    uart_write_string("\r\n  Result: ");
    uart_write_uint32((uint32_t)pass);
    uart_write_string("/");
    uart_write_uint32((uint32_t)total);
    uart_write_string(" outputs match.  Worst |err| = ");
    print_float3(worst * 1000.0f);
    uart_write_string("e-3\r\n");

    if (pass == total)
        print_pass("On-chip NN output matches Python for every golden vector");
    else {
        print_fail("On-chip NN output differs from Python");
        uart_write_string("  Likely causes:\r\n");
        uart_write_string("    1. The board runs an old build (reflash)\r\n");
        uart_write_string("    2. weights.h changed but golden vectors were not regenerated (make golden)\r\n");
    }
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████ ███████
 *     ██    ██      ██         ██         ██
 *     ██    █████   ███████    ██     █████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██    ███████
 *
 *  HARDWARE-IN-THE-LOOP NN CLOSED LOOP
 *
 *  Purpose: Run the EXACT main.c control loop (sensors → obs →
 *           NN → IIR smoothing → motor mapping) with REAL sensor
 *           data, but WITHOUT actually driving the motors.
 *           Print the motor commands the NN *would* send.
 *
 *  This answers the critical question: "If I armed the drone
 *  right now on this bench, what would the motors do?"
 *
 *  Procedure: Board flat and still on table for 10 seconds.
 *             The test runs the full 100 Hz control loop.
 *
 *  What to look for:
 *
 *    1. CONVERGENCE: Motor commands should settle to stable values
 *       within 2-3 seconds (SETTLE_TICKS=200 in main.c, then the
 *       IIR filter needs another ~20 ticks to converge).
 *
 *    2. SYMMETRY: At steady state, all 4 motors should be within
 *       ~50 µs of each other. Large asymmetry (>100 µs) means the
 *       NN is trying to correct for something it shouldn't be —
 *       a sign error in the obs, a persistent AHRS bias, etc.
 *
 *    3. AVERAGE PWM (informational only): base 1323 µs (the hover
 *       GUESS from the linear 8.5 N thrust model) + the altitude P+D
 *       term (+45 µs on a table: 0.3 m below target) + the NN's mean
 *       output. The real hover PWM is unknown until measured on a thrust
 *       stand (test 14), so the average is no longer graded.
 *
 *    4. STABILITY: Commands should not oscillate. If you see
 *       motors alternating between high and low every few ticks,
 *       the NN is fighting itself — likely a sign error.
 *
 *  Auto-validation:
 *    - PWM spread < 80 µs between motors  (PASS)
 *    - Steady-state stddev < 20 µs        (PASS = stable)
 *  Balance report (what the imbalance would do to the airframe):
 *    roll  = left pair − right pair   (>0 → rolls RIGHT, right side down)
 *    pitch = front pair − rear pair   (>0 → nose UP, <0 → nose DOWN)
 *    yaw   = CW pair − CCW pair       (>0 → yaws CCW seen from above)
 *
 * ============================================================ */

/* Action filter: action_filter_step() / ACTION_FILTER_ALPHA from
 * fc_config.h — identical to main.c and to training. */

/* Altitude P+D constants — MUST match main.c */
/* Altitude P+D constants — aliases for fc_config.h. (The harness used to
 * keep its own copy with ALT_KD = 0.08 while main.c used 0.02, so tests 7
 * and 13 were not simulating the real firmware.) */
#define HIL_ALT_KP          ALT_KP
#define HIL_ALT_KD          ALT_KD
#define HIL_ALT_ADJ_MAX     ALT_ADJ_MAX
#define HIL_ALT_OBS_CLAMP   ALT_OBS_CLAMP
#define HIL_TARGET_ALT      TARGET_ALT
#define HIL_ALT_SCALE       ALT_SCALE
#define HIL_VVEL_SCALE      VVEL_SCALE

static void test7_hil_closed_loop(imu_type_t imu, int baro_ok)
{
    print_banner("TEST 7: Hardware-in-the-Loop Closed Loop");
    uart_write_string("Runs the FULL control loop with real sensors,\r\n");
    uart_write_string("but motors OFF. Shows what PWM would be sent.\r\n\r\n");
    uart_write_string("Board must be FLAT and STILL for 10 seconds.\r\n\r\n");

    /* Reset AHRS fresh */
    ahrs_init();

    /* If baro present, set ground reference */
    if (baro_ok) {
        uart_write_string("  Setting baro ground reference...\r\n");
        /* Settle baro */
        for (int i = 0; i < 50; i++) {
            baro_update(0.04f);
            delay_ms(40);
        }
        baro_set_ground_reference();
    }

    /* ── Settle phase: run AHRS for 2 seconds + calibrate IMU bias ── */
    uart_write_string("  AHRS settling + IMU bias calibration (2 sec)...\r\n");

    /* Re-init SysTick — previous tests may have left it in a stale state */
    systick_init_100hz();

    float cal_sum_gx = 0, cal_sum_gy = 0, cal_sum_gz = 0;
    float cal_sum_ax = 0, cal_sum_ay = 0, cal_sum_az = 0;
    int   cal_count = 0;

    for (int i = 0; i < 200; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        if (baro_ok) baro_update(0.01f);

        /* Accumulate for bias cal (skip first 50 ticks for IMU warmup) */
        if (i >= 50) {
            cal_sum_gx += gx;  cal_sum_gy += gy;  cal_sum_gz += gz;
            cal_sum_ax += ax;  cal_sum_ay += ay;  cal_sum_az += az;
            cal_count++;
        }
    }

    /* Compute biases */
    float gyro_bias[3]  = {0, 0, 0};
    float accel_bias[3] = {0, 0, 0};
    if (cal_count > 0) {
        float inv_n = 1.0f / (float)cal_count;
        gyro_bias[0]  = cal_sum_gx * inv_n;
        gyro_bias[1]  = cal_sum_gy * inv_n;
        gyro_bias[2]  = cal_sum_gz * inv_n;
        accel_bias[0] = cal_sum_ax * inv_n;          /* expect 0g */
        accel_bias[1] = cal_sum_ay * inv_n;          /* expect 0g */
        accel_bias[2] = cal_sum_az * inv_n - 1.0f;   /* expect +1g */
    }

    uart_write_string("  Bias calibration results:\r\n");
    uart_write_string("    gyro_bias:  gx=");
    print_float3(gyro_bias[0]);
    uart_write_string("  gy=");
    print_float3(gyro_bias[1]);
    uart_write_string("  gz=");
    print_float3(gyro_bias[2]);
    uart_write_string(" rad/s\r\n");
    uart_write_string("    accel_bias: ax=");
    print_float3(accel_bias[0]);
    uart_write_string("  ay=");
    print_float3(accel_bias[1]);
    uart_write_string("  az=");
    print_float3(accel_bias[2]);
    uart_write_string(" g\r\n");

    /* Show what angles the bias was causing */
    /* roll_from_bias ≈ atan2(ay_bias, 1g) ≈ ay_bias in radians ≈ ay_bias * 57.3 deg */
    float bias_roll  = accel_bias[1] * 57.296f;
    float bias_pitch = -accel_bias[0] * 57.296f;
    uart_write_string("    Predicted AHRS error without cal: R=");
    print_float3(bias_roll);
    uart_write_string(" deg  P=");
    print_float3(bias_pitch);
    uart_write_string(" deg\r\n");

    /* Re-init AHRS so it converges fresh with corrected data */
    ahrs_init();
    uart_write_string("  AHRS re-initialized. Re-settling with corrected data (1 sec)...\r\n");

    for (int i = 0; i < 100; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gyro_bias[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gyro_bias[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gyro_bias[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - accel_bias[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - accel_bias[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - accel_bias[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        if (baro_ok) baro_update(0.01f);
    }

    uart_write_string("  Post-cal AHRS: R=");
    print_float3(ahrs_get_roll());
    uart_write_string(" P=");
    print_float3(ahrs_get_pitch());
    uart_write_string(" (should be ~0,0)\r\n");

    uart_write_string("  Armed. Running NN control loop for 8 seconds...\r\n\r\n");

    /* ── Control loop: 800 ticks = 8 seconds at 100 Hz ── */
    #define HIL_TICKS       800
    #define HIL_RAMP_TICKS  50     /* matches main.c RAMP_TICKS */

    float nn_smoothed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float motor_cmd[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t ramp_counter = 0;

    /* Stats for the last 200 ticks (steady state, seconds 6-8) */
    #define HIL_STATS_START  600
    float ss_sum[4]  = {0};   /* sum of PWM in steady state */
    float ss_sum2[4] = {0};   /* sum of PWM^2 for variance */
    int   ss_count   = 0;

    /* Track min/max PWM in steady state */
    float ss_min[4] = {9999, 9999, 9999, 9999};
    float ss_max[4] = {0, 0, 0, 0};

    for (int tick = 0; tick < HIL_TICKS; tick++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        /* ── A: Read IMU + apply bias correction ── */
        imu_raw_t s;
        imu_read_raw(imu, &s);

        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gyro_bias[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gyro_bias[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gyro_bias[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - accel_bias[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - accel_bias[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - accel_bias[2];

        /* ── B: AHRS ── */
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        float roll_deg  = ahrs_get_roll();
        float pitch_deg = ahrs_get_pitch();

        /* ── C: Baro ── */
        if (baro_ok) baro_update(0.01f);

        /* ── D: Build obs (same as main.c) ── */
        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
        obs[6] = clampf(roll_deg  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch_deg * OBS_ANGLE_SCALE, -1.0f, 1.0f);

        float alt_throttle_adj = 0.0f;
        if (baro_ok) {
            float alt     = baro_get_altitude();
            float alt_err = alt - HIL_TARGET_ALT;
            float vvel    = baro_get_vvel();
            obs[8]  = clampf(alt_err / HIL_ALT_SCALE,
                             -HIL_ALT_OBS_CLAMP, HIL_ALT_OBS_CLAMP);
            obs[9]  = clampf(vvel / HIL_VVEL_SCALE, -1.0f, 1.0f);
            alt_throttle_adj = clampf(
                HIL_ALT_KP * (HIL_TARGET_ALT - alt) + HIL_ALT_KD * (-vvel),
                -HIL_ALT_ADJ_MAX, HIL_ALT_ADJ_MAX
            );
        } else {
            obs[8]  = 0.0f;
            obs[9]  = 0.0f;
        }
        obs[10] = 0.0f;
        obs[11] = 0.0f;

        /* ── E: NN forward pass ── */
        float nn_out[4];
        nn_forward(obs, nn_out);
        for (int k = 0; k < 4; k++)
            nn_out[k] = clampf(nn_out[k], -1.0f, 1.0f);

        /* ── F: Action filter (fc_config.h, same as main.c) ── */
        action_filter_step(nn_smoothed, nn_out);
        for (int k = 0; k < 4; k++)
            nn_out[k] = nn_smoothed[k];

        /* ── G: Motor command mapping ── */
        float base_throttle = NN_HOVER_THROTTLE + alt_throttle_adj;
        for (int k = 0; k < 4; k++) {
            motor_cmd[k] = clampf(
                base_throttle + nn_out[k] * NN_ACTION_SCALE,
                0.0f, 1.0f
            );
        }

        /* ── H: Throttle ramp (same as main.c) ── */
        if (ramp_counter < HIL_RAMP_TICKS) {
            ramp_counter++;
            float ramp_frac = (float)ramp_counter / (float)HIL_RAMP_TICKS;
            for (int k = 0; k < 4; k++) {
                float nn_residual = nn_out[k] * NN_ACTION_SCALE;
                float ramped_base = ramp_frac * NN_HOVER_THROTTLE;
                float floor       = ramp_frac * 0.06f;
                float cmd = ramped_base + ramp_frac * nn_residual;
                motor_cmd[k] = clampf(cmd, floor, 1.0f);
            }
        }

        /* ── Convert to PWM µs for display ── */
        float pwm[4];
        for (int k = 0; k < 4; k++)
            pwm[k] = 1000.0f + motor_cmd[k] * 1000.0f;

        /* ── Collect steady-state stats (last 200 ticks) ── */
        if (tick >= HIL_STATS_START) {
            for (int k = 0; k < 4; k++) {
                ss_sum[k]  += pwm[k];
                ss_sum2[k] += pwm[k] * pwm[k];
                if (pwm[k] < ss_min[k]) ss_min[k] = pwm[k];
                if (pwm[k] > ss_max[k]) ss_max[k] = pwm[k];
            }
            ss_count++;
        }

        /* ── Print at 5 Hz (every 20 ticks) ── */
        if (tick % 20 == 0) {
            /* Time in seconds */
            uart_write_string("  [");
            print_float3((float)tick * 0.01f);
            uart_write_string("s] ");

            /* Motor PWMs */
            static const char *mnames[4] = {"FL:", "FR:", "RR:", "RL:"};
            for (int k = 0; k < 4; k++) {
                uart_write_string(mnames[k]);
                uart_write_uint32((uint32_t)(pwm[k] + 0.5f));
                if (k < 3) uart_write_char(' ');
            }

            /* Roll/pitch for context */
            uart_write_string(" | R=");
            print_float3(roll_deg);
            uart_write_string(" P=");
            print_float3(pitch_deg);

            /* Mark ramp phase */
            if (tick < HIL_RAMP_TICKS) {
                uart_write_string(" [RAMP]");
            }

            uart_write_string("\r\n");
        }
    }

    /* ============================================================
     * STEADY-STATE ANALYSIS (last 200 ticks = 2 seconds)
     * ============================================================ */
    uart_write_string("\r\n---- STEADY-STATE ANALYSIS (last 2 sec) ----\r\n\r\n");

    float ss_avg[4], ss_var[4];
    float overall_min = 9999.0f, overall_max = 0.0f;
    float overall_sum = 0.0f;

    static const char *motor_labels[4] = {
        "FL(CCW)", "FR(CW) ", "RR(CCW)", "RL(CW) "
    };

    for (int k = 0; k < 4; k++) {
        ss_avg[k] = ss_sum[k] / (float)ss_count;
        ss_var[k] = ss_sum2[k] / (float)ss_count - ss_avg[k] * ss_avg[k];
        /* sqrt via hardware FPU */
        float stddev;
        __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(stddev) : "t"(ss_var[k]));

        uart_write_string("  ");
        uart_write_string(motor_labels[k]);
        uart_write_string("  avg=");
        uart_write_uint32((uint32_t)(ss_avg[k] + 0.5f));
        uart_write_string(" us  stddev=");
        print_float3(stddev);
        uart_write_string(" us  range=[");
        uart_write_uint32((uint32_t)(ss_min[k] + 0.5f));
        uart_write_string("-");
        uart_write_uint32((uint32_t)(ss_max[k] + 0.5f));
        uart_write_string("]\r\n");

        overall_sum += ss_avg[k];
        if (ss_avg[k] < overall_min) overall_min = ss_avg[k];
        if (ss_avg[k] > overall_max) overall_max = ss_avg[k];
    }

    float overall_avg = overall_sum / 4.0f;
    float spread = overall_max - overall_min;

    uart_write_string("\r\n  Overall avg = ");
    uart_write_uint32((uint32_t)(overall_avg + 0.5f));
    uart_write_string(" us  (expect ~");
    uart_write_uint32((uint32_t)(1000.0f + NN_HOVER_THROTTLE * 1000.0f + 0.5f));
    uart_write_string(" us)\r\n");
    uart_write_string("  Motor spread = ");
    uart_write_uint32((uint32_t)(spread + 0.5f));
    uart_write_string(" us  (max - min across 4 motors)\r\n\r\n");

    /* ---- Auto-checks ---- */

    /* Check 1: Spread < 80 µs */
    if (spread < 80.0f)
        print_pass("Motor spread < 80 us — well balanced at hover");
    else if (spread < 120.0f) {
        uart_write_string("  [WARN] Motor spread = ");
        uart_write_uint32((uint32_t)(spread + 0.5f));
        uart_write_string(" us — moderate asymmetry. May fly but with drift.\r\n");
    } else
        print_fail("Motor spread > 120 us — large asymmetry, NN is compensating for something");

    /* Check 2 (informational): average PWM. Not graded — the real hover
     * PWM has not been measured yet (thrust stand / test 14). */
    uart_write_string("  [INFO] Average PWM is not graded: real hover PWM is unmeasured.\r\n");
    uart_write_string("         On a table it includes ~+45 us of altitude P term\r\n");
    uart_write_string("         (board is 0.3 m below TARGET_ALT).\r\n");

    /* Check 3: Stability — stddev of each motor */
    int stable = 1;
    for (int k = 0; k < 4; k++) {
        float stddev;
        __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(stddev) : "t"(ss_var[k]));
        if (stddev > 20.0f) stable = 0;
    }
    if (stable)
        print_pass("All motors stable (stddev < 20 us) — no oscillation");
    else
        print_fail("Motor commands oscillating — NN is fighting itself");

    /* Check 4: Balance per axis — what the imbalance would do in the air.
     * (The old code printed the CW/CCW pair twice — once as "yaw" and once,
     * sign-flipped, as "diagonal roll/pitch" — and never reported roll or
     * pitch at all.)  Indices: 0=FL(CCW) 1=FR(CW) 2=RR(CCW) 3=RL(CW). */
    float roll_bal  = (ss_avg[0] + ss_avg[3]) / 2.0f - (ss_avg[1] + ss_avg[2]) / 2.0f;
    float pitch_bal = (ss_avg[0] + ss_avg[1]) / 2.0f - (ss_avg[2] + ss_avg[3]) / 2.0f;
    float yaw_bal   = (ss_avg[1] + ss_avg[3]) / 2.0f - (ss_avg[0] + ss_avg[2]) / 2.0f;

    uart_write_string("\r\n  Balance (|x| < 30 us is balanced):\r\n");
    uart_write_string("    Roll : left - right = ");
    print_float3(roll_bal);
    if (FABSF(roll_bal) < 30.0f)      uart_write_string(" us  (balanced)\r\n");
    else if (roll_bal > 0.0f)         uart_write_string(" us  → would roll RIGHT (right side down)\r\n");
    else                              uart_write_string(" us  → would roll LEFT (left side down)\r\n");

    uart_write_string("    Pitch: front - rear = ");
    print_float3(pitch_bal);
    if (FABSF(pitch_bal) < 30.0f)     uart_write_string(" us  (balanced)\r\n");
    else if (pitch_bal > 0.0f)        uart_write_string(" us  → would pitch NOSE UP\r\n");
    else                              uart_write_string(" us  → would pitch NOSE DOWN\r\n");

    uart_write_string("    Yaw  : CW - CCW     = ");
    print_float3(yaw_bal);
    if (FABSF(yaw_bal) < 30.0f)       uart_write_string(" us  (balanced)\r\n");
    else if (yaw_bal > 0.0f)          uart_write_string(" us  → would yaw CCW (seen from above)\r\n");
    else                              uart_write_string(" us  → would yaw CW (seen from above)\r\n");
}


/* ============================================================
 * Helpers shared by tests 8, 9 and 10
 * ============================================================ */

static const char *const MOTOR_NAMES[4] = { "FL", "FR", "RR", "RL" };

/* Bias-calibrate on a level, still board (same method as main.c), then
 * re-settle the AHRS with the corrected data. */
static void calibrate_level(imu_type_t imu, float gb[3], float ab[3])
{
    float sg[3] = {0}, sa[3] = {0};
    int n = 0;

    ahrs_init();
    uart_write_string("  Calibrating (2 sec, keep board FLAT and STILL)...\r\n");
    for (int i = 0; i < 200; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, NULL, NULL, g, a);
        ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);
        if (i >= 50) {
            for (int k = 0; k < 3; k++) { sg[k] += g[k]; sa[k] += a[k]; }
            n++;
        }
    }
    for (int k = 0; k < 3; k++) { gb[k] = sg[k] / (float)n; ab[k] = sa[k] / (float)n; }
    ab[2] -= 1.0f;                       /* level board should read +1 g on Z */

    ahrs_init();
    track_ticks(imu, gb, ab, 100);
    uart_write_string("  Calibrated. R=");
    print_float3(ahrs_get_roll());
    uart_write_string(" P=");
    print_float3(ahrs_get_pitch());
    uart_write_string("\r\n");
}

/* Bench observation vector (no baro, hvel = 0), as in main.c. */
static void build_obs_bench(const float g[3], const float a[3],
                            float roll, float pitch, float obs[12])
{
    for (int k = 0; k < 3; k++) {
        obs[k]     = clampf(g[k] / MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3 + k] = clampf(a[k] / ACCEL_OBS_SCALE_G, -1.0f, 1.0f);
    }
    obs[6] = clampf(roll  * ANGLE_OBS_SCALE, -1.0f, 1.0f);
    obs[7] = clampf(pitch * ANGLE_OBS_SCALE, -1.0f, 1.0f);
    obs[8] = 0.0f; obs[9] = 0.0f; obs[10] = 0.0f; obs[11] = 0.0f;
}

/* One control tick: read IMU, update AHRS, run the NN.
 * nn[] = clipped raw NN output (no filter). Returns roll/pitch in degrees. */
static void nn_tick(imu_type_t imu, const float gb[3], const float ab[3],
                    float nn[4], float *roll, float *pitch)
{
    float g[3], a[3], obs[12];
    wait_tick();
    read_body(imu, gb, ab, g, a);
    ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);
    *roll  = ahrs_get_roll();
    *pitch = ahrs_get_pitch();
    build_obs_bench(g, a, *roll, *pitch, obs);
    nn_forward(obs, nn);
    for (int k = 0; k < 4; k++) nn[k] = clampf(nn[k], -1.0f, 1.0f);
}

static float nn_to_pwm(float nn)
{
    return 1000.0f + 1000.0f * clampf(NN_HOVER_THROTTLE + nn * NN_ACTION_SCALE, 0.0f, 1.0f);
}

/* Average NN output over `ticks` with the board held level. */
static void level_baseline(imu_type_t imu, const float gb[3], const float ab[3],
                           int ticks, float base_nn[4])
{
    float sum[4] = {0};
    for (int i = 0; i < ticks; i++) {
        float nn[4], r, p;
        nn_tick(imu, gb, ab, nn, &r, &p);
        for (int k = 0; k < 4; k++) sum[k] += nn[k];
    }
    for (int k = 0; k < 4; k++) base_nn[k] = sum[k] / (float)ticks;
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  █████
 *     ██    ██      ██         ██    ██   ██
 *     ██    █████   ███████    ██     █████
 *     ██    ██           ██    ██    ██   ██
 *     ██    ███████ ███████    ██     █████
 *
 *  LIVE TILT → MOTOR RESPONSE VERIFICATION
 *
 *  Purpose: Tilt the board and check the NN pushes the LOW side up.
 *           If it pushes the wrong side, the drone flips instead of
 *           correcting. NO MOTORS ARE DRIVEN.
 *
 *  Physics (training convention, X fwd / Y left / Z up):
 *    right side DOWN (roll > 0)  → RIGHT pair (FR, RR) must speed up
 *    nose DOWN       (pitch > 0) → FRONT pair (FL, FR) must speed up
 *    nose UP         (pitch < 0) → REAR  pair (RR, RL) must speed up
 *
 *  Method: the NN has a built-in imbalance even when level (it was
 *  trained with inputs the drone never supplies), so absolute PWMs are
 *  misleading. The test first records a LEVEL BASELINE, then reports the
 *  CHANGE per motor when tilted, and grades the change.
 *
 *  (The old version compared absolute averages and expected "nose down
 *   → rear speeds up", which is physically backwards: it graded the
 *   pitch-inverted firmware as PASS.)
 *
 *  PASS if the correct pair rises ≥ 20 µs more than the opposite pair.
 * ============================================================ */

/* axis: 0 = roll, 1 = pitch. sign: +1 selects angle > +10°, -1 < -10°.
 * Returns number of tilted samples; d[4] = average PWM change vs base. */
static int tilt_response_phase(imu_type_t imu, const float gb[3], const float ab[3],
                               const float base_nn[4], int axis, int sign,
                               float d[4])
{
    countdown_tracking(imu, gb, ab);
    uart_write_string("TILT AND HOLD\r\n\r\n");

    float sum[4] = {0};
    int count = 0;
    float peak_roll = 0.0f, peak_pitch = 0.0f;   /* largest |angle| seen, signed */

    for (int i = 0; i < 300; i++) {
        float nn[4], roll, pitch;
        nn_tick(imu, gb, ab, nn, &roll, &pitch);
        float angle = (axis == 0) ? roll : pitch;
        if (FABSF(roll)  > FABSF(peak_roll))  peak_roll  = roll;
        if (FABSF(pitch) > FABSF(peak_pitch)) peak_pitch = pitch;

        if ((float)sign * angle > 10.0f) {
            for (int k = 0; k < 4; k++) sum[k] += nn_to_pwm(nn[k]) - nn_to_pwm(base_nn[k]);
            count++;
        }

        if (i % 30 == 0) {
            uart_write_string("  R=");  print_float3(roll);
            uart_write_string(" P=");   print_float3(pitch);
            for (int k = 0; k < 4; k++) {
                uart_write_char(' ');
                uart_write_string(MOTOR_NAMES[k]);
                uart_write_char(':');
                uart_write_uint32((uint32_t)(nn_to_pwm(nn[k]) + 0.5f));
            }
            uart_write_string("\r\n");
        }
    }

    for (int k = 0; k < 4; k++) d[k] = count ? sum[k] / (float)count : 0.0f;

    /* Not enough tilt on the expected axis: say what the board DID feel,
     * so "held it turned 90 degrees" / "FC mounted rotated" is obvious. */
    if (count <= 10) {
        float pr = peak_roll, pp = peak_pitch;
        float big = FABSF(pr) > FABSF(pp) ? FABSF(pr) : FABSF(pp);
        uart_write_string("\r\n  The board felt: ");
        if (big < 10.0f) {
            uart_write_string("less than 10 deg of tilt on either axis.\r\n");
        } else {
            if (FABSF(pr) >= FABSF(pp))
                uart_write_string(pr > 0 ? "RIGHT SIDE DOWN" : "LEFT SIDE DOWN");
            else
                uart_write_string(pp > 0 ? "NOSE DOWN" : "NOSE UP");
            uart_write_string("  (peak roll ");  print_float3(pr);
            uart_write_string(", pitch ");      print_float3(pp);
            uart_write_string(" deg)\r\n");
            uart_write_string("  That is not the tilt asked for. If you DID tilt as asked, the flight\r\n"
                              "  controller is mounted rotated on the frame: its arrow must point to\r\n"
                              "  the nose (firmware assumes the CW180 mount Betaflight reported).\r\n");
        }
    }

    if (count > 10) {
        uart_write_string("\r\n  Change vs level baseline (");
        uart_write_uint32((uint32_t)count);
        uart_write_string(" tilted samples):");
        for (int k = 0; k < 4; k++) {
            uart_write_string("  ");
            uart_write_string(MOTOR_NAMES[k]);
            uart_write_char(' ');
            print_float3(d[k]);
        }
        uart_write_string(" us\r\n");
    }
    return count;
}

/* Grade: `good` = average change of the pair that SHOULD rise minus the
 * opposite pair. */
static void grade_pair(int count, float good, const char *pass_msg,
                       const char *fail_msg, const char *what)
{
    uart_write_string("  Correct pair - opposite pair = ");
    print_float3(good);
    uart_write_string(" us\r\n");
    if (count <= 10) {
        uart_write_string("  [WARN] Not enough tilted samples (need >10 deg): ");
        uart_write_string(what);
        uart_write_string("\r\n");
    } else if (good > 20.0f) {
        print_pass(pass_msg);
    } else if (good < -20.0f) {
        print_fail(fail_msg);
    } else {
        uart_write_string("  [WARN] No clear response. Tilt further (~20 deg) and hold still.\r\n");
    }
}

static void test8_tilt_response(imu_type_t imu)
{
    print_banner("TEST 8: Live Tilt → Motor Response");
    uart_write_string("Tilt the board; checks the NN speeds up the LOW side.\r\n");
    uart_write_string("Graded on the CHANGE from a level baseline.\r\n");
    uart_write_string("NO MOTORS WILL SPIN.\r\n\r\n");

    float gb[3], ab[3], base[4], d[4];
    calibrate_level(imu, gb, ab);

    uart_write_string("  Measuring LEVEL baseline (1 sec, keep FLAT)...\r\n");
    level_baseline(imu, gb, ab, 100, base);
    uart_write_string("  Baseline PWM:");
    for (int k = 0; k < 4; k++) {
        uart_write_string("  ");
        uart_write_string(MOTOR_NAMES[k]);
        uart_write_char(' ');
        uart_write_uint32((uint32_t)(nn_to_pwm(base[k]) + 0.5f));
    }
    uart_write_string(" us\r\n");

    /* ---- A: right side down → right pair (FR, RR) up ---- */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT A: RIGHT SIDE DOWN (~20 deg)\r\n");
    uart_write_string("  Correct response: RIGHT motors (FR, RR) speed up\r\n");
    int n = tilt_response_phase(imu, gb, ab, base, 0, +1, d);
    grade_pair(n, (d[1] + d[2]) / 2.0f - (d[0] + d[3]) / 2.0f,
               "RIGHT DOWN: right motors speed up → corrects",
               "RIGHT DOWN: LEFT motors speed up → ROLL INVERTED, would flip",
               "roll never exceeded +10 deg");

    uart_write_string("\r\n  Return to FLAT (2 sec)...\r\n");
    track_ticks(imu, gb, ab, 200);

    /* ---- B: nose down → front pair (FL, FR) up ---- */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT B: NOSE DOWN (~20 deg)  — lift the REAR\r\n");
    uart_write_string("  Correct response: FRONT motors (FL, FR) speed up\r\n");
    n = tilt_response_phase(imu, gb, ab, base, 1, +1, d);
    grade_pair(n, (d[0] + d[1]) / 2.0f - (d[2] + d[3]) / 2.0f,
               "NOSE DOWN: front motors speed up → lifts nose → corrects",
               "NOSE DOWN: REAR motors speed up → pushes nose further down → PITCH INVERTED",
               "pitch never exceeded +10 deg (nose down is POSITIVE pitch)");

    uart_write_string("\r\n  Return to FLAT (2 sec)...\r\n");
    track_ticks(imu, gb, ab, 200);

    /* ---- C: nose up → rear pair (RR, RL) up ---- */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT C: NOSE UP (~20 deg)  — lift the FRONT\r\n");
    uart_write_string("  Correct response: REAR motors (RR, RL) speed up\r\n");
    n = tilt_response_phase(imu, gb, ab, base, 1, -1, d);
    grade_pair(n, (d[2] + d[3]) / 2.0f - (d[0] + d[1]) / 2.0f,
               "NOSE UP: rear motors speed up → lifts tail → corrects",
               "NOSE UP: FRONT motors speed up → pushes nose further up → PITCH INVERTED",
               "pitch never went below -10 deg (nose up is NEGATIVE pitch)");

    print_banner("TEST 8 COMPLETE");
    uart_write_string("A FAIL on one axis only (roll OR pitch) means that axis's\r\n");
    uart_write_string("sign is inverted (ALIGN_xx in fc_config.h). If test 3 passes\r\n");
    uart_write_string("but this fails, suspect NN_TO_MOTOR[] / motor wiring.\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  █████
 *     ██    ██      ██         ██    ██   ██
 *     ██    █████   ███████    ██     ██████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██     █████
 *
 *  ACTION FILTER LAG TEST
 *
 *  Purpose: Show how far the filtered NN output (what reaches the
 *           motors) trails the raw NN output during a real tilt.
 *
 *  Filter: fixed-alpha EMA from fc_config.h (alpha 0.3, identical to
 *  training). At 100 Hz its time constant is ~28 ms; during a steady
 *  ramp the filtered value trails the raw value by ~2.3 ticks (23 ms).
 *  (The old adaptive filter sat at alpha 0.08 for small changes —
 *   a ~120 ms time constant.)
 *
 *  Informational — prints the largest lag seen, no PASS/FAIL.
 * ============================================================ */
static void test9_iir_lag(imu_type_t imu)
{
    print_banner("TEST 9: Action Filter Lag");
    uart_write_string("Raw NN output vs filtered output (alpha ");
    print_float3(ACTION_FILTER_ALPHA);
    uart_write_string(", same as training).\r\n\r\n");

    float gb[3], ab[3];
    calibrate_level(imu, gb, ab);

    uart_write_string("  Tilt RIGHT ~20 deg during the countdown and hold.\r\n");
    countdown_tracking(imu, gb, ab);
    uart_write_string("TILT NOW\r\n\r\n");

    float nn_smoothed[4] = {0, 0, 0, 0};
    float max_lag = 0.0f;

    uart_write_string("  Roll    | Raw FR   | Filtered FR | Lag(us)\r\n");
    uart_write_string("  --------+----------+-------------+--------\r\n");

    for (int i = 0; i < 500; i++) {
        float nn[4], roll, pitch;
        nn_tick(imu, gb, ab, nn, &roll, &pitch);
        action_filter_step(nn_smoothed, nn);

        float lag_us = (nn[1] - nn_smoothed[1]) * NN_ACTION_SCALE * 1000.0f;
        /* ignore the first 0.2 s while the filter starts from zero */
        if (i >= 20 && FABSF(lag_us) > max_lag) max_lag = FABSF(lag_us);

        if (i % 10 == 0) {
            uart_write_string("  ");
            print_float3(roll);
            uart_write_string(" | ");
            print_float3(nn[1]);
            uart_write_string("   | ");
            print_float3(nn_smoothed[1]);
            uart_write_string("      | ");
            print_float3(lag_us);
            uart_write_string("\r\n");
        }
    }

    uart_write_string("\r\n  Largest |lag| after start-up: ");
    print_float3(max_lag);
    uart_write_string(" us\r\n");
    uart_write_string("  Roll should go POSITIVE for right-side-down. With alpha 0.3\r\n");
    uart_write_string("  the filtered value catches up within ~7 ticks (70 ms).\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██████
 *     ██    ██      ██         ██    ███ ██  ██ ██
 *     ██    █████   ███████    ██     ██ ██  ██
 *     ██    ██           ██    ██     ██ ██  ██
 *     ██    ███████ ███████    ██     ██  ██████
 *
 *  MOTOR WIRE VERIFICATION (MOTORS SPIN!)  — opt-in: TESTS=10
 *
 *  ** REMOVE PROPELLERS BEFORE RUNNING THIS TEST **
 *
 *  You tilt the board; the test pulses the motor whose NN output rose
 *  the MOST compared with the level baseline. You verify it is on the
 *  LOW side (the side that must be lifted):
 *    right side down → FR or RR
 *    nose down       → FL or FR   (front)
 *    nose up         → RR or RL   (rear)
 *
 *  (The old version picked the largest ABSOLUTE output — RL wins even at
 *   level because of the NN's built-in imbalance — and expected the rear
 *   to pulse for nose-down, which is physically backwards.)
 * ============================================================ */

/* NN to motor index mapping — same as main.c */
static const int TEST_NN_TO_MOTOR[4] = {
    3,   /* nn[0]=FL(CCW) → idx3 → PA9 */
    2,   /* nn[1]=FR(CW)  → idx2 → PA8 */
    1,   /* nn[2]=RR(CCW) → idx1 → PC9 */
    0    /* nn[3]=RL(CW)  → idx0 → PC8 */
};

static void test10_motor_wire_verify(imu_type_t imu)
{
    print_banner("TEST 10: Motor Wire Verification");
    uart_write_string("** REMOVE ALL PROPELLERS BEFORE CONTINUING **\r\n");
    uart_write_string("** MOTORS WILL SPIN IN THIS TEST **\r\n\r\n");
    uart_write_string("Starting in 5 seconds (unplug the battery NOW to skip)...\r\n");
    delay_ms(5000);

    motor_init();
    delay_ms(2000);  /* ESC arming */

    float gb[3], ab[3], base[4];
    calibrate_level(imu, gb, ab);
    uart_write_string("  Measuring LEVEL baseline (1 sec, keep FLAT)...\r\n");
    level_baseline(imu, gb, ab, 100, base);

    static const struct {
        const char *label;
        const char *instruction;
        int         axis;          /* 0=roll, 1=pitch */
        float       threshold;     /* trigger: angle beyond this (sign matters) */
        const char *expected_side;
    } phases[3] = {
        {"RIGHT SIDE DOWN", "Tilt RIGHT side down ~20 deg", 0, +15.0f, "RIGHT side (FR or RR)"},
        {"NOSE DOWN",       "Tilt NOSE DOWN ~20 deg",       1, +15.0f, "FRONT (FL or FR)"},
        {"NOSE UP",         "Tilt NOSE UP ~20 deg",         1, -15.0f, "REAR (RR or RL)"},
    };

    for (int p = 0; p < 3; p++) {
        uart_write_string("\r\n========================================\r\n");
        uart_write_string(phases[p].label);
        uart_write_string("\r\n  ");
        uart_write_string(phases[p].instruction);
        uart_write_string("\r\n  Expected motor: ");
        uart_write_string(phases[p].expected_side);
        uart_write_string("\r\n");
        countdown_tracking(imu, gb, ab);
        uart_write_string("GO (8 s window)\r\n\r\n");

        int fired = 0;
        for (int i = 0; i < 800 && !fired; i++) {
            float nn[4], roll, pitch;
            nn_tick(imu, gb, ab, nn, &roll, &pitch);
            float angle = (phases[p].axis == 0) ? roll : pitch;

            if (i % 50 == 0) {
                uart_write_string("  R=");  print_float3(roll);
                uart_write_string(" P=");   print_float3(pitch);
                uart_write_string("  (waiting for ");
                print_float3(phases[p].threshold);
                uart_write_string(" deg)\r\n");
            }

            int triggered = (phases[p].threshold > 0) ? (angle > phases[p].threshold)
                                                      : (angle < phases[p].threshold);
            if (!triggered) continue;

            /* Motor whose output ROSE the most relative to level */
            int strongest = 0;
            for (int k = 1; k < 4; k++)
                if (nn[k] - base[k] > nn[strongest] - base[strongest]) strongest = k;
            int motor_idx = TEST_NN_TO_MOTOR[strongest];

            uart_write_string("\r\n  ** TRIGGERED at ");
            print_float3(angle);
            uart_write_string(" deg **\r\n  Change vs level:");
            for (int k = 0; k < 4; k++) {
                uart_write_string("  ");
                uart_write_string(MOTOR_NAMES[k]);
                uart_write_char(' ');
                print_float3(nn[k] - base[k]);
            }
            uart_write_string("\r\n  Largest increase: ");
            uart_write_string(MOTOR_NAMES[strongest]);
            uart_write_string(" → motor_set(");
            uart_write_uint32((uint32_t)motor_idx);
            uart_write_string(")\r\n  Expected position: ");
            uart_write_string(phases[p].expected_side);
            uart_write_string("\r\n\r\n");

            uart_write_string("  >>> PULSING MOTOR NOW — watch which one spins <<<\r\n");
            for (int pulse = 0; pulse < 3; pulse++) {
                motor_set(motor_idx, 0.30f);
                delay_ms(400);
                motor_all_stop();
                delay_ms(300);
            }
            uart_write_string("  >>> DONE — was it on the ");
            uart_write_string(phases[p].expected_side);
            uart_write_string("? <<<\r\n");
            fired = 1;
        }

        if (!fired)
            uart_write_string("  [WARN] Tilt threshold not reached in 8 seconds.\r\n");

        uart_write_string("\r\n  Return board to FLAT (3 sec)...\r\n");
        track_ticks(imu, gb, ab, 300);
    }

    motor_all_stop();

    print_banner("TEST 10 COMPLETE");
    uart_write_string("For each tilt, verify the pulsing motor is at the\r\n");
    uart_write_string("expected position on the frame.\r\n\r\n");
    uart_write_string("If the wrong motor pulsed but test 8 passed:\r\n");
    uart_write_string("  1. Note which nn output → which motor index → which pin\r\n");
    uart_write_string("  2. Swap ESC signal wires, OR fix NN_TO_MOTOR[] in main.c\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██
 *     ██    ██      ██         ██    ███  ██
 *     ██    █████   ███████    ██     ██  ██
 *     ██    ██           ██    ██     ██  ██
 *     ██    ███████ ███████    ██     ██  ██
 *
 *  ACTION FILTER MATHS  (pure computation — no hardware)
 *
 *  Verifies the on-chip action_filter_step() (fc_config.h):
 *    A: step 0 → 0.5 reaches 90 % in exactly 7 ticks
 *       (alpha 0.3: 1 - 0.7^7 = 0.918 ≥ 0.9, 1 - 0.7^6 = 0.882 < 0.9)
 *    B: matches the training formula  s = 0.3*a + 0.7*s  on a long
 *       pseudo-random sequence (max difference < 1e-5)
 *    C: chatter attenuation — a ±0.08 square wave at 25 Hz (vibration-
 *       like) is reported in µs before and after the filter (info)
 *
 *  (Replaces the old adaptive-filter limit-cycle test; the adaptive
 *   filter was removed from main.c.)
 * ============================================================ */
static void test11_iir_threshold_stress(void)
{
    print_banner("TEST 11: Action Filter Maths");
    uart_write_string("  ACTION_FILTER_ALPHA = ");
    print_float3(ACTION_FILTER_ALPHA);
    uart_write_string("  (training uses 0.300)\r\n\r\n");

    /* ---- A: step response ---- */
    {
        float s[4] = {0, 0, 0, 0}, in[4] = {0.5f, 0.5f, 0.5f, 0.5f};
        int ticks90 = -1;
        uart_write_string("Scenario A: step 0 → 0.5\r\n");
        for (int t = 1; t <= 20; t++) {
            action_filter_step(s, in);
            if (t <= 8) {
                uart_write_string("  tick ");
                uart_write_uint32((uint32_t)t);
                uart_write_string(": ");
                print_float3(s[0]);
                uart_write_string("\r\n");
            }
            if (ticks90 < 0 && s[0] >= 0.45f) ticks90 = t;
        }
        uart_write_string("  90% reached at tick ");
        uart_write_int32(ticks90);
        uart_write_string(" (expect 7 = 70 ms)\r\n");
        if (ticks90 == 7) print_pass("Scenario A: step response matches alpha 0.3");
        else              print_fail("Scenario A: step response does not match alpha 0.3");
    }

    /* ---- B: identical to training formula ---- */
    {
        float s[4] = {0, 0, 0, 0};
        float ref = 0.0f, worst = 0.0f;
        uint32_t lcg = 12345U;
        for (int t = 0; t < 2000; t++) {
            lcg = lcg * 1664525U + 1013904223U;
            float a = ((float)(lcg >> 8) / 8388608.0f) - 1.0f;   /* [-1, 1) */
            float in[4] = {a, a, a, a};
            action_filter_step(s, in);
            ref = 0.3f * a + 0.7f * ref;                          /* drone_env.py */
            float e = FABSF(s[0] - ref);
            if (e > worst) worst = e;
        }
        uart_write_string("\r\nScenario B: 2000 random inputs vs training formula\r\n");
        uart_write_string("  max |firmware - training| = ");
        print_float3(worst * 1e6f);
        uart_write_string("e-6\r\n");
        if (worst < 1e-5f) print_pass("Scenario B: identical to training filter");
        else               print_fail("Scenario B: differs from training filter");
    }

    /* ---- C: chatter attenuation (informational) ---- */
    {
        float s[4] = {0.15f, 0.15f, 0.15f, 0.15f};
        float lo = 1e9f, hi = -1e9f;
        for (int t = 0; t < 400; t++) {
            float a = ((t / 2) % 2) ? 0.23f : 0.07f;   /* 25 Hz square, ±0.08 */
            float in[4] = {a, a, a, a};
            action_filter_step(s, in);
            if (t >= 100) { if (s[0] < lo) lo = s[0]; if (s[0] > hi) hi = s[0]; }
        }
        uart_write_string("\r\nScenario C: ±0.08 square wave at 25 Hz\r\n");
        uart_write_string("  input  peak-to-peak = ");
        print_float3(0.16f * NN_ACTION_SCALE * 1000.0f);
        uart_write_string(" us\r\n  output peak-to-peak = ");
        print_float3((hi - lo) * NN_ACTION_SCALE * 1000.0f);
        uart_write_string(" us  (info)\r\n");
    }
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██████
 *     ██    ██      ██         ██    ███ ██    ██
 *     ██    █████   ███████    ██     ██ ██████
 *     ██    ██           ██    ██     ██      ██
 *     ██    ███████ ███████    ██     ██ ██████
 *
 *  LOOP TIMING JITTER & OVERRUN DETECTOR
 *
 *  Purpose: Measure the real tick-period variance to catch:
 *
 *    OVERRUN: tick_period > 10500 µs means the loop body took
 *    longer than the 10 ms SysTick period. The COUNTFLAG had
 *    already fired by the time we polled it — the NEXT tick
 *    fires immediately (0 µs gap), giving the AHRS two
 *    back-to-back updates with near-zero real dt.
 *    Result: gyro integration spike → attitude error → NN
 *    commands a large correction → motor pulse.
 *
 *    JITTER: the baro I2C read (every 4th tick) takes ~300-500 µs
 *    on-chip but is blocking. This makes every 4th tick ~4-5%
 *    longer. With hardcoded dt=0.01f the AHRS under-integrates
 *    on baro ticks. At 168 MHz the budget is comfortable, but
 *    an I2C clock-stretch or NACK-retry can push it over.
 *
 *  Pass criteria:
 *    - Zero overruns (no tick_period > 10500 µs)
 *    - Jitter (max-min period) < 600 µs
 *    - Baro read < 1500 µs
 *
 * ============================================================ */
static void test12_loop_jitter(imu_type_t imu, int baro_ok)
{
    print_banner("TEST 12: Loop Timing Jitter & Overrun Detector");
    uart_write_string("Measures real SysTick period variance and per-phase cost.\r\n");
    uart_write_string("Overruns cause AHRS spikes → NN commands → motor pulses.\r\n\r\n");
    uart_write_string("Board flat. Running 500 ticks (5 sec)...\r\n\r\n");

    dwt_init();
    ahrs_init();
    systick_init_100hz();

    /* Burn the stale COUNTFLAG set by systick_init_100hz */
    while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

    uint32_t prev_cyc    = dwt_now();
    uint32_t max_per     = 0;
    uint32_t min_per     = 0xFFFFFFFFU;
    uint32_t period_sum  = 0;
    uint32_t overruns    = 0;
    uint32_t max_imu_us  = 0;
    uint32_t max_ahrs_us = 0;
    uint32_t max_baro_us = 0;
    uint32_t max_tot_us  = 0;

    for (int tick = 0; tick < 500; tick++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        uint32_t t_start   = dwt_now();
        uint32_t period_us = cycles_to_us(t_start - prev_cyc);
        prev_cyc = t_start;

        period_sum += period_us;
        if (period_us > max_per) max_per = period_us;
        if (period_us < min_per) min_per = period_us;
        if (period_us > 10500)   overruns++;

        /* ── A: IMU read ── */
        uint32_t t0 = dwt_now();
        imu_raw_t s;
        imu_read_raw(imu, &s);
        uint32_t imu_us = cycles_to_us(dwt_now() - t0);
        if (imu_us > max_imu_us) max_imu_us = imu_us;

        /* ── B: AHRS ── */
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
        uint32_t ta = dwt_now();
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
        uint32_t ahrs_us = cycles_to_us(dwt_now() - ta);
        if (ahrs_us > max_ahrs_us) max_ahrs_us = ahrs_us;

        /* ── C: Baro (blocks every 4th tick for I2C) ── */
        uint32_t baro_us = 0;
        if (baro_ok) {
            uint32_t tb = dwt_now();
            baro_update(0.01f);
            baro_us = cycles_to_us(dwt_now() - tb);
            if (baro_us > max_baro_us) max_baro_us = baro_us;
        }

        uint32_t tot_us = cycles_to_us(dwt_now() - t_start);
        if (tot_us > max_tot_us) max_tot_us = tot_us;

        /* Print at 1 Hz (every 100 ticks) */
        if (tick % 100 == 0) {
            uart_write_string("  [");
            uart_write_uint32(tick);
            uart_write_string("] period="); uart_write_uint32(period_us);
            uart_write_string(" imu=");    uart_write_uint32(imu_us);
            uart_write_string(" ahrs=");   uart_write_uint32(ahrs_us);
            if (baro_ok) { uart_write_string(" baro="); uart_write_uint32(baro_us); }
            uart_write_string(" tot=");    uart_write_uint32(tot_us);
            uart_write_string(" us\r\n");
        }
    }

    uint32_t avg_per = period_sum / 500;
    uint32_t jitter  = max_per - min_per;

    uart_write_string("\r\n---- TIMING SUMMARY (500 ticks) ----\r\n\r\n");
    uart_write_string("  Tick period:   avg="); uart_write_uint32(avg_per);
    uart_write_string(" min="); uart_write_uint32(min_per);
    uart_write_string(" max="); uart_write_uint32(max_per);
    uart_write_string(" jitter="); uart_write_uint32(jitter);
    uart_write_string(" us\r\n");
    uart_write_string("  Overruns (>10500us): "); uart_write_uint32(overruns);
    uart_write_string("\r\n");
    uart_write_string("  Worst IMU read:   "); uart_write_uint32(max_imu_us);  uart_write_string(" us\r\n");
    uart_write_string("  Worst AHRS:       "); uart_write_uint32(max_ahrs_us); uart_write_string(" us\r\n");
    if (baro_ok) {
        uart_write_string("  Worst baro read:  "); uart_write_uint32(max_baro_us);
        uart_write_string(" us  (fires every 4 ticks — baro I2C blocking cost)\r\n");
    }
    uart_write_string("  Worst total tick: "); uart_write_uint32(max_tot_us);
    uart_write_string(" us  (budget = 10000 us)\r\n");
    uart_write_string("  Budget margin:    ");
    if (max_tot_us < 10000) {
        uart_write_uint32(10000 - max_tot_us);
        uart_write_string(" us free\r\n");
    } else {
        uart_write_string("OVER BUDGET\r\n");
    }

    uart_write_string("\r\n---- DIAGNOSTICS ----\r\n\r\n");

    if (overruns == 0)
        print_pass("No tick overruns — loop body fits within 10 ms period");
    else {
        print_fail("Tick overruns detected — loop is missing SysTick deadlines");
        uart_write_string("  Root cause: AHRS gets back-to-back updates with ~0 real dt\r\n");
        uart_write_string("  on overrun ticks → gyro integration spike → NN correction spike.\r\n");
        uart_write_string("  Fix: use DWT timestamps for AHRS dt instead of hardcoded 0.01f.\r\n");
    }

    if (jitter < 600)
        print_pass("Tick jitter < 600 us — timing is clean");
    else {
        print_fail("Jitter > 600 us — something is blocking the loop inconsistently");
        uart_write_string("  Most likely: I2C baro taking variable time (clock-stretch / retry).\r\n");
        uart_write_string("  Check pull-up resistors on PB8/PB9 should be 4.7k.\r\n");
    }

    if (baro_ok) {
        if (max_baro_us < 1500)
            print_pass("Baro I2C read < 1500 us — within loop budget");
        else {
            print_fail("Baro I2C read > 1500 us — contributing to jitter");
            uart_write_string("  Try: reduce DPS/BMP oversampling, or increase BARO_DECIMATION.\r\n");
        }
    }

    uart_write_string("\r\n  Note: NN forward pass (~15 us) not included here. Add to tot.\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██████
 *     ██    ██      ██         ██    ███ ██    ██
 *     ██    █████   ███████    ██      ██ ██████
 *     ██    ██           ██    ██      ██      ██
 *     ██    ███████ ███████    ██     ██  ██████
 *
 *  BARO THROTTLE NOISE CHARACTERIZATION
 *
 *  Purpose: Quantify how much noise the baro P+D altitude
 *           controller injects into the throttle command.
 *
 *  alt_throttle_adj = Kp*(target-alt) + Kd*(-vvel)
 *
 *  The KD term differentiates altitude at 25 Hz. In flight,
 *  motor downwash oscillates the local pressure heavily —
 *  the vvel estimate can swing ±0.5 m/s even when stationary.
 *  With ALT_KD = 0.02 (fc_config.h, same as main.c) that's ±10 µs of
 *  throttle oscillation at 25 Hz. (The harness used to simulate 0.08,
 *  four times the real firmware gain.)
 *
 *  Procedure: Board flat and still for 5 seconds.
 *
 *  Pass criteria:
 *    alt_throttle_adj peak-to-peak range < 20 µs  (±0.010)
 *
 *  If baro not present: test prints reference values only.
 *
 * ============================================================ */
static void test13_baro_throttle_noise(imu_type_t imu, int baro_ok)
{
    print_banner("TEST 13: Baro Throttle Noise Characterization");
    uart_write_string("Measures throttle noise injected by the baro P+D controller.\r\n");
    uart_write_string("Large values here = direct cause of motor pulsing.\r\n\r\n");

    if (!baro_ok) {
        uart_write_string("  No barometer detected — skipping live measurement.\r\n\r\n");
        uart_write_string("  Reference: when baro IS present, bench-rest expected values:\r\n");
        uart_write_string("    alt_throttle_adj range: < ±0.010 (~±10 us)  PASS\r\n");
        uart_write_string("    alt_throttle_adj range: > ±0.050 (~±50 us)  FAIL\r\n");
        uart_write_string("  In flight with prop downwash, vvel noise is 10-20x worse.\r\n");
        uart_write_string("  Fix: reduce ALT_KD in fc_config.h, or VVEL_ALPHA in baro.c.\r\n");
        return;
    }

    /* ── Step 1: Settle baro and set ground reference ── */
    uart_write_string("  Settling baro (2 sec)...\r\n");
    for (int i = 0; i < 50; i++) {
        baro_update(0.04f);
        delay_ms(40);
    }
    baro_set_ground_reference();
    uart_write_string("  Ground reference set. Bench offset will appear as DC in Kp term.\r\n\r\n");

    /* ── Step 2: Settle AHRS (1 sec) ── */
    ahrs_init();
    systick_init_100hz();
    uart_write_string("  AHRS settling (1 sec)...\r\n");
    for (int i = 0; i < 100; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        ahrs_update(
            (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS,
            (float)ALIGN_AX(s.ax) * accel_raw_to_g,
            (float)ALIGN_AY(s.ay) * accel_raw_to_g,
            (float)ALIGN_AZ(s.az) * accel_raw_to_g,
            0.01f);
        baro_update(0.01f);
    }

    /* ── Step 3: Sample 500 ticks (5 sec) ── */
    uart_write_string("  Sampling 5 sec — board FLAT, DO NOT MOVE.\r\n\r\n");
    uart_write_string("  alt(m) | vvel(m/s) | Kp_term  | Kd_term  | adj_total\r\n");
    uart_write_string("  -------+-----------+----------+----------+----------\r\n");

    float sum_adj   = 0.0f;
    float sum_adj2  = 0.0f;
    float min_adj   =  999.0f;
    float max_adj   = -999.0f;
    float min_vvel  =  999.0f;
    float max_vvel  = -999.0f;
    int   n = 0;

    for (int tick = 0; tick < 500; tick++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        imu_raw_t s;
        imu_read_raw(imu, &s);
        ahrs_update(
            (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS,
            (float)ALIGN_AX(s.ax) * accel_raw_to_g,
            (float)ALIGN_AY(s.ay) * accel_raw_to_g,
            (float)ALIGN_AZ(s.az) * accel_raw_to_g,
            0.01f);
        baro_update(0.01f);

        float alt  = baro_get_altitude();
        float vvel = baro_get_vvel();

        /* Replicate main.c alt_throttle_adj exactly */
        float kp_term = HIL_ALT_KP * (HIL_TARGET_ALT - alt);
        float kd_term = HIL_ALT_KD * (-vvel);
        float adj     = kp_term + kd_term;
        if (adj >  HIL_ALT_ADJ_MAX) adj =  HIL_ALT_ADJ_MAX;
        if (adj < -HIL_ALT_ADJ_MAX) adj = -HIL_ALT_ADJ_MAX;

        sum_adj  += adj;
        sum_adj2 += adj * adj;
        if (adj  < min_adj)  min_adj  = adj;
        if (adj  > max_adj)  max_adj  = adj;
        if (vvel < min_vvel) min_vvel = vvel;
        if (vvel > max_vvel) max_vvel = vvel;
        n++;

        /* Print at 10 Hz (every 10 ticks) */
        if (tick % 10 == 0) {
            uart_write_string("  ");
            print_float3(alt);      uart_write_string(" | ");
            print_float3(vvel);     uart_write_string(" | ");
            print_float3(kp_term);  uart_write_string(" | ");
            print_float3(kd_term);  uart_write_string(" | ");
            print_float3(adj);
            uart_write_string("\r\n");
        }
    }

    float avg_adj = sum_adj / (float)n;
    float var_adj = sum_adj2 / (float)n - avg_adj * avg_adj;
    float std_adj;
    __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(std_adj) : "t"(var_adj));

    /* Express in µs for intuition */
    float range_us = (max_adj - min_adj) * 1000.0f;
    float vvel_range = max_vvel - min_vvel;
    float kd_noise_us = HIL_ALT_KD * vvel_range * 1000.0f;

    uart_write_string("\r\n---- BARO THROTTLE NOISE RESULTS ----\r\n\r\n");
    uart_write_string("  alt_throttle_adj (throttle fraction):\r\n");
    uart_write_string("    avg=");    print_float3(avg_adj);
    uart_write_string("  min=");     print_float3(min_adj);
    uart_write_string("  max=");     print_float3(max_adj);
    uart_write_string("  stddev=");  print_float3(std_adj);
    uart_write_string("\r\n");
    uart_write_string("    peak-to-peak = ");
    uart_write_uint32((uint32_t)(range_us + 0.5f));
    uart_write_string(" us  (direct PWM oscillation caused by baro)\r\n");

    uart_write_string("\r\n  vvel range: min=");  print_float3(min_vvel);
    uart_write_string(" max="); print_float3(max_vvel);
    uart_write_string(" m/s\r\n");
    uart_write_string("  KD noise contribution: ALT_KD * vvel_range * 1000 = ");
    uart_write_uint32((uint32_t)(kd_noise_us + 0.5f));
    uart_write_string(" us\r\n");

    uart_write_string("\r\n---- DIAGNOSTICS ----\r\n\r\n");

    if (range_us < 20.0f)
        print_pass("Baro throttle noise < 20 us — not causing pulsing");
    else if (range_us < 50.0f) {
        uart_write_string("  [WARN] Baro throttle range = ");
        uart_write_uint32((uint32_t)(range_us + 0.5f));
        uart_write_string(" us — marginal on bench.\r\n");
        uart_write_string("  In flight, motor downwash will make this 5-10x worse.\r\n");
    } else {
        print_fail("Baro throttle noise > 50 us — direct cause of motor pulsing");
        uart_write_string("  The ALT_KD term is amplifying baro vvel noise.\r\n");
        uart_write_string("  Fixes in order of priority:\r\n");
        uart_write_string("    1. Reduce ALT_KD in fc_config.h (currently 0.02)\r\n");
        uart_write_string("    2. Reduce VVEL_ALPHA in baro.c (currently 0.1)\r\n");
        uart_write_string("    3. Increase BARO_DECIMATION: 4 → 8 in baro.c (12.5 Hz)\r\n");
    }

    if (kd_noise_us > range_us * 0.6f)
        uart_write_string("  KD term is >60% of noise — reduce ALT_KD first.\r\n");
    else
        uart_write_string("  KP term dominates (drone not at 0.3m on bench — DC offset expected).\r\n");
}


/* ============================================================
 * TEST 14: Throttle Staircase — REAL hover point + vibration
 * ============================================================
 *
 * THE most important unknown: at what PWM does this aircraft
 * actually hover? The sim assumes a LINEAR thrust map where
 * hover = 0.3226 → 1323 µs. Real ESC+prop combos are nonlinear;
 * if the true hover point is above ~1420 µs (base + P-loop cap),
 * the firmware can NEVER sustain altitude — no NN can fix that.
 *
 * ── SETUP (required) ───────────────────────────────────────
 *   1. PROPS ON (this measures real thrust).
 *   2. STRAP the drone to a kitchen scale with tape/zip ties,
 *      or strap it to a heavy bench plate. It must NOT be able
 *      to lift off or tip. Cable-tie through the frame arms.
 *   3. Note the scale reading at idle (= drone+strap weight).
 *   4. At each throttle step, write down the scale reading.
 *      thrust(g) = idle_reading - current_reading
 *      HOVER PWM = the step where thrust ≈ takeoff weight.
 *
 * While stepping, the test ALSO measures what the flight code
 * will experience at each power level:
 *   - accel/gyro noise std (vibration vs the 0.22 g / 0.09 rad/s
 *     the NN was trained with)
 *   - AHRS roll/pitch drift (does Madgwick survive vibration?)
 * ============================================================ */
#define T14_STEP_START_US   1100
#define T14_STEP_END_US     1700
#define T14_STEP_SIZE_US      50
#define T14_STEP_TICKS       300   /* 3 s per step at 100 Hz */
#define T14_ABORT_TILT_DEG    40.0f

static void test14_throttle_staircase(imu_type_t imu)
{
    print_banner("TEST 14: Throttle Staircase (REAL hover point)");
    uart_write_string("** PROPS ON — DRONE MUST BE STRAPPED DOWN / ON A SCALE **\r\n");
    uart_write_string("** IT WILL PULL HARD AT THE TOP STEPS. SECURE IT. **\r\n\r\n");
    uart_write_string("Procedure: note scale reading at idle, then at each step.\r\n");
    uart_write_string("  thrust(g) = idle_reading - reading\r\n");
    uart_write_string("  hover PWM = step where thrust = takeoff weight\r\n\r\n");
    uart_write_string("Starting in 15 seconds (power off NOW to skip)...\r\n");
    for (int i = 15; i > 0; i--) {
        uart_write_uint32((uint32_t)i);
        uart_write_string("... ");
        led_on();  delay_ms(500);
        led_off(); delay_ms(500);
    }
    uart_write_string("\r\n");

    motor_init();
    delay_ms(3000);   /* ESC arming */

    /* AHRS reference at idle */
    ahrs_init();
    systick_init_100hz();
    for (int i = 0; i < 200; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        ahrs_update(
            (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS,
            (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS,
            (float)ALIGN_AX(s.ax) * accel_raw_to_g,
            (float)ALIGN_AY(s.ay) * accel_raw_to_g,
            (float)ALIGN_AZ(s.az) * accel_raw_to_g,
            0.01f);
    }
    float roll_ref  = ahrs_get_roll();
    float pitch_ref = ahrs_get_pitch();

    uart_write_string("\r\n  PWM(us) | az_mean | az_std | gyr_std | AHRS drift(deg)\r\n");
    uart_write_string("  --------+---------+--------+---------+----------------\r\n");

    float worst_drift  = 0.0f;
    float az_std_hover = 0.0f;    /* az_std at the sim's assumed hover PWM */
    int   aborted      = 0;

    for (uint32_t pwm = T14_STEP_START_US;
         pwm <= T14_STEP_END_US && !aborted;
         pwm += T14_STEP_SIZE_US)
    {
        float cmd = (float)(pwm - 1000U) / 1000.0f;
        for (int m = 0; m < 4; m++) motor_set(m, cmd);

        /* 0.5 s settle so ESC/rpm reaches steady state */
        for (int i = 0; i < 50; i++) {
            while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
            imu_raw_t s;
            imu_read_raw(imu, &s);
            ahrs_update(
                (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS,
                (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS,
                (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS,
                (float)ALIGN_AX(s.ax) * accel_raw_to_g,
                (float)ALIGN_AY(s.ay) * accel_raw_to_g,
                (float)ALIGN_AZ(s.az) * accel_raw_to_g,
                0.01f);
        }

        /* Sample this step */
        float sum_az = 0, sum_az2 = 0;
        float sum_g = 0, sum_g2 = 0;   /* pooled gyro x/y/z */
        int   n = 0, ng = 0;

        for (int i = 0; i < T14_STEP_TICKS; i++) {
            while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
            imu_raw_t s;
            imu_read_raw(imu, &s);
            float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS;
            float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS;
            float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS;
            float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g;
            float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g;
            float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g;
            ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

            sum_az += az;  sum_az2 += az * az;  n++;
            sum_g += gx + gy + gz;
            sum_g2 += gx * gx + gy * gy + gz * gz;  ng += 3;

            /* Safety: strapped drone should not tilt. If it does,
             * the strap failed — kill everything immediately. */
            float r = ahrs_get_roll()  - roll_ref;
            float p = ahrs_get_pitch() - pitch_ref;
            if (FABSF(r) > T14_ABORT_TILT_DEG ||
                FABSF(p) > T14_ABORT_TILT_DEG) {
                motor_all_stop();
                uart_write_string("\r\n  !! TILT DETECTED — ABORTED. "
                                  "Strap the drone properly. !!\r\n");
                aborted = 1;
                break;
            }
        }
        if (aborted) break;

        float az_mean = sum_az / (float)n;
        float az_var  = sum_az2 / (float)n - az_mean * az_mean;
        float g_mean  = sum_g / (float)ng;
        float g_var   = sum_g2 / (float)ng - g_mean * g_mean;
        float az_std, g_std;
        __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(az_std) : "t"(az_var));
        __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(g_std)  : "t"(g_var));

        float drift_r = ahrs_get_roll()  - roll_ref;
        float drift_p = ahrs_get_pitch() - pitch_ref;
        float drift = FABSF(drift_r) > FABSF(drift_p)
                        ? FABSF(drift_r) : FABSF(drift_p);
        if (drift > worst_drift) worst_drift = drift;
        if (pwm == 1300 || pwm == 1350) {
            if (az_std > az_std_hover) az_std_hover = az_std;
        }

        uart_write_string("    ");
        uart_write_uint32(pwm);
        uart_write_string("  | ");
        print_float3(az_mean);
        uart_write_string(" | ");
        print_float3(az_std);
        uart_write_string(" | ");
        print_float3(g_std);
        uart_write_string("  | ");
        print_float3(drift);
        uart_write_string("   << WRITE DOWN SCALE READING\r\n");
    }

    motor_all_stop();

    uart_write_string("\r\n---- THROTTLE STAIRCASE RESULTS ----\r\n\r\n");
    uart_write_string("  1. HOVER POINT: find the PWM where your scale showed\r\n");
    uart_write_string("     thrust = takeoff weight. Firmware budget:\r\n");
    {
        uint32_t base = (uint32_t)(1000.0f + NN_HOVER_THROTTLE * 1000.0f + 0.5f);
        uint32_t cap  = base + (uint32_t)(ALT_ADJ_MAX * 1000.0f + 0.5f);
        uart_write_string("       base (NN_HOVER_THROTTLE)  = "); uart_write_uint32(base);
        uart_write_string(" us\r\n       + P-alt loop cap          = "); uart_write_uint32(cap);
        uart_write_string(" us MAX SUSTAINED\r\n");
        uart_write_string("     hover PWM within ~25 us of base → healthy   PASS\r\n");
        uart_write_string("     hover PWM well above base, below cap → sags WARN\r\n");
        uart_write_string("     hover PWM above cap → CANNOT HOVER: fix the thrust\r\n");
        uart_write_string("       curve in fc-nn/drone_env.py and retrain.  FAIL\r\n\r\n");
    }

    uart_write_string("  2. VIBRATION at hover power: az_std = ");
    print_float3(az_std_hover);
    uart_write_string(" g\r\n");
    if (az_std_hover < 0.25f)
        print_pass("within NN training noise (0.22 g)");
    else if (az_std_hover < 0.5f)
        print_fail("2x above training noise — accel LPF + soft-mount needed");
    else
        print_fail(">2x training noise — NN inputs OUT OF DISTRIBUTION in flight");

    uart_write_string("\r\n  3. AHRS integrity: worst attitude drift = ");
    print_float3(worst_drift);
    uart_write_string(" deg (drone physically still)\r\n");
    if (worst_drift < 3.0f)
        print_pass("Madgwick survives vibration");
    else
        print_fail("attitude estimate corrupted by vibration — obs[6..7] are lies in flight");
}


/* ============================================================
 * MAIN
 * ============================================================ */

/* ============================================================
 * TEST 15: RC link + arming logic (motors NEVER driven)
 *
 * Runs main.c's exact RC path — rc_input_poll → crsf decoder → rc_update —
 * for 60 s and prints what the flight firmware would decide, 5× per second:
 *
 *   LQ  link quality      CH1..6  channel µs (what Betaflight showed)
 *   mode / ARMED / blocked-by    cmd roll/pitch/alt setpoints
 *
 * Things to do while it runs (radio ON, battery optional, props OFF):
 *   1. Move each stick and switch; check channel order and direction:
 *        roll stick RIGHT → CH1 up, pitch stick FORWARD → CH2 up,
 *        throttle UP → CH3 up, ARM switch → CH5, MODE switch → CH6.
 *   2. Arm with throttle low, board level → "ARMED". Disarm → "disarmed".
 *   3. Arm with throttle high → refused (thr), and it must NOT arm when
 *      you then lower the throttle — only after an OFF→ON cycle.
 *   4. STICK mode: throttle stick moves "alt" between 0.20 and 0.60 m.
 *      Lean commands are OFF (RC_STICK_LEAN_ENABLE 0) — cmd R/P stay 0.0.
 *      (With it set to 1: full right stick → cmd roll +8.0, forward → pitch +8.0.)
 *   5. While "ARMED", switch the radio OFF → "FAILSAFE" within 0.25 s.
 * ============================================================ */
static void print_block_reasons(int b)
{
    if (!b) { uart_write_string("-"); return; }
    if (b & RC_BLOCK_NO_LINK)      uart_write_string("link ");
    if (b & RC_BLOCK_NOT_CAL)      uart_write_string("cal ");
    if (b & RC_BLOCK_THROTTLE)     uart_write_string("thr ");
    if (b & RC_BLOCK_TILT)         uart_write_string("tilt ");
    if (b & RC_BLOCK_MODE_LAND)    uart_write_string("land ");
    if (b & RC_BLOCK_SWITCH_CYCLE) uart_write_string("cycle-arm-switch ");
}

static void test15_rc_link(imu_type_t imu)
{
    print_banner("TEST 15: RC Link + Arming Logic (motors OFF)");
    uart_write_string("ExpressLRS / CRSF on USART2 (RX2=PA3), 420000 baud.\r\n");
    uart_write_string("Motors are never driven in this test. Radio ON.\r\n");
    uart_write_string("Move sticks + switches, arm/disarm, try arming with throttle up,\r\n");
    uart_write_string("then switch the radio OFF while ARMED to see the failsafe.\r\n");
    uart_write_string(RC_STICK_LEAN_ENABLE ? "Stick lean commands: ENABLED (max 8 deg)\r\n\r\n"
                                           : "Stick lean commands: disabled (STICK mode = throttle -> altitude only)\r\n\r\n");

    static crsf_t     c;
    static rc_state_t rc;
    crsf_init(&c);
    rc_init(&rc);
    rc_input_init();
    ahrs_init();

    uint32_t link_age = RC_LINK_TIMEOUT_TICKS;
    uint32_t max_gap_ticks = 0, gap = 0, link_ticks = 0;
    unsigned arm_events = 0, kills = 0;
    int was_armed = 0;

    for (int tick = 0; tick < 6000; tick++) {          /* 60 s at 100 Hz */
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        imu_raw_t smp;
        imu_read_raw(imu, &smp);
        ahrs_update((float)ALIGN_GX(smp.gx) * GYRO_RAW_TO_RADS,
                    (float)ALIGN_GY(smp.gy) * GYRO_RAW_TO_RADS,
                    (float)ALIGN_GZ(smp.gz) * GYRO_RAW_TO_RADS,
                    (float)ALIGN_AX(smp.ax) * accel_raw_to_g,
                    (float)ALIGN_AY(smp.ay) * accel_raw_to_g,
                    (float)ALIGN_AZ(smp.az) * accel_raw_to_g, 0.01f);

        if (rc_input_poll(&c) > 0) {
            if (link_age < RC_LINK_TIMEOUT_TICKS && gap > max_gap_ticks) max_gap_ticks = gap;
            link_age = 0; gap = 0;
        } else {
            gap++;
            if (link_age < RC_LINK_TIMEOUT_TICKS) link_age++;
        }
        int link_ok = link_age < RC_LINK_TIMEOUT_TICKS;
        link_ticks += (uint32_t)link_ok;

        rc_inputs_t in;
        in.link_ok    = link_ok;
        in.roll       = crsf_to_unit(c.ch[RC_CH_ROLL]);
        in.pitch      = crsf_to_unit(c.ch[RC_CH_PITCH]);
        in.throttle   = crsf_to_unit(c.ch[RC_CH_THROTTLE]);
        in.yaw        = crsf_to_unit(c.ch[RC_CH_YAW]);
        in.arm_sw     = crsf_to_unit(c.ch[RC_CH_ARM]);
        in.mode_sw    = crsf_to_unit(c.ch[RC_CH_MODE]);
        in.calibrated = tick >= 200;                  /* like main.c's 2 s settle */
        in.roll_deg   = ahrs_get_roll();
        in.pitch_deg  = ahrs_get_pitch();
        rc_update(&rc, &in, 0.01f);

        if (rc.armed && !was_armed) {
            arm_events++;
            uart_write_string(">>> ARMED (flight firmware would start the 0.5 s ramp)\r\n");
        }
        if (!rc.armed && was_armed) {
            if (rc.last_disarm == RC_DISARM_FAILSAFE) uart_write_string(">>> FAILSAFE: link lost -> motors OFF\r\n");
            else { kills++; uart_write_string(">>> DISARMED by switch -> motors OFF this tick\r\n"); }
        }
        was_armed = rc.armed;

        if (tick % 20 == 0 && uart_tx_free() > 300U) {
            uart_write_string(link_ok ? "LINK ok  LQ " : "NO LINK  LQ ");
            uart_write_uint32(c.lq);
            uart_write_string(" | CH");
            for (int k = 0; k < 6; k++) {
                uart_write_char(' ');
                uart_write_uint32(crsf_to_us(c.ch[k]));
            }
            uart_write_string(" | ");
            uart_write_string(rc_mode_name(rc.mode));
            uart_write_string(rc.armed ? " ARMED   " : " disarmed");
            uart_write_string(" | block: ");
            print_block_reasons(rc.block);
            uart_write_string("| cmd R ");  print_float3(rc.cmd_roll_deg);
            uart_write_string(" P ");       print_float3(rc.cmd_pitch_deg);
            uart_write_string(" alt ");     print_float3(rc.target_alt);
            uart_write_string(" | att R "); print_float3(in.roll_deg);
            uart_write_string(" P ");       print_float3(in.pitch_deg);
            uart_write_string("\r\n");
        }
    }

    uart_write_string("\r\n---- TEST 15 SUMMARY ----\r\n");
    uart_write_string("  RC frames:      "); uart_write_uint32(c.rc_frames);
    uart_write_string("  (~"); uart_write_uint32(c.rc_frames / 60U); uart_write_string(" Hz)\r\n");
    uart_write_string("  CRC errors:     "); uart_write_uint32(c.crc_errors);
    uart_write_string("   UART overruns: "); uart_write_uint32(rc_input_overruns());
    uart_write_string("\r\n  Link up:        "); uart_write_uint32(link_ticks / 100U);
    uart_write_string(" of 60 s   longest gap while linked: ");
    uart_write_uint32(max_gap_ticks * 10U); uart_write_string(" ms\r\n");
    uart_write_string("  Arms: "); uart_write_uint32(arm_events);
    uart_write_string("   switch disarms: "); uart_write_uint32(kills);
    uart_write_string("   failsafes: "); uart_write_uint32(rc.failsafe_count);
    uart_write_string("\r\n");
    int ok = c.rc_frames > 0 && c.crc_errors * 100U <= c.rc_frames;
    uart_write_string(ok ? "  [PASS] receiver decoded, CRC errors <= 1%\r\n"
                         : "  [FAIL] no RC frames (check wiring RX2=PA3, receiver bound/powered) or >1% CRC errors\r\n");
    uart_write_string("  Manually confirm: channel directions, arm refused with throttle up,\r\n"
                      "  disarm instant, failsafe fired when the radio was switched off.\r\n");
}

/* ============================================================
 * TEST 16: Thrust ramp on a kitchen scale — drone NOT tied down
 * ============================================================
 *
 * Finds the real hover PWM without ballast. The drone stands free on
 * the scale; as the throttle rises the scale reading DROPS:
 *     thrust(g) = reading at 1000 us  -  reading at this step
 * Stop when the scale shows ~25-30 % of its 1000 us reading (the drone
 * is then carrying ~70-75 % of its weight). Hover = extrapolate the
 * last few points to reading 0.
 *
 * First a 5-step radio check (bytes on RX2, CRSF frames, link health,
 * channel values, ARM switch ON->OFF) prints PASS/FAIL + likely cause and
 * stops the test at the first failure. motor_init() is only called after
 * all five pass, so a broken radio can never leave the motors enabled.
 *
 * The radio ARM switch is then a dead-man switch:
 *   - motors only spin after ARM goes OFF -> ON while the link is up
 *   - ARM OFF, link lost (250 ms), tilt > 10 deg, tipping rate,
 *     or liftoff (accel z) -> ALL MOTORS OFF THE SAME 10 ms TICK
 *
 * Profile: 1000 us, then +25 us every 4 s up to T16_MAX_US
 * (default 1500; make flash-test TESTS=16 T16_MAX=1600 to go higher),
 * then a gentle ramp back down. Each line is printed when its step
 * STARTS, so a phone video of screen + scale pairs them up.
 *
 * Setup: battery in, props on, drone standing on a light ~25 cm stand
 * (box/pot) on the scale, props overhanging it, a slack safety string
 * to something heavy that is NOT on the scale. Stand back.
 * ============================================================ */
#ifndef T16_MAX_US
#define T16_MAX_US          1500
#endif
#define T16_STEP_US           25
#define T16_STEP_TICKS       400     /* 4 s per step                       */
#define T16_WAIT_TICKS     18000     /* 3 min to flip the ARM switch       */
#define T16_TILT_STOP_DEG   10.0f
#define T16_RATE_STOP_RADS   0.5f    /* filtered roll/pitch rate            */
/* Liftoff check: 0.5 s average of az must rise this far above the at-rest
 * value. Was 0.12 g with a 0.2 s EMA — tripped at 1200 us with the drone
 * still on the scale: prop vibration (~0.2 g std) sampled at 100 Hz
 * aliases into slow swings the short filter passes. A GENTLE liftoff
 * (thrust just above weight) adds only a few hundredths of a g and can't
 * be seen by the accel at all; this only catches a violent jump. The
 * ARM switch + safety string are the real protection. */
#define T16_LIFT_STOP_G      0.35f
#define T16_LIFT_AVG_TICKS     50

/* One 10 ms tick of test 16 bookkeeping: IMU → AHRS, drain the receiver,
 * age the link. Returns 1 while a valid RC frame arrived within 250 ms. */
static int t16_tick(imu_type_t imu, crsf_t *c, uint32_t *link_age)
{
    float g[3], a[3];
    wait_tick();
    read_body(imu, NULL, NULL, g, a);
    ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);
    if (rc_input_poll(c) > 0) *link_age = 0;
    else if (*link_age < RC_LINK_TIMEOUT_TICKS) (*link_age)++;
    return *link_age < RC_LINK_TIMEOUT_TICKS;
}

/* CRSF unit threshold → µs (crsf_to_us maps -1..+1 to 988..2012 µs) */
#define T16_ARM_ON_US  ((uint32_t)(1500.0f + RC_ARM_ON_THRESHOLD * 512.0f))

/* ============================================================
 * Radio pre-flight for test 16. Five steps, each printed PASS/FAIL with
 * the likely cause, stopping at the first failure:
 *   1. bytes arriving on RX2      (receiver powered + wired)
 *   2. valid CRSF frames          (protocol / baud / inversion right)
 *   3. frame rate, CRC, LQ        (link healthy)
 *   4. channel values sane
 *   5. ARM switch: ON then OFF on CH(RC_CH_ARM+1), crossing the threshold
 * ============================================================ */
static int t16_radio_checks(imu_type_t imu, crsf_t *c, uint32_t *link_age)
{
    uart_write_string("---- RADIO CHECKS (motors disabled — no signal to the ESCs) ----\r\n");
    uart_write_string("  Radio transmitter ON. Battery in (some boards only power the\r\n"
                      "  receiver from the battery, not from USB).\r\n\r\n");

    /* ── 1 + 2: wait up to 20 s for bytes, then frames ── */
    int t;
    for (t = 0; t < 2000; t++) {
        t16_tick(imu, c, link_age);
        if (c->rc_frames >= 10) break;
        if (t % 200 == 0) {
            uart_write_string("  listening on RX2 (PA3)... bytes ");
            uart_write_uint32(rc_input_bytes());
            uart_write_string("  RC frames ");
            uart_write_uint32(c->rc_frames);
            uart_write_string("\r\n");
        }
    }
    uart_write_string("\r\n  Step 1  bytes from receiver: ");
    uart_write_uint32(rc_input_bytes());
    uart_write_string("\r\n");
    if (rc_input_bytes() == 0) {
        print_fail("Step 1: NOTHING received on RX2 in 20 s");
        uart_write_string("    Likely: receiver not powered (LED off? plug the battery in),\r\n"
                          "    receiver TX wire not on the FC's RX2 pad (PA3) — TX goes to RX,\r\n"
                          "    or receiver not bound / transmitter off (ELRS sends nothing\r\n"
                          "    until it is connected: receiver LED should be SOLID).\r\n");
        return 0;
    }
    print_pass("Step 1: receiver is sending bytes");

    uart_write_string("  Step 2  valid RC frames ");
    uart_write_uint32(c->rc_frames);
    uart_write_string(", other frames ");
    uart_write_uint32(c->other_frames);
    uart_write_string(", CRC errors ");
    uart_write_uint32(c->crc_errors);
    uart_write_string(", junk bytes ");
    uart_write_uint32(c->bad_headers);
    uart_write_string(", overruns ");
    uart_write_uint32(rc_input_overruns());
    uart_write_string("\r\n");
    if (c->rc_frames < 10) {
        print_fail("Step 2: bytes arrive but they are not CRSF RC frames");
        uart_write_string("    Likely: receiver outputs another protocol (SBUS/PPM) or a\r\n"
                          "    different baud — in the ELRS receiver config set serial\r\n"
                          "    protocol = CRSF, 420000 baud. Mostly 'junk bytes' = wrong\r\n"
                          "    baud/protocol; mostly 'CRC errors' = noisy wiring/ground.\r\n");
        return 0;
    }
    print_pass("Step 2: CRSF RC frames decode");

    /* ── 3: link health over 2 s ── */
    uint32_t f0 = c->rc_frames, e0 = c->crc_errors, drops = 0;
    for (t = 0; t < 200; t++)
        if (!t16_tick(imu, c, link_age)) drops++;
    uint32_t rate = (c->rc_frames - f0) / 2U;
    uint32_t errs = c->crc_errors - e0;
    uart_write_string("  Step 3  RC frames/s ");
    uart_write_uint32(rate);
    uart_write_string(", CRC errors in 2 s ");
    uart_write_uint32(errs);
    uart_write_string(", LQ ");
    if (c->other_frames) { uart_write_uint32(c->lq); uart_write_string(" %"); }
    else uart_write_string("(no link-stats frames)");
    uart_write_string(", link dropouts ");
    uart_write_uint32(drops);
    uart_write_string(" ticks\r\n");
    if (rate < 25U || drops > 0U) {
        print_fail("Step 3: link too slow or dropping out");
        uart_write_string("    Need >= 25 frames/s with no gaps > 250 ms. Check the packet\r\n"
                          "    rate in the ELRS Lua script, antenna, and that the radio is\r\n"
                          "    not in failsafe / range-test / low-power mode.\r\n");
        return 0;
    }
    if (errs * 20U > (c->rc_frames - f0)) {
        print_fail("Step 3: more than 5 % CRC errors — noisy wiring");
        uart_write_string("    Shorten/twist the receiver wires, check the ground wire.\r\n");
        return 0;
    }
    if (c->other_frames && c->lq < 50U) {
        print_fail("Step 3: link quality below 50 %");
        uart_write_string("    Move the radio closer / check the receiver antenna.\r\n");
        return 0;
    }
    print_pass("Step 3: link healthy");

    /* ── 4: channel values ── */
    uart_write_string("  Step 4  CH1-8 (us):");
    int bad = 0;
    for (int k = 0; k < 8; k++) {
        uint32_t us = crsf_to_us(c->ch[k]);
        uart_write_char(' ');
        uart_write_uint32(us);
        if (k < 4 && (us < 950U || us > 2050U)) bad = 1;
    }
    uart_write_string("\r\n");
    if (bad) {
        print_fail("Step 4: stick channels outside 950-2050 us");
        uart_write_string("    Radio model / mixer misconfigured.\r\n");
        return 0;
    }
    print_pass("Step 4: channel values in range");

    /* ── 5: ARM switch, 60 s: must go ON then OFF on the ARM channel ── */
    uart_write_string("\r\n  Step 5  ARM switch is expected on CH");
    uart_write_uint32(RC_CH_ARM + 1);
    uart_write_string(" (ON = above ");
    uart_write_uint32(T16_ARM_ON_US);
    uart_write_string(" us).\r\n"
                      "          Flip ARM ON, then OFF. (60 s; don't move other switches)\r\n");
    uint16_t lo[CRSF_NUM_CHANNELS], hi[CRSF_NUM_CHANNELS];
    for (int k = 0; k < CRSF_NUM_CHANNELS; k++) lo[k] = hi[k] = crsf_to_us(c->ch[k]);
    int seen_on = 0, seen_off_after_on = 0;
    for (t = 0; t < 6000 && !seen_off_after_on; t++) {
        if (!t16_tick(imu, c, link_age)) {
            print_fail("Step 5: radio link lost during the switch check");
            return 0;
        }
        for (int k = 0; k < CRSF_NUM_CHANNELS; k++) {
            uint16_t us = crsf_to_us(c->ch[k]);
            if (us < lo[k]) lo[k] = us;
            if (us > hi[k]) hi[k] = us;
        }
        int arm_on = crsf_to_unit(c->ch[RC_CH_ARM]) > RC_ARM_ON_THRESHOLD;
        if (arm_on) seen_on = 1;
        else if (seen_on) seen_off_after_on = 1;

        if (t % 50 == 0) {
            uart_write_string("    CH");
            uart_write_uint32(RC_CH_ARM + 1);
            uart_write_string(" = ");
            uart_write_uint32(crsf_to_us(c->ch[RC_CH_ARM]));
            uart_write_string(arm_on ? " us  ARM ON " : " us  ARM off");
            uart_write_string("   | AUX CH5-12:");
            for (int k = 4; k < 12; k++) { uart_write_char(' '); uart_write_uint32(crsf_to_us(c->ch[k])); }
            uart_write_string(seen_on ? "   -> now flip OFF\r\n" : "   -> flip ON\r\n");
        }
    }
    if (seen_off_after_on) {
        print_pass("Step 5: ARM switch turns ON and OFF");
        return 1;
    }

    /* diagnose */
    print_fail("Step 5: ARM switch not seen going ON then OFF");
    uint32_t arm_lo = lo[RC_CH_ARM], arm_hi = hi[RC_CH_ARM];
    uart_write_string("    CH");
    uart_write_uint32(RC_CH_ARM + 1);
    uart_write_string(" ranged ");
    uart_write_uint32(arm_lo);
    uart_write_string(" - ");
    uart_write_uint32(arm_hi);
    uart_write_string(" us\r\n");
    int moved_other = -1;
    for (int k = 4; k < CRSF_NUM_CHANNELS; k++)
        if (k != RC_CH_ARM && hi[k] - lo[k] > 400 &&
            (moved_other < 0 || hi[k] - lo[k] > hi[moved_other] - lo[moved_other]))
            moved_other = k;
    if (arm_hi - arm_lo < 200U && moved_other >= 0) {
        uart_write_string("    Your switch is on CH");
        uart_write_uint32((uint32_t)moved_other + 1U);
        uart_write_string(" (it moved ");
        uart_write_uint32(lo[moved_other]);
        uart_write_string(" - ");
        uart_write_uint32(hi[moved_other]);
        uart_write_string(" us). Set RC_CH_ARM = ");
        uart_write_uint32((uint32_t)moved_other);
        uart_write_string(" in fc_config.h,\r\n    or move the switch to CH");
        uart_write_uint32(RC_CH_ARM + 1);
        uart_write_string(" (AUX1) in the radio's mixer.\r\n");
    } else if (arm_hi - arm_lo < 200U) {
        uart_write_string("    No AUX channel moved at all. The switch is not assigned to any\r\n"
                          "    channel in the radio model's mixer (or the wrong model is selected).\r\n");
    } else if (!seen_on) {
        uart_write_string("    CH moves but never reaches ON. Set the switch's output to 100 %\r\n"
                          "    (should read ~2000 us), or use a 2-position switch.\r\n");
    } else {
        uart_write_string("    Went ON but never back OFF within 60 s.\r\n");
    }
    return 0;
}

static void test16_thrust_ramp(imu_type_t imu)
{
    print_banner("TEST 16: Thrust Ramp on a Scale (free-standing, ARM = dead-man)");
    uart_write_string("** PROPS ON. Drone free on the scale, slack safety string. STAND BACK. **\r\n");
    uart_write_string("  1. Battery in, radio ON. Radio checks run first (motors disabled).\r\n");
    uart_write_string("  2. Step 5 asks you to flip ARM ON then OFF — that tests the kill.\r\n");
    uart_write_string("  3. At READY: start filming, flip ARM ON -> ramp (+25 us every 4 s).\r\n");
    uart_write_string("  4. When the scale reads ~30 % of its starting value: ARM OFF.\r\n");
    uart_write_string("     ARM OFF / radio lost / tilt / liftoff = motors off instantly.\r\n");
    uart_write_string("  Ceiling this build: ");
    uart_write_uint32(T16_MAX_US);
    uart_write_string(" us\r\n\r\n");

    static crsf_t c;
    crsf_init(&c);
    rc_input_init();
    ahrs_init();
    /* motor_init() is NOT called until every radio check passes: until
     * then the motor pins send no pulses at all, so the ESCs cannot spin. */

    uint32_t link_age = RC_LINK_TIMEOUT_TICKS;
    if (!t16_radio_checks(imu, &c, &link_age)) {
        uart_write_string("\r\n  ==> RADIO CHECK FAILED — test 16 stopped. Motors were never enabled.\r\n"
                          "      Fix the step marked [FAIL] above and rerun.\r\n");
        return;
    }

    /* ── ESC arming, then wait for ARM OFF -> ON to start ── */
    uart_write_string("\r\n  Enabling motor outputs at 1000 us (ESCs beep = armed). 3 s...\r\n");
    motor_init();
    int seen_off = 0, started = 0;
    for (int t = 0; t < T16_WAIT_TICKS; t++) {
        int link_ok = t16_tick(imu, &c, &link_age);
        if (!link_ok) {
            motor_all_stop();
            uart_write_string("  [FAIL] radio link lost while waiting to start — stopped.\r\n");
            return;
        }
        int arm_on = crsf_to_unit(c.ch[RC_CH_ARM]) > RC_ARM_ON_THRESHOLD;
        if (!arm_on) seen_off = 1;
        if (t == 300)
            uart_write_string("  READY. Start filming. Flip ARM ON to start the ramp"
                              " (flip it OFF first if it is already ON).\r\n");
        if (t >= 300 && seen_off && arm_on) { started = 1; break; }
        if (t > 300 && t % 300 == 0) {
            uart_write_string(arm_on ? "  waiting: ARM is ON — flip OFF then ON\r\n"
                                     : "  waiting: ARM OFF — flip ON to start\r\n");
        }
    }
    if (!started) {
        motor_all_stop();
        uart_write_string("\r\n  Not started within 3 min — skipped (motors never spun).\r\n");
        return;
    }
    uart_write_string("  ARM ON -> ramp starting.\r\n");

    /* ── Phase 2: 1 s at 1000 us to take the at-rest reference ── */
    float az_ref = 0.0f;
    for (int i = 0; i < 100; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, NULL, NULL, g, a);
        ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);
        rc_input_poll(&c);
        az_ref += a[2];
    }
    az_ref /= 100.0f;
    float roll_ref = ahrs_get_roll(), pitch_ref = ahrs_get_pitch();
    float rate_f = 0.0f;
    static float az_buf[T16_LIFT_AVG_TICKS];
    float az_sum = az_ref * (float)T16_LIFT_AVG_TICKS;
    int az_i = 0;
    for (int k = 0; k < T16_LIFT_AVG_TICKS; k++) az_buf[k] = az_ref;
    float trip_val = 0.0f;
    link_age = 0;

    uart_write_string("\r\n  PWM(us) | step | az_std(g) | gyr_std(rad/s) | max az shift(g) | max tilt(deg)   <- read the scale now\r\n");
    uart_write_string("  --------+------+-----------+----------------+-----------------+--------------\r\n");
    uart_write_string("  (auto-stop limits: az shift > 0.35 g, tilt > 10 deg, rate > 0.5 rad/s)\r\n");

    const char *stop_reason = NULL;
    uint32_t last_pwm = 1000U;

    /* ── Phase 3: staircase ── */
    for (uint32_t pwm = 1000U; pwm <= T16_MAX_US && !stop_reason; pwm += T16_STEP_US) {
        for (int m = 0; m < 4; m++) motor_set(m, (float)(pwm - 1000U) / 1000.0f);
        last_pwm = pwm;

        uart_write_string("    ");
        uart_write_uint32(pwm);
        uart_write_string("  | ");
        uart_write_uint32((pwm - 1000U) / T16_STEP_US + 1U);
        uart_write_string("   | ");

        float s_az = 0, s_az2 = 0, s_g = 0, s_g2 = 0;
        float max_shift = 0.0f, max_tilt = 0.0f;
        int n = 0;
        for (int i = 0; i < T16_STEP_TICKS; i++) {
            float g[3], a[3];
            wait_tick();
            read_body(imu, NULL, NULL, g, a);
            ahrs_update(g[0], g[1], g[2], a[0], a[1], a[2], 0.01f);

            /* dead-man: link + ARM switch, checked every tick */
            if (rc_input_poll(&c) > 0) link_age = 0;
            else if (link_age < RC_LINK_TIMEOUT_TICKS) link_age++;
            if (link_age >= RC_LINK_TIMEOUT_TICKS)                     stop_reason = "RADIO LINK LOST";
            else if (!(crsf_to_unit(c.ch[RC_CH_ARM]) > RC_ARM_ON_THRESHOLD)) stop_reason = "ARM switch OFF";

            /* tipping / liftoff */
            float dr = ahrs_get_roll() - roll_ref, dp = ahrs_get_pitch() - pitch_ref;
            float rate = FABSF(g[0]) > FABSF(g[1]) ? FABSF(g[0]) : FABSF(g[1]);
            rate_f += 0.1f  * (rate - rate_f);
            az_sum += a[2] - az_buf[az_i];
            az_buf[az_i] = a[2];
            az_i = (az_i + 1) % T16_LIFT_AVG_TICKS;
            float shift = az_sum / (float)T16_LIFT_AVG_TICKS - az_ref;
            float tilt  = FABSF(dr) > FABSF(dp) ? FABSF(dr) : FABSF(dp);
            if (shift > max_shift) max_shift = shift;
            if (tilt  > max_tilt)  max_tilt  = tilt;
            if (!stop_reason) {
                if (tilt > T16_TILT_STOP_DEG) {
                    stop_reason = "TILT > 10 deg (tipping, or AHRS upset by vibration)"; trip_val = tilt;
                } else if (rate_f > T16_RATE_STOP_RADS) {
                    stop_reason = "ROLL/PITCH RATE (tipping)"; trip_val = rate_f;
                } else if (shift > T16_LIFT_STOP_G) {
                    stop_reason = "LIFTOFF (0.5 s avg accel z jumped)"; trip_val = shift;
                }
            }

            if (stop_reason) { motor_all_stop(); break; }

            s_az += a[2]; s_az2 += a[2] * a[2];
            s_g  += g[0] + g[1] + g[2];
            s_g2 += g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            n++;
        }

        if (n > 1) {
            float az_m = s_az / (float)n, g_m = s_g / (float)(3 * n);
            float az_v = s_az2 / (float)n - az_m * az_m;
            float g_v  = s_g2 / (float)(3 * n) - g_m * g_m;
            float az_sd, g_sd;
            if (az_v < 0.0f) az_v = 0.0f;
            if (g_v  < 0.0f) g_v  = 0.0f;
            __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(az_sd) : "t"(az_v));
            __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(g_sd)  : "t"(g_v));
            print_float3(az_sd);
            uart_write_string("     | ");
            print_float3(g_sd);
            uart_write_string("          | ");
            print_float3(max_shift);
            uart_write_string("          | ");
            print_float3(max_tilt);
        }
        if (stop_reason) uart_write_string("   (cut short)");
        uart_write_string("\r\n");
    }

    /* ── Phase 4: finished the ceiling normally → ramp down gently ── */
    if (!stop_reason) {
        for (uint32_t pwm = last_pwm; pwm > 1000U; pwm -= 5U) {
            for (int m = 0; m < 4; m++) motor_set(m, (float)(pwm - 1000U) / 1000.0f);
            wait_tick();
        }
    }
    motor_all_stop();

    uart_write_string("\r\n---- TEST 16 RESULTS ----\r\n  Motors OFF. ");
    if (stop_reason) {
        uart_write_string("Stopped by: ");
        uart_write_string(stop_reason);
        uart_write_string(" at ");
        uart_write_uint32(last_pwm);
        uart_write_string(" us");
        if (trip_val != 0.0f) {
            uart_write_string("  (value ");
            print_float3(trip_val);
            uart_write_string(")");
        }
        uart_write_string("\r\n");
    } else {
        uart_write_string("Reached the ");
        uart_write_uint32(T16_MAX_US);
        uart_write_string(" us ceiling.\r\n  If the scale was still above ~30 % there, hover is ABOVE ");
        uart_write_uint32(T16_MAX_US);
        uart_write_string(" us. Rerun with a higher T16_MAX to find it.\r\n");
    }
    uart_write_string("  thrust(g) = reading@1000us - reading@step;  hover = where reading -> 0.\r\n"
                      "  This weights.h: base ");
    uart_write_uint32((uint32_t)(1000.0f + NN_HOVER_THROTTLE * 1000.0f + 0.5f));
    uart_write_string(" us, firmware can sustain at most ");
    uart_write_uint32((uint32_t)(1000.0f + (NN_HOVER_THROTTLE + ALT_ADJ_MAX) * 1000.0f + 0.5f));
    uart_write_string(" us.\r\n"
                      "  Measured 2026-11-21: hover ~1372 us (W 1110 g) — experiments/bench-2026-11-21.\r\n"
                      "  az_std at the top steps vs 0.22 g trained noise = vibration check.\r\n");
}

/* ============================================================
 * TEST 17: RC channel finder (motors NEVER driven)
 *
 * Prints every channel live and announces each switch/button change, so
 * you can see which physical switch drives which channel and choose ARM.
 * Runs 90 s. Radio ON, receiver powered (battery in if USB doesn't power
 * it). Leave the sticks alone and flip one switch at a time.
 * ============================================================ */
static void test17_rc_channel_finder(imu_type_t imu)
{
    print_banner("TEST 17: RC Channel Finder (motors OFF)");
    uart_write_string("Flip ONE switch / press ONE button at a time and watch for '>>> CHn'.\r\n");
    uart_write_string("Currently configured: ARM = CH");
    uart_write_uint32(RC_CH_ARM + 1);
    uart_write_string(", MODE = CH");
    uart_write_uint32(RC_CH_MODE + 1);
    uart_write_string("  (fc_config.h, RC_CH_ARM = channel - 1)\r\n\r\n");

    static crsf_t c;
    crsf_init(&c);
    rc_input_init();
    uint32_t link_age = RC_LINK_TIMEOUT_TICKS;

    uint16_t lo[CRSF_NUM_CHANNELS], hi[CRSF_NUM_CHANNELS], rep[CRSF_NUM_CHANNELS];
    int have_ref = 0, was_linked = 0;

    for (int t = 0; t < 9000; t++) {                 /* 90 s */
        int link_ok = t16_tick(imu, &c, &link_age);
        if (link_ok && !have_ref) {
            for (int k = 0; k < CRSF_NUM_CHANNELS; k++)
                lo[k] = hi[k] = rep[k] = crsf_to_us(c.ch[k]);
            have_ref = 1;
            uart_write_string(">>> RADIO LINK UP — starting positions recorded\r\n");
        }
        if (!link_ok && was_linked) uart_write_string(">>> RADIO LINK LOST\r\n");
        was_linked = link_ok;

        if (!have_ref) {
            if (t % 200 == 0) {
                uart_write_string("  no RC frames yet... bytes received ");
                uart_write_uint32(rc_input_bytes());
                uart_write_string(", frames ");
                uart_write_uint32(c.rc_frames);
                uart_write_string(rc_input_bytes() ? "  (bytes but no frames: receiver not CRSF/420000?)\r\n"
                                                   : "  (nothing: receiver powered? wired to RX2/PA3? bound?)\r\n");
            }
            continue;
        }

        /* announce changes: switches/buttons jump; 300 us hysteresis */
        for (int k = 0; k < CRSF_NUM_CHANNELS; k++) {
            uint16_t us = crsf_to_us(c.ch[k]);
            if (us < lo[k]) lo[k] = us;
            if (us > hi[k]) hi[k] = us;
            int d = (int)us - (int)rep[k];
            if ((d > 300 || d < -300) && uart_tx_free() > 120U) {
                uart_write_string(">>> CH");
                uart_write_uint32((uint32_t)k + 1U);
                uart_write_string(k < 4 ? " (stick)" : "");
                uart_write_string(" changed ");
                uart_write_uint32(rep[k]);
                uart_write_string(" -> ");
                uart_write_uint32(us);
                uart_write_string(" us");
                if (k == RC_CH_ARM) uart_write_string("   <- this is the channel used as ARM now");
                uart_write_string("\r\n");
                rep[k] = us;
            }
        }

        if (t % 100 == 0 && uart_tx_free() > 300U) {
            uart_write_string(link_ok ? "  LQ " : "  NO LINK  LQ ");
            uart_write_uint32(c.lq);
            uart_write_string(" | sticks CH1-4:");
            for (int k = 0; k < 4; k++) { uart_write_char(' '); uart_write_uint32(crsf_to_us(c.ch[k])); }
            uart_write_string(" | CH5-12:");
            for (int k = 4; k < 12; k++) { uart_write_char(' '); uart_write_uint32(crsf_to_us(c.ch[k])); }
            uart_write_string("\r\n");
        }
    }

    uart_write_string("\r\n---- TEST 17 SUMMARY: range seen on each channel ----\r\n");
    if (!have_ref) {
        print_fail("never received RC frames — see the hints printed above");
        return;
    }
    for (int k = 0; k < CRSF_NUM_CHANNELS; k++) {
        uart_write_string("  CH");
        uart_write_uint32((uint32_t)k + 1U);
        uart_write_string(k < 9 ? " : " : ": ");
        uart_write_uint32(lo[k]);
        uart_write_string(" - ");
        uart_write_uint32(hi[k]);
        uart_write_string(" us");
        if (hi[k] - lo[k] < 50) uart_write_string("   (not moved)");
        if (k == RC_CH_ARM)  uart_write_string("   <- ARM now");
        if (k == RC_CH_MODE) uart_write_string("   <- MODE now");
        uart_write_string("\r\n");
    }
    uart_write_string("\r\n  To make channel N the ARM switch: fc_config.h -> RC_CH_ARM = N-1\r\n"
                      "  (or, better for ELRS, assign your arm switch to CH5/AUX1 in the radio mixer).\r\n"
                      "  ARM = ON above ");
    uart_write_uint32(T16_ARM_ON_US);
    uart_write_string(" us, so pick a 2-position switch that reaches ~2000 us.\r\n");
}

/* ============================================================
 * TEST 18: Level calibration, stored in flash (issue 003)
 *
 *   Two measurements on the same spot, the second rotated 180° about
 *   vertical. The surface slope flips sign between them, the sensor's own
 *   offset (chip bias + board mounting) does not, so
 *       offset = (a0 + a180) / 2,   surface slope = (a0 − a180) / 2.
 *   The surface does not need to be perfectly level. "Level" afterwards =
 *   the plane the landing gear stands on. Saved to flash sector 10;
 *   main.c uses it on every boot (level_cal.h).
 *
 *   Props OFF, battery optional (USB power is fine). Firm flat surface.
 * ============================================================ */
#define T18_SAMPLES        300            /* 3 s per measurement          */
#define T18_ACC_STD_MAX    0.010f         /* g: "still" (props off, no hands) */
#define T18_GYR_STD_MAX    0.020f         /* rad/s                         */
#define T18_ROT_MIN_DEG    170.0f
#define T18_ROT_MAX_DEG    190.0f
#define T18_STILL_TICKS    150            /* 1.5 s still to accept rotation */
#define T18_ROT_TIMEOUT    6000           /* 60 s                          */

/* Mean body-frame accel/gyro over T18_SAMPLES; returns 1 if the board was still. */
static int t18_measure(imu_type_t imu, float a_mean[3], float g_mean[3])
{
    float sa[3] = {0}, sa2[3] = {0}, sg[3] = {0}, sg2[3] = {0};
    for (int i = 0; i < T18_SAMPLES; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, NULL, NULL, g, a);
        for (int k = 0; k < 3; k++) {
            sa[k] += a[k]; sa2[k] += a[k] * a[k];
            sg[k] += g[k]; sg2[k] += g[k] * g[k];
        }
    }
    int still = 1;
    const float n = (float)T18_SAMPLES;
    for (int k = 0; k < 3; k++) {
        a_mean[k] = sa[k] / n;
        g_mean[k] = sg[k] / n;
        float va = sa2[k] / n - a_mean[k] * a_mean[k];
        float vg = sg2[k] / n - g_mean[k] * g_mean[k];
        if (va > T18_ACC_STD_MAX * T18_ACC_STD_MAX) still = 0;
        if (vg > T18_GYR_STD_MAX * T18_GYR_STD_MAX) still = 0;
    }
    float norm2 = a_mean[0] * a_mean[0] + a_mean[1] * a_mean[1] + a_mean[2] * a_mean[2];
    if (norm2 < 0.81f || norm2 > 1.21f) still = 0;        /* |a| outside 0.9–1.1 g */
    return still;
}

static int t18_measure_retry(imu_type_t imu, const char *what, float a[3], float g[3])
{
    for (int attempt = 1; attempt <= 3; attempt++) {
        uart_write_string("  Measuring ");
        uart_write_string(what);
        uart_write_string(" (3 s, don't touch)...\r\n");
        if (t18_measure(imu, a, g)) {
            uart_write_string("    ax "); print_float3(a[0]);
            uart_write_string("  ay "); print_float3(a[1]);
            uart_write_string("  az "); print_float3(a[2]);
            uart_write_string(" g\r\n");
            return 1;
        }
        uart_write_string("    moved / vibrating - keep it still. Retrying...\r\n");
        delay_ms(1000);
    }
    return 0;
}

static void t18_print_angles(const char *label, float ax_g, float ay_g)
{
    uart_write_string(label);
    uart_write_string(" roll ");
    print_float3(level_cal_roll_deg(ay_g));
    uart_write_string(" deg, pitch ");
    print_float3(level_cal_pitch_deg(ax_g));
    uart_write_string(" deg\r\n");
}

static void test18_level_calibration(imu_type_t imu)
{
    print_banner("TEST 18: Level calibration (stored in flash)");

    level_cal_t old;
    int had_old = level_cal_load(&old);
    if (had_old) t18_print_angles("  Currently stored offset:", old.offset[0], old.offset[1]);
    else         uart_write_string("  No calibration stored yet.\r\n");

    uart_write_string(
        "\r\n  PROPS OFF. Stand the drone on its landing gear on a firm, flat\r\n"
        "  surface (a table). It does NOT need to be perfectly level.\r\n"
        "  Hands off. Step 1 of 2 starts in 5 s...\r\n");
    delay_ms(5000);

    float a0[3], a180[3], g0[3], g180[3];
    if (!t18_measure_retry(imu, "position 1", a0, g0)) {
        print_fail("TEST 18: board never still - nothing saved");
        return;
    }

    uart_write_string(
        "\r\n  Step 2 of 2: ROTATE the drone 180 deg IN PLACE (spin it around on\r\n"
        "  the same spot, keep it standing on the surface), then let go.\r\n"
        "  The angle is printed live; it continues once it reads 170-190 deg\r\n"
        "  and the drone is still for 1.5 s.\r\n");

    float yaw = 0.0f;              /* rad, gyro-integrated, bias from position 1 */
    int still_ticks = 0, ok = 0;
    for (int i = 0; i < T18_ROT_TIMEOUT; i++) {
        float g[3], a[3];
        wait_tick();
        read_body(imu, g0, NULL, g, a);
        yaw += g[2] * 0.01f;
        float m = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        still_ticks = (m < 0.03f * 0.03f) ? still_ticks + 1 : 0;
        float deg = level_cal_absf(yaw) * 57.2958f;
        if (i % 50 == 0) {
            uart_write_string("    rotated ");
            print_float3(deg);
            uart_write_string(" deg\r\n");
        }
        if (still_ticks >= T18_STILL_TICKS && deg >= T18_ROT_MIN_DEG && deg <= T18_ROT_MAX_DEG) {
            ok = 1;
            break;
        }
    }
    if (!ok) {
        print_fail("TEST 18: no 180 deg rotation within 60 s - nothing saved");
        return;
    }
    uart_write_string("    rotation accepted: ");
    print_float3(level_cal_absf(yaw) * 57.2958f);
    uart_write_string(" deg\r\n");

    if (!t18_measure_retry(imu, "position 2", a180, g180)) {
        print_fail("TEST 18: board never still - nothing saved");
        return;
    }

    level_cal_t c;
    level_cal_from_two(a0, a180, &c);

    uart_write_string("\r\n  RESULT\r\n");
    t18_print_angles("    Board/frame level offset (now corrected):", c.offset[0], c.offset[1]);
    uart_write_string("    offset ax "); print_float3(c.offset[0]);
    uart_write_string("  ay "); print_float3(c.offset[1]);
    uart_write_string("  az "); print_float3(c.offset[2]);
    uart_write_string(" g\r\n");
    t18_print_angles("    Your surface slope (ignored):", c.surface[0], c.surface[1]);
    uart_write_string("    (the old boot calibration would have flown leaning by\r\n"
                      "     offset + slope; now only the gear plane counts)\r\n");
    if (level_cal_absf(a0[2] - a180[2]) > 0.01f)
        uart_write_string("    note: z differs between positions by > 0.01 g -\r\n"
                          "          surface uneven or drone moved; consider re-running.\r\n");
    if (had_old)
        t18_print_angles("    Change vs stored:", c.offset[0] - old.offset[0],
                         c.offset[1] - old.offset[1]);

    if (!level_cal_plausible(&c)) {
        print_fail("TEST 18: offset > 0.10 g (~6 deg) - not saved. Check the board mounting.");
        return;
    }
    uart_write_string("  Saving to flash (sector 10)...\r\n");
    if (level_cal_save(&c)) {
        print_pass("TEST 18: level calibration saved and read back");
        uart_write_string("  Now flash the flight firmware: make flash\r\n"
                          "  Its boot message should say: LEVEL: stored calibration\r\n");
    } else {
        print_fail("TEST 18: flash write/verify FAILED");
    }
}


int main(void)
{
    /* ── 1. Clock ── */
    int clock_ok = clock_init();

    /* ── 2. LED setup ── */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
    (void)RCC_AHB1ENR;
    GPIOC_MODER &= ~(3U << (LED_PIN * 2));
    GPIOC_MODER |=  (1U << (LED_PIN * 2));
    led_off();

    if (clock_ok != 0) {
        /* Blink rapidly if clock failed */
        while (1) {
            led_on(); delay_ms(50);
            led_off(); delay_ms(50);
        }
    }

    /* ── 3. UART ── */
    uart_init(115200);
    delay_ms(500);

    uart_write_string("\r\n\r\n");
    uart_write_string("########################################\r\n");
    uart_write_string("#   DRONE FC HARDWARE TEST HARNESS     #\r\n");
    uart_write_string("########################################\r\n");
    uart_write_string("Selected tests (TESTS=): " HARNESS_TESTS "\r\n");
    if (test_selected(10) || test_selected(14) || test_selected(16))
        uart_write_string("** MOTOR TESTS SELECTED (10/14/16) — motors WILL spin. **\r\n"
                          "** Props off for 10; secured rig for 14; scale + radio for 16. Unplug battery to skip. **\r\n\r\n");
    else
        uart_write_string("Motor tests 10 and 14 are NOT selected — no motor output.\r\n\r\n");

    /* ── 4. SPI + IMU detect ── */
    delay_ms(100);
    spi1_init();
    delay_ms(10);

    uint8_t whoami = 0;
    imu_type_t imu = imu_detect(&whoami);

    uart_write_string("IMU WHO_AM_I = ");
    uart_write_hex8(whoami);
    uart_write_string("\r\n");

    switch (imu) {
        case IMU_ICM42688P:
            uart_write_string("Detected: ICM-42688-P\r\n");
            accel_raw_to_g = 1.0f / 2048.0f;
            break;
        case IMU_MPU6000:
            uart_write_string("Detected: MPU-6000\r\n");
            accel_raw_to_g = 1.0f / 4096.0f;
            break;
        case IMU_BMI270:
            uart_write_string("Detected: BMI-270\r\n");
            accel_raw_to_g = 1.0f / 4096.0f;
            break;
        default:
            uart_write_string("** NO IMU DETECTED **\r\n");
            uart_write_string("SPI bus is dead. Check wiring/CS pin.\r\n");
            while (1) {
                led_on(); delay_ms(100);
                led_off(); delay_ms(100);
            }
    }

    /* ── 5. IMU init (same sequence as main.c) ── */
    imu_init(imu);
    delay_ms(50);

    /* ── 6. AHRS init + timers ──
     * SysTick (100 Hz tick) and the DWT cycle counter are started here so
     * any subset of tests can run on its own (tests used to rely on test 2
     * having started SysTick). */
    ahrs_init();
    systick_init_100hz();
    dwt_init();

    /* ── 7. I2C + Barometer ── */
    i2c1_init();
    uint8_t baro_id = 0;
    baro_type_t baro = baro_init(&baro_id);
    int baro_ok = (baro != BARO_NONE);

    if (baro_ok) {
        uart_write_string("Barometer detected (id=");
        uart_write_hex8(baro_id);
        uart_write_string(")\r\n");
    } else {
        uart_write_string("No barometer detected (tests 5 obs[8,9] will be 0)\r\n");
    }

    /* ============================================================
     * RUN TESTS
     * ============================================================ */

    /* Each test runs only if selected (see HARNESS_TESTS / make TESTS=).
     * Motor tests 10 and 14 are excluded unless explicitly listed. */
    if (test_selected(1))  test1_spi_readback(imu);              /* immediate */
    if (test_selected(2))  test2_static_imu(imu);                /* 5 s, flat */
    if (test_selected(3))  test3_dynamic_axis(imu);              /* interactive */
    if (test_selected(4))  test4_ahrs_convergence(imu);          /* interactive */
    if (test_selected(5))  test5_obs_vector(imu, baro_ok);       /* flat */
    if (test_selected(6))  test6_nn_inference();                 /* immediate */
    if (test_selected(7))  test7_hil_closed_loop(imu, baro_ok);  /* 10 s, flat */
    if (test_selected(8))  test8_tilt_response(imu);             /* interactive */
    if (test_selected(9))  test9_iir_lag(imu);                   /* interactive */
    if (test_selected(10)) test10_motor_wire_verify(imu);        /* MOTORS SPIN, props off */
    if (test_selected(11)) test11_iir_threshold_stress();        /* immediate */
    if (test_selected(12)) test12_loop_jitter(imu, baro_ok);     /* 5 s, flat */
    if (test_selected(13)) test13_baro_throttle_noise(imu, baro_ok); /* 8 s, flat */
    if (test_selected(14)) test14_throttle_staircase(imu);       /* MOTORS SPIN, props on, secured */
    if (test_selected(15)) test15_rc_link(imu);                  /* RC link, radio ON, motors OFF */
    if (test_selected(16)) test16_thrust_ramp(imu);              /* MOTORS SPIN, props on, ARM = dead-man */
    if (test_selected(17)) test17_rc_channel_finder(imu);        /* RC channel finder, motors OFF */
    if (test_selected(18)) test18_level_calibration(imu);        /* writes flash sector 10, props off */

    /* ── Done ── */
    print_banner("ALL SELECTED TESTS COMPLETE");
    uart_write_string("Review output above for PASS/FAIL.\r\n");
    uart_write_string("LED solid = done.\r\n");

    led_on();
    while (1) { /* halt */ }

    return 0;
}
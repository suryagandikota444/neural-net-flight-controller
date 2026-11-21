/*
 * test_harness.c — Systematic hardware validation for drone FC
 *
 * ── BUILD ──────────────────────────────────────────────────────
 *
 *   Compile this INSTEAD of main.c:
 *     Replace main.c with test_harness.c in your Makefile, or:
 *     arm-none-eabi-gcc ... test_harness.c startup.c clock.c spi.c
 *                           imu.c uart.c ahrs.c nn.c i2c.c baro.c
 *                           memset.c -o test.elf
 *
 *   Motors are never initialized = completely safe on bench with props on.
 *
 * ── USAGE ──────────────────────────────────────────────────────
 *
 *   1. Flash test.elf
 *   2. Open serial: screen /dev/cu.SLAB_USBtoUART 115200
 *   3. The three tests run automatically in sequence:
 *
 *      TEST 1: SPI register readback  (immediate, ~1 second)
 *      TEST 2: Static IMU readings    (5 seconds, board FLAT and STILL)
 *      TEST 3: Dynamic axis alignment (30 seconds, follow prompts to tilt)
 *      TEST 4: AHRS convergence       (20 seconds, follow prompts to tilt/return)
 *      TEST 5: Observation vector     (5 seconds, board FLAT and STILL)
 *      TEST 6: NN inference           (immediate, pure computation)
 *
 * ── EXPECTED OUTPUT ────────────────────────────────────────────
 *
 *   See detailed PASS/FAIL criteria printed inline and in the
 *   comments below each test section.
 */

#include <stdint.h>
#include "clock.h"
#include "spi.h"
#include "imu.h"
#include "uart.h"
#include "ahrs.h"
#include "nn.h"        /* NN_HOVER_THROTTLE, NN_ACTION_SCALE, nn_forward() */
#include "i2c.h"
#include "baro.h"
#include "motor.h"     /* motor_init, motor_set, motor_all_stop for test 10 */


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
 * Axis alignment macros — MUST match main.c exactly
 * If you change them in main.c, change them here too.
 * ============================================================ */
#define ALIGN_AX(raw)  ( (raw))
#define ALIGN_AY(raw)  ( -(raw))
#define ALIGN_AZ(raw)  ( (raw))
#define ALIGN_GX(raw)  ( -(raw))
#define ALIGN_GY(raw)  ( (raw))
#define ALIGN_GZ(raw)  ( (raw))


/* Gyro scale: same as main.c */
#define GYRO_RAW_TO_RADS   (1.0f / 939.65f)
#define MAX_GYRO_RADS      34.907f

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
    uart_write_string(" (0x");
    uart_write_hex8(addr);
    uart_write_string(") = 0x");
    uart_write_hex8(got);
    uart_write_string("  expected 0x");
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
        print_regcheck("GYRO_ACCEL_CF", ICM_REG_GYRO_ACCEL_CONFIG0, v, 0x06);
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
 *
 *  ████████ ███████ ███████ ████████ ██████
 *     ██    ██      ██         ██         ██
 *     ██    █████   ███████    ██     █████
 *     ██    ██           ██    ██         ██
 *     ██    ███████ ███████    ██    ██████
 *
 *  DYNAMIC AXIS ALIGNMENT
 *
 *  Purpose: Verify that physical tilts produce the correct
 *           sign in both raw sensor data AND AHRS output.
 *           This is the #1 sim-to-real failure mode.
 *
 *  Convention we need to match (MuJoCo NWU):
 *    X = forward (nose), Y = left, Z = up
 *
 *  Procedure: The test prompts you to do 3 tilts, 8 seconds each.
 *    You hold the tilt for the middle 4 seconds while it samples.
 *
 *  Expected results:
 *
 *    TILT 1: Nose down (pitch forward)
 *      → ax should go POSITIVE  (gravity projects onto +X)
 *      → AHRS pitch should go NEGATIVE (nose below horizon)
 *      → gy (pitch rate) should briefly go NEGATIVE when tilting
 *
 *    TILT 2: Right side down (roll right)
 *      → ay should go POSITIVE  (gravity projects onto +Y left axis
 *                                 when right side drops? NO — this
 *                                 depends on the convention.)
 *        ACTUALLY: In NWU, Y = left. When you tilt RIGHT side down,
 *        gravity projects onto -Y. So ay should go NEGATIVE.
 *        BUT: the AHRS roll should go POSITIVE for a right tilt
 *        (right wing down = positive roll in aerospace convention).
 *
 *        *** This is exactly the kind of thing you need to verify
 *            empirically rather than reason about. That's the
 *            whole point of this test. ***
 *
 *    TILT 3: Rotate CW (viewed from above)
 *      → gz should go NEGATIVE (right-hand rule: CW about +Z = neg)
 *      → AHRS yaw should decrease
 *
 *  The test prints live values so you can see what actually happens.
 *  Compare against the expected signs and note any discrepancies.
 *
 * ============================================================ */

/* AHRS helpers: get degrees */
extern float ahrs_get_roll(void);
extern float ahrs_get_pitch(void);
extern float ahrs_get_yaw(void);

static void run_tilt_phase(imu_type_t imu, const char *label,
                           const char *instruction, int samples)
{
    uart_write_string("\r\n  >> ");
    uart_write_string(instruction);
    uart_write_string(" <<\r\n");
    uart_write_string("  (Hold for ~");
    uart_write_uint32(samples / 100);
    uart_write_string(" seconds)\r\n\r\n");

    /* Let user get into position */
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
    uart_write_string("SAMPLING\r\n\r\n");

    float sum_ax = 0, sum_ay = 0, sum_az = 0;
    float sum_gx = 0, sum_gy = 0, sum_gz = 0;
    float sum_roll = 0, sum_pitch = 0, sum_yaw = 0;

    for (int i = 0; i < samples; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}

        imu_raw_t sample;
        imu_read_raw(imu, &sample);

        float gx = (float)ALIGN_GX(sample.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(sample.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(sample.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(sample.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(sample.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(sample.az) * accel_raw_to_g;

        /* Feed AHRS so roll/pitch/yaw update */
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        float roll  = ahrs_get_roll();
        float pitch = ahrs_get_pitch();
        float yaw   = ahrs_get_yaw();

        sum_ax += ax;  sum_ay += ay;  sum_az += az;
        sum_gx += gx;  sum_gy += gy;  sum_gz += gz;
        sum_roll += roll;  sum_pitch += pitch;  sum_yaw += yaw;

        /* Print at 5 Hz (every 20 ticks) */
        if (i % 20 == 0) {
            uart_write_string("  ax=");
            print_float3(ax);
            uart_write_string(" ay=");
            print_float3(ay);
            uart_write_string(" az=");
            print_float3(az);
            uart_write_string(" | R=");
            print_float3(roll);
            uart_write_string(" P=");
            print_float3(pitch);
            uart_write_string(" Y=");
            print_float3(yaw);
            uart_write_string("\r\n");
        }
    }

    float n = (float)samples;
    uart_write_string("\r\n  ---- ");
    uart_write_string(label);
    uart_write_string(" AVERAGES ----\r\n");
    uart_write_string("    ax="); print_float3(sum_ax / n);
    uart_write_string("  ay="); print_float3(sum_ay / n);
    uart_write_string("  az="); print_float3(sum_az / n);
    uart_write_string("\r\n");
    uart_write_string("    gx="); print_float3(sum_gx / n);
    uart_write_string("  gy="); print_float3(sum_gy / n);
    uart_write_string("  gz="); print_float3(sum_gz / n);
    uart_write_string("\r\n");
    uart_write_string("    AHRS roll="); print_float3(sum_roll / n);
    uart_write_string("  pitch="); print_float3(sum_pitch / n);
    uart_write_string("  yaw="); print_float3(sum_yaw / n);
    uart_write_string("\r\n");
}


static void test3_dynamic_axis(imu_type_t imu)
{
    print_banner("TEST 3: Dynamic Axis Alignment");
    uart_write_string("This test verifies that physical tilts produce\r\n");
    uart_write_string("the correct signs in both raw accel and AHRS.\r\n\r\n");
    uart_write_string("You will tilt the board 3 ways, ~30 deg each.\r\n");
    uart_write_string("Follow the prompts.\r\n");

    /* First, let AHRS settle for 2 seconds flat on table */
    uart_write_string("\r\nStep 0: Place board FLAT. AHRS settling (2 sec)...\r\n");
    ahrs_init();
    for (int i = 0; i < 200; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t sample;
        imu_read_raw(imu, &sample);
        float gx = (float)ALIGN_GX(sample.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(sample.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(sample.gz) * GYRO_RAW_TO_RADS;
        float ax = (float)ALIGN_AX(sample.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(sample.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(sample.az) * accel_raw_to_g;
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
    }

    uart_write_string("  AHRS settled. Roll=");
    print_float3(ahrs_get_roll());
    uart_write_string(" Pitch=");
    print_float3(ahrs_get_pitch());
    uart_write_string("\r\n  (Should be near 0,0 if board is flat)\r\n");

    /* ---- Tilt 1: Nose down ---- */
    uart_write_string("\r\n");
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 1: NOSE DOWN (~30 degrees)\r\n");
    uart_write_string("  Lift the REAR of the board so the nose dips.\r\n");
    uart_write_string("  Expected: ax > 0, AHRS pitch < 0\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "NOSE-DOWN", "Tilt NOSE DOWN now and hold steady", 400);

    /* ---- Tilt 2: Right side down ---- */
    uart_write_string("\r\n");
    uart_write_string("  Return to FLAT for 3 seconds...\r\n");
    delay_ms(3000);
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 2: RIGHT SIDE DOWN (~30 degrees)\r\n");
    uart_write_string("  Lift the LEFT side so the RIGHT side dips.\r\n");
    uart_write_string("  Expected: AHRS roll > 0 (right-wing-down = positive)\r\n");
    uart_write_string("  Watch the sign of ay — document what you see.\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "RIGHT-DOWN", "Tilt RIGHT SIDE DOWN now and hold steady", 400);

    /* ---- Tilt 3: Yaw CW ---- */
    uart_write_string("\r\n");
    uart_write_string("  Return to FLAT for 3 seconds...\r\n");
    delay_ms(3000);
    uart_write_string("----------------------------------------\r\n");
    uart_write_string("TILT 3: ROTATE CW (viewed from above)\r\n");
    uart_write_string("  Keep board flat, rotate clockwise slowly.\r\n");
    uart_write_string("  Expected: gz < 0 (CW about +Z = negative)\r\n");
    uart_write_string("  AHRS yaw should decrease.\r\n");
    uart_write_string("----------------------------------------\r\n");
    run_tilt_phase(imu, "YAW-CW", "Rotate CLOCKWISE now (keep flat)", 400);

    /* ---- Summary ---- */
    print_banner("TEST 3 COMPLETE — Review above");
    uart_write_string("Compare the sign of each measurement against expected.\r\n");
    uart_write_string("Any sign flip = axis alignment bug in ALIGN_xx macros.\r\n\r\n");
    uart_write_string("Quick reference (MuJoCo NWU convention):\r\n");
    uart_write_string("  Nose down:     ax>0, pitch<0\r\n");
    uart_write_string("  Right down:    roll>0\r\n");
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
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
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
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
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
 *    obs[8]  =  0.000   altitude error (at ground ref)
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

/* Obs vector scaling constants — MUST match main.c exactly */
#define OBS_MAX_GYRO_RADS   34.907f
#define OBS_ANGLE_SCALE     (1.0f / 45.0f)
#define OBS_ALT_SCALE       2.0f
#define OBS_VVEL_SCALE      5.0f
#define OBS_ALT_OBS_CLAMP   0.05f
#define OBS_TARGET_ALT      0.3f

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

    /* Expected values */
    static const float expected[12] = {
        0.0f, 0.0f, 0.0f,       /* gyro XYZ */
        0.0f, 0.0f, 0.5f,       /* accel XYZ / 2g */
        0.0f, 0.0f,             /* roll, pitch error */
        0.0f, 0.0f,             /* alt error, vvel (baro at ground ref or absent) */
        0.0f, 0.0f              /* hvel XY */
    };
    /* Tolerances per element */
    static const float tol[12] = {
        0.01f, 0.01f, 0.01f,    /* gyro: tight, should be near zero */
        0.05f, 0.05f, 0.08f,    /* accel: moderate (noise) */
        0.04f, 0.04f,           /* AHRS angles: should be near 0 if flat */
        0.05f, 0.02f,           /* baro: near 0 at ground ref */
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
 *  Purpose: Feed a KNOWN obs vector to nn_forward() on the STM32
 *           and compare the output against the Python reference.
 *           This catches: wrong weights.h flashed, stale build,
 *           byte ordering issues, float precision problems.
 *
 *  Method: Two test vectors with pre-computed expected outputs.
 *
 *  Test vector A: "flat on table"
 *    obs = [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0]
 *    Expected nn_out (computed from weights.h in Python):
 *      out[0] = +0.087682053
 *      out[1] = -0.052892834
 *      out[2] = +0.032626629
 *      out[3] = +0.263809144
 *
 *  Test vector B: "10 deg roll"
 *    obs = [0, 0, 0, 0, 0.086824, 0.492404, 0.222222, 0, 0, 0, 0, 0]
 *    Expected nn_out:
 *      out[0] = +0.024362043
 *      out[1] = +0.368108302
 *      out[2] = +0.501782894
 *      out[3] = +0.229994148
 *
 *  Tolerance: ±0.001 (should match to ~5-6 decimal places since
 *  both use IEEE 754 float32; any mismatch > 0.001 means different
 *  weights or a computation bug).
 *
 * ============================================================ */
static void test6_nn_inference(void)
{
    print_banner("TEST 6: NN Inference Validation");
    uart_write_string("Feeds known obs vectors to nn_forward() and compares\r\n");
    uart_write_string("output against Python reference values.\r\n\r\n");

    int total_pass = 0;
    int total_checks = 0;

    #define NN_TOL 0.001f

    /* ---- Test Vector A: flat on table ---- */
    {
        uart_write_string("Vector A: flat on table\r\n");
        uart_write_string("  obs = [0,0,0, 0,0,+0.5, 0,0,0,0,0,0]\r\n\r\n");

        float obs_a[12] = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f,
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f
        };

        /* Expected output computed from weights.h by Python */
        static const float expected_a[4] = {
            8.768205345e-02f,
           -5.289283395e-02f,
            3.262662888e-02f,
            2.638091445e-01f
        };
        static const char *motor_names[4] = {
            "FL(CCW)", "FR(CW) ", "RR(CCW)", "RL(CW) "
        };

        float out_a[4];
        nn_forward(obs_a, out_a);

        for (int k = 0; k < 4; k++) {
            float err = out_a[k] - expected_a[k];
            if (err < 0) err = -err;

            uart_write_string("  out[");
            uart_write_uint32(k);
            uart_write_string("] ");
            uart_write_string(motor_names[k]);
            uart_write_string(" = ");
            print_float3(out_a[k]);
            uart_write_string("  expect ");
            print_float3(expected_a[k]);
            uart_write_string("  err=");
            print_float3(err);

            total_checks++;
            if (err <= NN_TOL) {
                uart_write_string("  OK\r\n");
                total_pass++;
            } else {
                uart_write_string("  ** FAIL **\r\n");
            }
        }

        /* Also print motor commands for context */
        uart_write_string("\r\n  Motor commands (hover_thr=");
        print_float3(NN_HOVER_THROTTLE);
        uart_write_string(", act_scale=");
        print_float3(NN_ACTION_SCALE);
        uart_write_string("):\r\n");
        for (int k = 0; k < 4; k++) {
            float cmd = NN_HOVER_THROTTLE + out_a[k] * NN_ACTION_SCALE;
            if (cmd < 0.0f) cmd = 0.0f;
            if (cmd > 1.0f) cmd = 1.0f;
            uint32_t pwm_us = 1000 + (uint32_t)(cmd * 1000.0f);
            uart_write_string("    ");
            uart_write_string(motor_names[k]);
            uart_write_string(" cmd=");
            print_float3(cmd);
            uart_write_string("  pwm=");
            uart_write_uint32(pwm_us);
            uart_write_string(" us\r\n");
        }
    }

    /* ---- Test Vector B: 10 deg roll ---- */
    {
        uart_write_string("\r\nVector B: 10 deg roll\r\n");
        uart_write_string("  obs = [0,0,0, 0,+0.087,+0.492, +0.222,0, 0,0,0,0]\r\n\r\n");

        float obs_b[12] = {
            0.0f, 0.0f, 0.0f,
            0.0f, 8.682408929e-02f, 4.924038768e-01f,
            2.222222239e-01f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f
        };

        static const float expected_b[4] = {
            2.436204255e-02f,
            3.681083024e-01f,
            5.017828941e-01f,
            2.299941480e-01f
        };
        static const char *motor_names[4] = {
            "FL(CCW)", "FR(CW) ", "RR(CCW)", "RL(CW) "
        };

        float out_b[4];
        nn_forward(obs_b, out_b);

        for (int k = 0; k < 4; k++) {
            float err = out_b[k] - expected_b[k];
            if (err < 0) err = -err;

            uart_write_string("  out[");
            uart_write_uint32(k);
            uart_write_string("] ");
            uart_write_string(motor_names[k]);
            uart_write_string(" = ");
            print_float3(out_b[k]);
            uart_write_string("  expect ");
            print_float3(expected_b[k]);
            uart_write_string("  err=");
            print_float3(err);

            total_checks++;
            if (err <= NN_TOL) {
                uart_write_string("  OK\r\n");
                total_pass++;
            } else {
                uart_write_string("  ** FAIL **\r\n");
            }
        }
    }

    /* ---- Summary ---- */
    uart_write_string("\r\n  Result: ");
    uart_write_uint32(total_pass);
    uart_write_string("/");
    uart_write_uint32(total_checks);
    uart_write_string(" outputs match Python reference.\r\n");

    if (total_pass == total_checks)
        print_pass("NN weights match — inference is correct");
    else {
        print_fail("NN output mismatch — wrong weights.h or computation error");
        uart_write_string("  Likely causes:\r\n");
        uart_write_string("    1. Stale weights.h (rerun export_weights.py)\r\n");
        uart_write_string("    2. Didn't rebuild after updating weights.h\r\n");
        uart_write_string("    3. Float precision issue (unlikely with same IEEE 754)\r\n");
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
 *    3. HOVER POINT: The average PWM should be near 1323 µs
 *       (= HOVER_THROTTLE * 1000 + 1000). If it's much higher,
 *       the drone will climb; much lower, it won't take off.
 *
 *    4. STABILITY: Commands should not oscillate. If you see
 *       motors alternating between high and low every few ticks,
 *       the NN is fighting itself — likely a sign error.
 *
 *  Auto-validation:
 *    - PWM spread < 80 µs between motors  (PASS)
 *    - Average PWM within 1250-1400 µs    (PASS)
 *    - Steady-state variance < 20 µs      (PASS = stable)
 *
 * ============================================================ */

/* IIR smoothing constants — MUST match main.c */
#define ACTION_ALPHA_SLOW   0.08f
#define ACTION_ALPHA_FAST   0.60f
#define ACTION_SNAP_THRESH  0.15f

/* Altitude P+D constants — MUST match main.c */
#define HIL_ALT_KP          0.15f
#define HIL_ALT_KD          0.08f
#define HIL_ALT_ADJ_MAX     0.10f
#define HIL_ALT_OBS_CLAMP   0.05f
#define HIL_TARGET_ALT      0.3f
#define HIL_ALT_SCALE        2.0f
#define HIL_VVEL_SCALE       5.0f

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

        /* ── F: Adaptive IIR smoothing (same as main.c) ── */
        for (int k = 0; k < 4; k++) {
            float diff  = nn_out[k] - nn_smoothed[k];
            float adiff = diff < 0.0f ? -diff : diff;
            float alpha = adiff >= ACTION_SNAP_THRESH
                            ? ACTION_ALPHA_FAST
                            : ACTION_ALPHA_SLOW;
            nn_smoothed[k] += alpha * diff;
        }
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

    /* Check 2: Average PWM in sane hover range */
    if (overall_avg >= 1250.0f && overall_avg <= 1400.0f)
        print_pass("Average PWM in hover range (1250-1400 us)");
    else if (overall_avg < 1250.0f) {
        print_fail("Average PWM too low — drone won't generate enough thrust");
        uart_write_string("         Check NN_HOVER_THROTTLE or drone weight.\r\n");
    } else {
        print_fail("Average PWM too high — drone will climb uncontrollably on arm");
        uart_write_string("         NN is commanding excess thrust at rest.\r\n");
    }

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

    /* Check 4: Diagnostic — are the CW vs CCW motors balanced? */
    float ccw_avg = (ss_avg[0] + ss_avg[2]) / 2.0f;   /* FL + RR */
    float cw_avg  = (ss_avg[1] + ss_avg[3]) / 2.0f;   /* FR + RL */
    float yaw_bias = cw_avg - ccw_avg;  /* positive = CW motors stronger = yaw CCW */
    uart_write_string("\r\n  Yaw balance: CW_avg - CCW_avg = ");
    print_float3(yaw_bias);
    uart_write_string(" us\r\n");
    if (FABSF(yaw_bias) < 30.0f)
        uart_write_string("    (< 30 us — yaw should be balanced)\r\n");
    else {
        uart_write_string("    (> 30 us — NN is commanding a yaw trim. ");
        if (yaw_bias > 0)
            uart_write_string("CW motors hotter → yaw CCW)\r\n");
        else
            uart_write_string("CCW motors hotter → yaw CW)\r\n");
    }

    /* Diagonal balance (roll/pitch) */
    float diag1 = (ss_avg[0] + ss_avg[2]) / 2.0f;  /* FL+RR diagonal */
    float diag2 = (ss_avg[1] + ss_avg[3]) / 2.0f;  /* FR+RL diagonal */
    float diag_diff = diag1 - diag2;
    uart_write_string("  Diagonal balance: (FL+RR)/2 - (FR+RL)/2 = ");
    print_float3(diag_diff);
    uart_write_string(" us\r\n");
    if (FABSF(diag_diff) < 30.0f)
        uart_write_string("    (< 30 us — roll/pitch balanced)\r\n");
    else
        uart_write_string("    (> 30 us — NN is commanding a roll/pitch trim)\r\n");
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
 *  Purpose: You tilt the board, and the test shows which motors
 *           the NN would spin harder. This catches the "errors
 *           compound" bug — if the NN sends thrust to the WRONG
 *           side when tilted, the drone will flip instead of correct.
 *
 *  Procedure: 3 tilt tests, prompted interactively.
 *    For each: tilt and hold ~20 deg, test prints which side
 *    gets more thrust, and whether that's correct.
 *
 *  NO MOTORS ARE DRIVEN. This is sensor + NN computation only.
 *  But we DO init motors so we can optionally pulse the "winner"
 *  motor briefly so you can HEAR which one the NN picked.
 *
 *  The key diagnostic: if you tilt RIGHT and the LEFT motors
 *  get more thrust, the NN is fighting the tilt (CORRECT).
 *  If the RIGHT motors get more thrust, something is inverted.
 *
 * ============================================================ */

static void test8_tilt_response(imu_type_t imu)
{
    print_banner("TEST 8: Live Tilt → Motor Response");
    uart_write_string("Tilt the board and see which motors the NN would\r\n");
    uart_write_string("command harder. Catches inverted control loops.\r\n\r\n");
    uart_write_string("NO MOTORS WILL SPIN. Safe for bench.\r\n\r\n");

    /* Bias calibrate fresh */
    ahrs_init();
    systick_init_100hz();

    float cal_sum_gx = 0, cal_sum_gy = 0, cal_sum_gz = 0;
    float cal_sum_ax = 0, cal_sum_ay = 0, cal_sum_az = 0;
    int cal_count = 0;

    uart_write_string("  Calibrating (2 sec, keep board FLAT)...\r\n");
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
        if (i >= 50) {
            cal_sum_gx += gx; cal_sum_gy += gy; cal_sum_gz += gz;
            cal_sum_ax += ax; cal_sum_ay += ay; cal_sum_az += az;
            cal_count++;
        }
    }
    float gb[3] = {0}, ab[3] = {0};
    if (cal_count > 0) {
        float inv = 1.0f / (float)cal_count;
        gb[0] = cal_sum_gx * inv; gb[1] = cal_sum_gy * inv; gb[2] = cal_sum_gz * inv;
        ab[0] = cal_sum_ax * inv; ab[1] = cal_sum_ay * inv; ab[2] = cal_sum_az * inv - 1.0f;
    }

    /* Re-settle with bias correction */
    ahrs_init();
    for (int i = 0; i < 100; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
    }
    uart_write_string("  Calibrated. R=");
    print_float3(ahrs_get_roll());
    uart_write_string(" P=");
    print_float3(ahrs_get_pitch());
    uart_write_string("\r\n");

    /* Helper: read sensors, run NN, return motor PWMs */
    /* We'll do this inline in each test phase */

    static const char *mnames[4] = {"FL", "FR", "RR", "RL"};

    /* ============================================================
     * TILT A: Roll RIGHT (right side down)
     *
     * Expected: NN increases RIGHT side motors (FR, RR) to push
     * the right side back up. If LEFT side increases instead,
     * the roll correction is INVERTED.
     * ============================================================ */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT A: Roll RIGHT (right side down, ~20 deg)\r\n");
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
    uart_write_string("TILT AND HOLD\r\n\r\n");

    float sum_pwm_a[4] = {0};
    int count_a = 0;

    for (int i = 0; i < 300; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        float roll  = ahrs_get_roll();
        float pitch = ahrs_get_pitch();

        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
        obs[6] = clampf(roll  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[8] = 0.0f; obs[9] = 0.0f; obs[10] = 0.0f; obs[11] = 0.0f;

        float nn_out[4];
        nn_forward(obs, nn_out);
        float pwm[4];
        for (int k = 0; k < 4; k++) {
            float cmd = clampf(NN_HOVER_THROTTLE + clampf(nn_out[k], -1.0f, 1.0f) * NN_ACTION_SCALE, 0.0f, 1.0f);
            pwm[k] = 1000.0f + cmd * 1000.0f;
        }

        /* Only count samples where roll > 10 deg (actually tilted) */
        if (roll > 10.0f) {
            for (int k = 0; k < 4; k++) sum_pwm_a[k] += pwm[k];
            count_a++;
        }

        if (i % 30 == 0) {
            uart_write_string("  R=");
            print_float3(roll);
            uart_write_string(" P=");
            print_float3(pitch);
            uart_write_string("  FL:");
            uart_write_uint32((uint32_t)(pwm[0]+0.5f));
            uart_write_string(" FR:");
            uart_write_uint32((uint32_t)(pwm[1]+0.5f));
            uart_write_string(" RR:");
            uart_write_uint32((uint32_t)(pwm[2]+0.5f));
            uart_write_string(" RL:");
            uart_write_uint32((uint32_t)(pwm[3]+0.5f));
            uart_write_string("\r\n");
        }
    }

    if (count_a > 10) {
        float avg[4];
        for (int k = 0; k < 4; k++) avg[k] = sum_pwm_a[k] / (float)count_a;
        float right_avg = (avg[1] + avg[2]) / 2.0f;  /* FR + RR */
        float left_avg  = (avg[0] + avg[3]) / 2.0f;  /* FL + RL */

        uart_write_string("\r\n  Right tilt avg (");
        uart_write_uint32(count_a);
        uart_write_string(" samples with R>10):\r\n");
        for (int k = 0; k < 4; k++) {
            uart_write_string("    ");
            uart_write_string(mnames[k]);
            uart_write_string(": ");
            uart_write_uint32((uint32_t)(avg[k]+0.5f));
            uart_write_string(" us\r\n");
        }
        uart_write_string("  Right side (FR+RR) avg: ");
        uart_write_uint32((uint32_t)(right_avg+0.5f));
        uart_write_string("   Left side (FL+RL) avg: ");
        uart_write_uint32((uint32_t)(left_avg+0.5f));
        uart_write_string("\r\n");

        if (right_avg > left_avg + 20.0f)
            print_pass("ROLL RIGHT: Right side gets more thrust → corrects tilt");
        else if (left_avg > right_avg + 20.0f)
            print_fail("ROLL RIGHT: LEFT side gets more thrust → INVERTED! Will compound error!");
        else
            uart_write_string("  [WARN] No clear asymmetry. Did you tilt far enough?\r\n");
    } else {
        uart_write_string("  [WARN] Not enough tilted samples. Roll never exceeded 10 deg.\r\n");
    }

    /* ============================================================
     * TILT B: Pitch NOSE DOWN
     *
     * Expected: NN increases FRONT motors (FL, FR) to push nose up.
     * Actually: more front thrust -> nose pitches DOWN (negative
     * pitch torque). Wait no — for correction, we need to push
     * nose UP = increase REAR motors.
     *
     * NO. Tracing through carefully:
     * - Nose down = negative pitch
     * - Need positive pitch torque to correct
     * - Rear motors at -X create positive pitch torque
     * - So NN should increase REAR motors (RR, RL)
     *
     * If front increases instead, pitch is inverted.
     * ============================================================ */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT B: Pitch NOSE DOWN (~20 deg)\r\n");
    uart_write_string("  Lift the REAR so the nose dips.\r\n");
    uart_write_string("  Return to FLAT first, then...\r\n");
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
    uart_write_string("TILT NOSE DOWN AND HOLD\r\n\r\n");

    float sum_pwm_b[4] = {0};
    int count_b = 0;

    for (int i = 0; i < 300; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        float roll  = ahrs_get_roll();
        float pitch = ahrs_get_pitch();

        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
        obs[6] = clampf(roll  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[8] = 0.0f; obs[9] = 0.0f; obs[10] = 0.0f; obs[11] = 0.0f;

        float nn_out[4];
        nn_forward(obs, nn_out);
        float pwm[4];
        for (int k = 0; k < 4; k++) {
            float cmd = clampf(NN_HOVER_THROTTLE + clampf(nn_out[k], -1.0f, 1.0f) * NN_ACTION_SCALE, 0.0f, 1.0f);
            pwm[k] = 1000.0f + cmd * 1000.0f;
        }

        /* Count when pitch < -10 deg (nose actually down) */
        if (pitch < -10.0f) {
            for (int k = 0; k < 4; k++) sum_pwm_b[k] += pwm[k];
            count_b++;
        }

        if (i % 30 == 0) {
            uart_write_string("  R=");
            print_float3(roll);
            uart_write_string(" P=");
            print_float3(pitch);
            uart_write_string("  FL:");
            uart_write_uint32((uint32_t)(pwm[0]+0.5f));
            uart_write_string(" FR:");
            uart_write_uint32((uint32_t)(pwm[1]+0.5f));
            uart_write_string(" RR:");
            uart_write_uint32((uint32_t)(pwm[2]+0.5f));
            uart_write_string(" RL:");
            uart_write_uint32((uint32_t)(pwm[3]+0.5f));
            uart_write_string("\r\n");
        }
    }

    if (count_b > 10) {
        float avg[4];
        for (int k = 0; k < 4; k++) avg[k] = sum_pwm_b[k] / (float)count_b;
        float front_avg = (avg[0] + avg[1]) / 2.0f;  /* FL + FR */
        float rear_avg  = (avg[2] + avg[3]) / 2.0f;  /* RR + RL */

        uart_write_string("\r\n  Nose-down avg (");
        uart_write_uint32(count_b);
        uart_write_string(" samples with P<-10):\r\n");
        for (int k = 0; k < 4; k++) {
            uart_write_string("    ");
            uart_write_string(mnames[k]);
            uart_write_string(": ");
            uart_write_uint32((uint32_t)(avg[k]+0.5f));
            uart_write_string(" us\r\n");
        }
        uart_write_string("  Front (FL+FR) avg: ");
        uart_write_uint32((uint32_t)(front_avg+0.5f));
        uart_write_string("   Rear (RR+RL) avg: ");
        uart_write_uint32((uint32_t)(rear_avg+0.5f));
        uart_write_string("\r\n");

        if (rear_avg > front_avg + 20.0f)
            print_pass("NOSE DOWN: Rear gets more thrust → pushes nose UP → corrects");
        else if (front_avg > rear_avg + 20.0f)
            print_fail("NOSE DOWN: FRONT gets more thrust → pushes nose FURTHER DOWN → INVERTED!");
        else
            uart_write_string("  [WARN] No clear front/rear asymmetry. Tilt more?\r\n");
    } else {
        uart_write_string("  [WARN] Not enough tilted samples. Pitch never went below -10 deg.\r\n");
    }

    /* ============================================================
     * TILT C: Pitch NOSE UP
     *
     * Expected: NN increases FRONT motors (FL, FR).
     * More front thrust = negative pitch torque = nose goes DOWN = correct.
     * ============================================================ */
    uart_write_string("\r\n========================================\r\n");
    uart_write_string("TILT C: Pitch NOSE UP (~20 deg)\r\n");
    uart_write_string("  Lift the FRONT so the nose rises.\r\n");
    uart_write_string("  Return to FLAT first, then...\r\n");
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
    uart_write_string("TILT NOSE UP AND HOLD\r\n\r\n");

    float sum_pwm_c[4] = {0};
    int count_c = 0;

    for (int i = 0; i < 300; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        float roll  = ahrs_get_roll();
        float pitch = ahrs_get_pitch();

        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
        obs[6] = clampf(roll  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[8] = 0.0f; obs[9] = 0.0f; obs[10] = 0.0f; obs[11] = 0.0f;

        float nn_out[4];
        nn_forward(obs, nn_out);
        float pwm[4];
        for (int k = 0; k < 4; k++) {
            float cmd = clampf(NN_HOVER_THROTTLE + clampf(nn_out[k], -1.0f, 1.0f) * NN_ACTION_SCALE, 0.0f, 1.0f);
            pwm[k] = 1000.0f + cmd * 1000.0f;
        }

        if (pitch > 10.0f) {
            for (int k = 0; k < 4; k++) sum_pwm_c[k] += pwm[k];
            count_c++;
        }

        if (i % 30 == 0) {
            uart_write_string("  R=");
            print_float3(roll);
            uart_write_string(" P=");
            print_float3(pitch);
            uart_write_string("  FL:");
            uart_write_uint32((uint32_t)(pwm[0]+0.5f));
            uart_write_string(" FR:");
            uart_write_uint32((uint32_t)(pwm[1]+0.5f));
            uart_write_string(" RR:");
            uart_write_uint32((uint32_t)(pwm[2]+0.5f));
            uart_write_string(" RL:");
            uart_write_uint32((uint32_t)(pwm[3]+0.5f));
            uart_write_string("\r\n");
        }
    }

    if (count_c > 10) {
        float avg[4];
        for (int k = 0; k < 4; k++) avg[k] = sum_pwm_c[k] / (float)count_c;
        float front_avg = (avg[0] + avg[1]) / 2.0f;
        float rear_avg  = (avg[2] + avg[3]) / 2.0f;

        uart_write_string("\r\n  Nose-up avg (");
        uart_write_uint32(count_c);
        uart_write_string(" samples with P>10):\r\n");
        for (int k = 0; k < 4; k++) {
            uart_write_string("    ");
            uart_write_string(mnames[k]);
            uart_write_string(": ");
            uart_write_uint32((uint32_t)(avg[k]+0.5f));
            uart_write_string(" us\r\n");
        }
        uart_write_string("  Front (FL+FR) avg: ");
        uart_write_uint32((uint32_t)(front_avg+0.5f));
        uart_write_string("   Rear (RR+RL) avg: ");
        uart_write_uint32((uint32_t)(rear_avg+0.5f));
        uart_write_string("\r\n");

        if (front_avg > rear_avg + 20.0f)
            print_pass("NOSE UP: Front gets more thrust → pushes nose DOWN → corrects");
        else if (rear_avg > front_avg + 20.0f)
            print_fail("NOSE UP: REAR gets more thrust → pushes nose FURTHER UP → INVERTED!");
        else
            uart_write_string("  [WARN] No clear front/rear asymmetry. Tilt more?\r\n");
    } else {
        uart_write_string("  [WARN] Not enough tilted samples. Pitch never exceeded +10 deg.\r\n");
    }

    /* ---- Summary ---- */
    print_banner("TEST 8 COMPLETE");
    uart_write_string("If any FAIL appeared, the motor mapping or axis signs\r\n");
    uart_write_string("are wrong. The NN correction goes to the wrong side,\r\n");
    uart_write_string("causing the 'compounding errors' behavior.\r\n\r\n");
    uart_write_string("Most likely fix: swap motor wires or fix NN_TO_MOTOR[].\r\n");
    uart_write_string("The motor that SHOULD get more thrust is printed above.\r\n");
    uart_write_string("Compare against which PHYSICAL motor is at that position.\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  █████
 *     ██    ██      ██         ██    ██   ██
 *     ██    █████   ███████    ██     ████
 *     ██    ██           ██    ██    ██   ██
 *     ██    ███████ ███████    ██     █████
 *
 *  IIR FILTER LAG TEST
 *
 *  Purpose: Same as test 8, but WITH the adaptive IIR filter in
 *           the loop. Shows how much the filter delays corrections.
 *           If ALPHA_SLOW=0.08 eats the first few degrees of
 *           correction, the drone tilts further before the filter
 *           lets the correction through — causing the "compounding"
 *           feel even when the direction is correct.
 *
 *  Procedure: Tilt right ~20 deg and hold. Test runs for 5 seconds.
 *           Shows both raw NN output and post-IIR output side by side.
 *
 * ============================================================ */
static void test9_iir_lag(imu_type_t imu)
{
    print_banner("TEST 9: IIR Filter Lag Test");
    uart_write_string("Shows raw NN output vs post-IIR filtered output.\r\n");
    uart_write_string("If the filter adds too much lag, corrections arrive\r\n");
    uart_write_string("too late and the drone tilts further before correcting.\r\n\r\n");

    /* Calibrate */
    ahrs_init();
    systick_init_100hz();

    float cal_gx = 0, cal_gy = 0, cal_gz = 0;
    float cal_ax = 0, cal_ay = 0, cal_az = 0;
    int cal_n = 0;

    uart_write_string("  Calibrating (2 sec, keep FLAT)...\r\n");
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
        if (i >= 50) {
            cal_gx += gx; cal_gy += gy; cal_gz += gz;
            cal_ax += ax; cal_ay += ay; cal_az += az;
            cal_n++;
        }
    }
    float gb[3] = {0}, ab[3] = {0};
    if (cal_n > 0) {
        float inv = 1.0f / (float)cal_n;
        gb[0] = cal_gx*inv; gb[1] = cal_gy*inv; gb[2] = cal_gz*inv;
        ab[0] = cal_ax*inv; ab[1] = cal_ay*inv; ab[2] = cal_az*inv - 1.0f;
    }

    ahrs_init();
    for (int i = 0; i < 100; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);
    }

    uart_write_string("  Now tilt RIGHT ~20 deg and hold.\r\n");
    uart_write_string("  Preparing in 3...");
    delay_ms(1000);
    uart_write_string("2...");
    delay_ms(1000);
    uart_write_string("1...");
    delay_ms(1000);
    uart_write_string("TILT NOW\r\n\r\n");

    /* IIR filter state — starts at zero like main.c */
    float nn_smoothed[4] = {0, 0, 0, 0};

    uart_write_string("  Roll   | Raw NN (FR)  | IIR out (FR) | Alpha | Lag(us)\r\n");
    uart_write_string("  -------+--------------+--------------+-------+--------\r\n");

    for (int i = 0; i < 500; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
        float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
        float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
        float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
        float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
        float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
        ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

        float roll = ahrs_get_roll();
        float pitch = ahrs_get_pitch();

        float obs[12];
        obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
        obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
        obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
        obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
        obs[6] = clampf(roll  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[7] = clampf(pitch * OBS_ANGLE_SCALE, -1.0f, 1.0f);
        obs[8] = 0; obs[9] = 0; obs[10] = 0; obs[11] = 0;

        float nn_out[4];
        nn_forward(obs, nn_out);
        for (int k = 0; k < 4; k++)
            nn_out[k] = clampf(nn_out[k], -1.0f, 1.0f);

        /* Save raw FR output before IIR */
        float raw_fr = nn_out[1];

        /* Apply IIR — same as main.c */
        float alpha_used = 0;
        for (int k = 0; k < 4; k++) {
            float diff = nn_out[k] - nn_smoothed[k];
            float adiff = diff < 0.0f ? -diff : diff;
            float alpha = adiff >= ACTION_SNAP_THRESH
                            ? ACTION_ALPHA_FAST : ACTION_ALPHA_SLOW;
            if (k == 1) alpha_used = alpha;  /* track FR's alpha */
            nn_smoothed[k] += alpha * diff;
        }
        float iir_fr = nn_smoothed[1];

        /* Compute PWM difference (lag in µs) */
        float raw_pwm = (NN_HOVER_THROTTLE + raw_fr * NN_ACTION_SCALE) * 1000.0f;
        float iir_pwm = (NN_HOVER_THROTTLE + iir_fr * NN_ACTION_SCALE) * 1000.0f;
        float lag_us = raw_pwm - iir_pwm;

        /* Print at 10 Hz */
        if (i % 10 == 0) {
            uart_write_string("  ");
            print_float3(roll);
            uart_write_string(" | ");
            print_float3(raw_fr);
            uart_write_string("      | ");
            print_float3(iir_fr);
            uart_write_string("      | ");
            print_float3(alpha_used);
            uart_write_string(" | ");
            print_float3(lag_us);
            uart_write_string("\r\n");
        }
    }

    uart_write_string("\r\n  Key: if 'Lag' stays large (>50us) and Alpha stays at 0.08,\r\n");
    uart_write_string("  the IIR filter is suppressing corrections. The drone will\r\n");
    uart_write_string("  feel sluggish and tilt further before correcting.\r\n\r\n");
    uart_write_string("  If Alpha switches to 0.60 quickly, the filter is working\r\n");
    uart_write_string("  as intended — fast tracking for real corrections.\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██████
 *     ██    ██      ██         ██    ███ ██  ██ ██
 *     ██    █████   ███████    ██     ██ ██  ██
 *     ██    ██           ██    ██     ██ ██  ██
 *     ██    ███████ ███████    ██     ██  ██████
 *
 *  MOTOR WIRE VERIFICATION (MOTORS SPIN!)
 *
 *  ** REMOVE PROPELLERS BEFORE RUNNING THIS TEST **
 *
 *  Purpose: You tilt the board, and the test SPINS the motor
 *           that the NN says should be strongest. You physically
 *           verify: is the motor that spins on the correct side?
 *
 *  This is the DEFINITIVE motor wiring test. Test 8 proved the
 *  NN computes the right correction. This test proves the
 *  correction reaches the right physical motor.
 *
 *  Procedure:
 *    1. REMOVE ALL PROPS
 *    2. Tilt right → the motor on the right side should pulse
 *    3. Tilt nose down → a rear motor should pulse
 *    4. Tilt nose up → a front motor should pulse
 *
 *  The test prints which nn output / motor index it's firing
 *  AND which physical position it SHOULD be. You watch and verify.
 *
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
    uart_write_string("Tilt the board → the NN picks the strongest motor →\r\n");
    uart_write_string("that motor pulses briefly. Verify it's on the correct side.\r\n\r\n");
    uart_write_string("Starting in 5 seconds (remove props NOW)...\r\n");
    delay_ms(5000);

    /* Init motors */
    motor_init();
    delay_ms(2000);  /* ESC arming */

    /* Calibrate IMU */
    ahrs_init();
    systick_init_100hz();

    float cal_gx = 0, cal_gy = 0, cal_gz = 0;
    float cal_ax = 0, cal_ay = 0, cal_az = 0;
    int cal_n = 0;

    uart_write_string("  Calibrating (2 sec, keep FLAT)...\r\n");
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
        if (i >= 50) {
            cal_gx += gx; cal_gy += gy; cal_gz += gz;
            cal_ax += ax; cal_ay += ay; cal_az += az;
            cal_n++;
        }
    }
    float gb[3] = {0}, ab[3] = {0};
    if (cal_n > 0) {
        float inv = 1.0f / (float)cal_n;
        gb[0] = cal_gx*inv; gb[1] = cal_gy*inv; gb[2] = cal_gz*inv;
        ab[0] = cal_ax*inv; ab[1] = cal_ay*inv; ab[2] = cal_az*inv - 1.0f;
    }

    ahrs_init();
    for (int i = 0; i < 100; i++) {
        while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
        imu_raw_t s;
        imu_read_raw(imu, &s);
        ahrs_update(
            (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0],
            (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1],
            (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2],
            (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0],
            (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1],
            (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2],
            0.01f);
    }

    static const char *nn_names[4] = {"FL(nn0)", "FR(nn1)", "RR(nn2)", "RL(nn3)"};

    /* Run 3 tilt phases */
    static const struct {
        const char *label;
        const char *instruction;
        int         axis;          /* 0=roll, 1=pitch */
        float       threshold;     /* positive or negative trigger */
        const char *expected_side;
    } phases[3] = {
        {"TILT RIGHT",    "Tilt RIGHT side down ~20 deg", 0, +15.0f,
         "RIGHT side (FR or RR)"},
        {"NOSE DOWN",     "Tilt NOSE DOWN ~20 deg",       1, -15.0f,
         "REAR (RR or RL)"},
        {"NOSE UP",       "Tilt NOSE UP ~20 deg",         1, +15.0f,
         "FRONT (FL or FR)"},
    };

    for (int p = 0; p < 3; p++) {
        uart_write_string("\r\n========================================\r\n");
        uart_write_string(phases[p].label);
        uart_write_string("\r\n  ");
        uart_write_string(phases[p].instruction);
        uart_write_string("\r\n  Expected motor: ");
        uart_write_string(phases[p].expected_side);
        uart_write_string("\r\n  Preparing in 3...");
        delay_ms(1000);
        uart_write_string("2...");
        delay_ms(1000);
        uart_write_string("1...");
        delay_ms(1000);
        uart_write_string("GO\r\n\r\n");

        int fired = 0;

        for (int i = 0; i < 400 && !fired; i++) {
            while (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {}
            imu_raw_t s;
            imu_read_raw(imu, &s);
            float gx = (float)ALIGN_GX(s.gx) * GYRO_RAW_TO_RADS - gb[0];
            float gy = (float)ALIGN_GY(s.gy) * GYRO_RAW_TO_RADS - gb[1];
            float gz = (float)ALIGN_GZ(s.gz) * GYRO_RAW_TO_RADS - gb[2];
            float ax = (float)ALIGN_AX(s.ax) * accel_raw_to_g   - ab[0];
            float ay = (float)ALIGN_AY(s.ay) * accel_raw_to_g   - ab[1];
            float az = (float)ALIGN_AZ(s.az) * accel_raw_to_g   - ab[2];
            ahrs_update(gx, gy, gz, ax, ay, az, 0.01f);

            float roll  = ahrs_get_roll();
            float pitch = ahrs_get_pitch();
            float angle = (phases[p].axis == 0) ? roll : pitch;

            /* Print at 5 Hz */
            if (i % 20 == 0) {
                uart_write_string("  R=");
                print_float3(roll);
                uart_write_string(" P=");
                print_float3(pitch);
                uart_write_string("  (waiting for ");
                print_float3(phases[p].threshold);
                uart_write_string(" deg)\r\n");
            }

            /* Check if threshold exceeded */
            int triggered = 0;
            if (phases[p].threshold > 0 && angle > phases[p].threshold) triggered = 1;
            if (phases[p].threshold < 0 && angle < phases[p].threshold) triggered = 1;

            if (triggered) {
                /* Run NN to find strongest motor */
                float obs[12];
                obs[0] = clampf(gx / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
                obs[1] = clampf(gy / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
                obs[2] = clampf(gz / OBS_MAX_GYRO_RADS, -1.0f, 1.0f);
                obs[3] = clampf(ax / 2.0f, -1.0f, 1.0f);
                obs[4] = clampf(ay / 2.0f, -1.0f, 1.0f);
                obs[5] = clampf(az / 2.0f, -1.0f, 1.0f);
                obs[6] = clampf(roll  * OBS_ANGLE_SCALE, -1.0f, 1.0f);
                obs[7] = clampf(pitch * OBS_ANGLE_SCALE, -1.0f, 1.0f);
                obs[8] = 0; obs[9] = 0; obs[10] = 0; obs[11] = 0;

                float nn_out[4];
                nn_forward(obs, nn_out);

                /* Find the motor with highest NN output */
                int strongest = 0;
                for (int k = 1; k < 4; k++) {
                    if (nn_out[k] > nn_out[strongest]) strongest = k;
                }

                int motor_idx = TEST_NN_TO_MOTOR[strongest];

                uart_write_string("\r\n  ** TRIGGERED at ");
                print_float3(angle);
                uart_write_string(" deg **\r\n");
                uart_write_string("  NN outputs: ");
                for (int k = 0; k < 4; k++) {
                    uart_write_string(nn_names[k]);
                    uart_write_string("=");
                    print_float3(nn_out[k]);
                    uart_write_string("  ");
                }
                uart_write_string("\r\n");
                uart_write_string("  Strongest: ");
                uart_write_string(nn_names[strongest]);
                uart_write_string(" → motor_set(");
                uart_write_uint32(motor_idx);
                uart_write_string(")\r\n");
                uart_write_string("  Expected position: ");
                uart_write_string(phases[p].expected_side);
                uart_write_string("\r\n\r\n");

                /* PULSE that motor at 30% for 500ms, 3 times */
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
        }

        if (!fired) {
            uart_write_string("  [WARN] Tilt threshold not reached in 4 seconds.\r\n");
        }

        /* Return to flat between phases */
        uart_write_string("\r\n  Return board to FLAT.\r\n");
        delay_ms(3000);
    }

    motor_all_stop();

    print_banner("TEST 10 COMPLETE");
    uart_write_string("For each tilt, verify the pulsing motor is at the\r\n");
    uart_write_string("expected position on the frame.\r\n\r\n");
    uart_write_string("If the wrong motor pulsed:\r\n");
    uart_write_string("  1. Note which nn output → which motor index → which pin\r\n");
    uart_write_string("  2. Swap the ESC wires so the correct physical motor\r\n");
    uart_write_string("     is connected to the correct pin\r\n");
    uart_write_string("  3. OR fix NN_TO_MOTOR[] in main.c\r\n");
}


/* ============================================================
 *
 *  ████████ ███████ ███████ ████████  ██  ██
 *     ██    ██      ██         ██    ███  ██
 *     ██    █████   ███████    ██     ██  ██
 *     ██    ██           ██    ██     ██  ██
 *     ██    ███████ ███████    ██     ██  ██
 *
 *  IIR LIMIT-CYCLE STRESS TEST  (pure computation — no hardware)
 *
 *  Purpose: Determine whether the adaptive IIR filter
 *           (ALPHA_SLOW / ALPHA_FAST / SNAP_THRESH) can generate
 *           a self-sustaining PWM oscillation when its input
 *           carries noise near the snap threshold.
 *
 *  This is the root-cause test for motor pulsing that passes all
 *  other tests: bench stable → in flight with vibration noise
 *  the NN output straddling SNAP_THRESH causes alpha to alternate
 *  FAST/SLOW every tick → motor command steps ±N µs at 10-50 Hz.
 *
 *  Three synthetic input scenarios:
 *    A: small noise (±0.05 around 0.10) — never crosses threshold
 *       Expected: ALPHA_SLOW throughout, <10 PWM swings > 5 µs
 *    B: on-threshold noise (±0.08 around 0.15) — straddles SNAP_THRESH
 *       Expected: limit cycle if threshold too close to noise floor
 *    C: large step 0 → +0.5 — verifies FAST path actually tracks
 *       Expected: converges within ~10 ticks
 *
 *  Pass metric: PWM toggles > 5 µs in 500 ticks
 *    A,B < 10 toggles: PASS
 *    A,B >= 10:        FAIL → raise ACTION_SNAP_THRESH to 0.25+
 *
 * ============================================================ */
static void test11_iir_threshold_stress(void)
{
    print_banner("TEST 11: IIR Limit-Cycle Stress Test");
    uart_write_string("Feeds synthetic NN outputs into the IIR filter.\r\n");
    uart_write_string("Detects limit cycles that cause motor pulsing.\r\n");
    uart_write_string("No hardware needed — runs immediately (~2 sec).\r\n\r\n");

    uart_write_string("  Current parameters:\r\n");
    uart_write_string("    ALPHA_SLOW  = "); print_float3(ACTION_ALPHA_SLOW);  uart_write_string("\r\n");
    uart_write_string("    ALPHA_FAST  = "); print_float3(ACTION_ALPHA_FAST);  uart_write_string("\r\n");
    uart_write_string("    SNAP_THRESH = "); print_float3(ACTION_SNAP_THRESH); uart_write_string("\r\n\r\n");

    #define IIR11_TICKS      500
    #define IIR11_MIN_SWING  5.0f   /* µs — swing larger than this = visible pulse */

    /* ────────────────────────────────────────────────────────────
     * SCENARIO A: small noise — square wave 0.05 → 0.15 at 20 Hz
     *   Amplitude 0.05 centered at 0.10.  The difference (nn - smoothed)
     *   should stay below SNAP_THRESH almost always → ALPHA_SLOW dominates
     *   and the output barely moves.  Zero/few PWM toggles expected.
     * ──────────────────────────────────────────────────────────── */
    uart_write_string("Scenario A: small noise  (0.05↔0.15, amp=0.05, center=0.10)\r\n");
    uart_write_string("  Expected: <10 PWM swings, ALPHA_SLOW throughout\r\n\r\n");
    {
        float smoothed  = 0.0f;
        float prev_pwm  = -1.0f;
        int   toggles   = 0;
        int   fast_ticks = 0;

        for (int i = 0; i < IIR11_TICKS; i++) {
            /* 20 Hz square wave: 5-tick high / 5-tick low */
            float nn    = (i % 10 < 5) ? 0.15f : 0.05f;
            float diff  = nn - smoothed;
            float adiff = diff < 0.0f ? -diff : diff;
            float alpha = adiff >= ACTION_SNAP_THRESH
                            ? ACTION_ALPHA_FAST : ACTION_ALPHA_SLOW;
            if (alpha >= ACTION_ALPHA_FAST) fast_ticks++;
            smoothed += alpha * diff;

            float pwm = 1000.0f + (NN_HOVER_THROTTLE + smoothed * NN_ACTION_SCALE) * 1000.0f;
            if (prev_pwm >= 0.0f) {
                float swing = pwm - prev_pwm;
                if (swing < 0.0f) swing = -swing;
                if (swing > IIR11_MIN_SWING) toggles++;
            }
            prev_pwm = pwm;

            if (i % 100 == 0) {
                uart_write_string("  ["); uart_write_uint32(i);
                uart_write_string("] nn="); print_float3(nn);
                uart_write_string(" iir="); print_float3(smoothed);
                uart_write_string(" pwm="); uart_write_uint32((uint32_t)(pwm + 0.5f));
                uart_write_string(" a="); print_float3(alpha);
                uart_write_string("\r\n");
            }
        }

        uart_write_string("\r\n  Toggles>"); uart_write_uint32((uint32_t)IIR11_MIN_SWING);
        uart_write_string("us: "); uart_write_uint32(toggles);
        uart_write_string("   FAST-alpha ticks: "); uart_write_uint32(fast_ticks);
        uart_write_string("\r\n");
        if (toggles < 10)
            print_pass("Scenario A: stable — small noise absorbed by ALPHA_SLOW");
        else
            print_fail("Scenario A: pulsing — noise floor is crossing SNAP_THRESH");
    }

    uart_write_string("\r\n");

    /* ────────────────────────────────────────────────────────────
     * SCENARIO B: on-threshold noise — square wave 0.07 → 0.23 at 20 Hz
     *   Amplitude 0.08 centered at 0.15 (= SNAP_THRESH).
     *   The input alternately pushes diff above and then below the
     *   threshold, causing alpha to flip FAST↔SLOW every few ticks.
     *   Classic limit-cycle condition for adaptive IIR filters.
     * ──────────────────────────────────────────────────────────── */
    uart_write_string("Scenario B: on-threshold noise  (0.07↔0.23, amp=0.08, center=0.15)\r\n");
    uart_write_string("  This is the vibration scenario in flight.\r\n");
    uart_write_string("  Expected: FAIL if SNAP_THRESH too close to noise floor\r\n\r\n");
    {
        float smoothed   = 0.0f;
        float prev_pwm   = -1.0f;
        int   toggles    = 0;
        int   fast_ticks = 0;
        int   alpha_flips = 0;
        float prev_alpha = -1.0f;

        for (int i = 0; i < IIR11_TICKS; i++) {
            float nn    = (i % 10 < 5) ? 0.23f : 0.07f;
            float diff  = nn - smoothed;
            float adiff = diff < 0.0f ? -diff : diff;
            float alpha = adiff >= ACTION_SNAP_THRESH
                            ? ACTION_ALPHA_FAST : ACTION_ALPHA_SLOW;
            if (alpha >= ACTION_ALPHA_FAST) fast_ticks++;
            if (prev_alpha >= 0.0f && alpha != prev_alpha) alpha_flips++;
            prev_alpha = alpha;
            smoothed += alpha * diff;

            float pwm = 1000.0f + (NN_HOVER_THROTTLE + smoothed * NN_ACTION_SCALE) * 1000.0f;
            if (prev_pwm >= 0.0f) {
                float swing = pwm - prev_pwm;
                if (swing < 0.0f) swing = -swing;
                if (swing > IIR11_MIN_SWING) toggles++;
            }
            prev_pwm = pwm;

            if (i % 100 == 0) {
                uart_write_string("  ["); uart_write_uint32(i);
                uart_write_string("] nn="); print_float3(nn);
                uart_write_string(" iir="); print_float3(smoothed);
                uart_write_string(" pwm="); uart_write_uint32((uint32_t)(pwm + 0.5f));
                uart_write_string(" a="); print_float3(alpha);
                uart_write_string("\r\n");
            }
        }

        uart_write_string("\r\n  Toggles>"); uart_write_uint32((uint32_t)IIR11_MIN_SWING);
        uart_write_string("us: "); uart_write_uint32(toggles);
        uart_write_string("   FAST-ticks: "); uart_write_uint32(fast_ticks);
        uart_write_string("   Alpha-flips: "); uart_write_uint32(alpha_flips);
        uart_write_string("\r\n");
        if (toggles < 10)
            print_pass("Scenario B: stable — no limit cycle at snap threshold");
        else {
            print_fail("LIMIT CYCLE CONFIRMED — on-threshold noise causes pulsing");
            uart_write_string("  Fix: raise ACTION_SNAP_THRESH (try 0.25) in main.c\r\n");
            uart_write_string("       or reduce ALPHA_SLOW (try 0.05) to damp the cycle.\r\n");
        }
    }

    uart_write_string("\r\n");

    /* ────────────────────────────────────────────────────────────
     * SCENARIO C: large step 0 → +0.5 at t=5 ticks
     *   Verifies the FAST path actually provides rapid convergence
     *   when a real correction event arrives.
     * ──────────────────────────────────────────────────────────── */
    uart_write_string("Scenario C: large step 0.0 → +0.5  (should converge in <15 ticks)\r\n\r\n");
    {
        float smoothed = 0.0f;
        int   ticks_to_converge = -1;

        for (int i = 0; i < 80; i++) {
            float nn    = (i < 5) ? 0.0f : 0.5f;
            float diff  = nn - smoothed;
            float adiff = diff < 0.0f ? -diff : diff;
            float alpha = adiff >= ACTION_SNAP_THRESH
                            ? ACTION_ALPHA_FAST : ACTION_ALPHA_SLOW;
            smoothed += alpha * diff;

            float err = smoothed - 0.5f;
            if (err < 0.0f) err = -err;
            if (ticks_to_converge < 0 && i >= 5 && err < 0.05f)
                ticks_to_converge = i - 5;

            if (i % 10 == 0 || i == 5) {
                uart_write_string("  ["); uart_write_uint32(i);
                uart_write_string("] iir="); print_float3(smoothed);
                uart_write_string(" err="); print_float3(smoothed - 0.5f);
                uart_write_string(" a="); print_float3(alpha);
                uart_write_string("\r\n");
            }
        }

        if (ticks_to_converge >= 0 && ticks_to_converge <= 15) {
            uart_write_string("  Converged in "); uart_write_uint32(ticks_to_converge);
            uart_write_string(" ticks after step\r\n");
            print_pass("Scenario C: FAST path working — large corrections track quickly");
        } else if (ticks_to_converge < 0) {
            print_fail("Scenario C: did not converge — FAST path may not be triggering");
        } else {
            uart_write_string("  Converged in "); uart_write_uint32(ticks_to_converge);
            uart_write_string(" ticks (slow)\r\n");
            print_fail("Scenario C: convergence too slow — raise ALPHA_FAST");
        }
    }

    /* Summary */
    uart_write_string("\r\n---- IIR TUNING GUIDE ----\r\n");
    uart_write_string("  If Scenario B FAILED: raise SNAP_THRESH from 0.15 → 0.25.\r\n");
    uart_write_string("  If motors still pulse: lower ALPHA_SLOW from 0.08 → 0.05.\r\n");
    uart_write_string("  If drone feels sluggish after fix: lower ALPHA_SLOW to 0.10.\r\n");
    uart_write_string("  Nuclear option: remove adaptive IIR, use fixed ALPHA = 0.20.\r\n");
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
 *  At Kd=0.08 that's ±40 µs throttle oscillation at 25 Hz
 *  (same as a ~5% PWM step). Exactly the period of pulsing.
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
        uart_write_string("  Fix: reduce ALT_KD (try 0.02), or increase VVEL_ALPHA in baro.c.\r\n");
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
        uart_write_string("    1. Reduce ALT_KD:       0.08 → 0.02\r\n");
        uart_write_string("    2. Reduce VVEL_ALPHA:   0.3  → 0.1 in baro.c\r\n");
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
    uart_write_string("       base (NN_HOVER_THROTTLE)  = 1323 us\r\n");
    uart_write_string("       + P-alt loop cap (+0.10)  = 1423 us MAX SUSTAINED\r\n");
    uart_write_string("     hover PWM <= 1370 us  → healthy margin      PASS\r\n");
    uart_write_string("     hover PWM 1370-1423   → marginal, will sag  WARN\r\n");
    uart_write_string("     hover PWM  > 1423 us  → CANNOT HOVER. Set\r\n");
    uart_write_string("       NN_HOVER_THROTTLE = (hover_us - 1000)/1000 in\r\n");
    uart_write_string("       weights.h AND retrain/re-verify.           FAIL\r\n\r\n");

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
    uart_write_string("#   Motors OFF — safe for bench use     #\r\n");
    uart_write_string("########################################\r\n\r\n");

    /* ── 4. SPI + IMU detect ── */
    delay_ms(100);
    spi1_init();
    delay_ms(10);

    uint8_t whoami = 0;
    imu_type_t imu = imu_detect(&whoami);

    uart_write_string("IMU WHO_AM_I = 0x");
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

    /* ── 6. AHRS init ── */
    ahrs_init();

    /* ── 7. I2C + Barometer ── */
    i2c1_init();
    uint8_t baro_id = 0;
    baro_type_t baro = baro_init(&baro_id);
    int baro_ok = (baro != BARO_NONE);

    if (baro_ok) {
        uart_write_string("Barometer detected (id=0x");
        uart_write_hex8(baro_id);
        uart_write_string(")\r\n");
    } else {
        uart_write_string("No barometer detected (tests 5 obs[8,9] will be 0)\r\n");
    }

    /* ============================================================
     * RUN TESTS
     * ============================================================ */

    /* TEST 1: Register readback — immediate */
    test1_spi_readback(imu);

    /* TEST 2: Static characterization — 5 seconds, board flat */
    test2_static_imu(imu);

    /* TEST 3: Dynamic axis alignment — ~30 seconds, interactive */
    test3_dynamic_axis(imu);

    /* TEST 4: AHRS convergence — ~20 seconds, interactive */
    test4_ahrs_convergence(imu);

    /* TEST 5: Observation vector ground truth — 5 seconds, board flat */
    test5_obs_vector(imu, baro_ok);

    /* TEST 6: NN inference validation — immediate, pure computation */
    test6_nn_inference();

    /* TEST 7: HIL closed loop — 10 seconds, board flat */
    test7_hil_closed_loop(imu, baro_ok);

    /* TEST 8: Live tilt → motor response — interactive */
    test8_tilt_response(imu);

    /* TEST 9: IIR filter lag — interactive, tilt right */
    test9_iir_lag(imu);

    /* TEST 10: Motor wire verification — PROPS OFF, motors spin */
    test10_motor_wire_verify(imu);

    /* TEST 11: IIR limit-cycle stress — immediate, pure computation */
    test11_iir_threshold_stress();

    /* TEST 12: Loop timing jitter & overrun detector — 5 seconds, board flat */
    test12_loop_jitter(imu, baro_ok);

    /* TEST 13: Baro throttle noise characterization — 8 sec, board flat */
    test13_baro_throttle_noise(imu, baro_ok);

    /* TEST 14: Throttle staircase — PROPS ON, DRONE STRAPPED DOWN.
     * Measures the REAL hover PWM (vs the 1323 us the sim assumes) and
     * vibration / AHRS integrity at flight power. Runs last: 15 s
     * countdown gives you time to power off if the rig isn't ready. */
    test14_throttle_staircase(imu);

    /* ── Done ── */
    print_banner("ALL TESTS COMPLETE");
    uart_write_string("Review output above for PASS/FAIL.\r\n");
    uart_write_string("LED solid = done.\r\n");

    led_on();
    while (1) { /* halt */ }

    return 0;
}
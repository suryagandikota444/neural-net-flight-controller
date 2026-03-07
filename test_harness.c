/*
 * test_harness.c — Systematic hardware validation for drone FC
 *
 * ── BUILD ──────────────────────────────────────────────────────
 *
 *   Compile this INSTEAD of main.c:
 *     Replace main.c with test_harness.c in your Makefile, or:
 *     arm-none-eabi-gcc ... test_harness.c startup.c clock.c spi.c
 *                           imu.c uart.c ahrs.c memset.c -o test.elf
 *
 *   Do NOT link motor.c, nn.c, baro.c, i2c.c, weights.h — not needed.
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

    /* ============================================================
     * RUN TESTS
     * ============================================================ */

    /* TEST 1: Register readback — immediate */
    test1_spi_readback(imu);

    /* TEST 2: Static characterization — 5 seconds, board flat */
    test2_static_imu(imu);

    /* TEST 3: Dynamic axis alignment — ~30 seconds, interactive */
    test3_dynamic_axis(imu);

    /* ── Done ── */
    print_banner("ALL TESTS COMPLETE");
    uart_write_string("Review output above for PASS/FAIL.\r\n");
    uart_write_string("LED solid = done.\r\n");

    led_on();
    while (1) { /* halt */ }

    return 0;
}
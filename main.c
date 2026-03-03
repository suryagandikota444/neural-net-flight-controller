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
#define LOOP_TIMING_DEBUG 0
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
#define TARGET_ALT         0.3f     /* hover 30 cm above baro ground reference */
#define ALT_SCALE          2.0f     /* ±2 m altitude error maps to ±1 */
#define VVEL_SCALE         5.0f     /* ±5 m/s vertical vel maps to ±1 */
#define HVEL_SCALE         3.0f     /* ±3 m/s horizontal vel maps to ±1 */

/* Simple P-altitude controller — supplements the NN which is only good
 * at attitude control.  The NN's obs[8] is clamped to a tiny band so
 * it stays in its trained regime; this P-loop handles the rest.
 *
 * alt_throttle_adj = Kp*(target - alt) + Kd*(-vvel)
 * Added to NN_HOVER_THROTTLE before mixing with NN residual.
 */
#define ALT_OBS_CLAMP      0.05f    /* clamp obs[8] to ±0.05 — NN comfort zone */
#define ALT_KP             0.15f    /* throttle/m — gentle P gain */
#define ALT_KD             0.08f    /* throttle/(m/s) — damp oscillation */
#define ALT_ADJ_MAX        0.10f    /* cap P-altitude adjustment ±10% throttle */


/* ============================================================
 * AXIS ALIGNMENT — Sologood F405 IMU mounting
 *
 * Empirically verified (test phases 2 & 3):
 *   Chip X = Board X (forward)    — no sign change
 *   Chip Y = Board Y (left)       — no sign change
 *   Chip Z = Board Z (up)         — no sign change
 *
 * The chip's native frame matches the MuJoCo NWU convention
 * (X-forward, Y-left, Z-up) directly. At rest:
 *   ax≈0, ay≈0, az≈+1g. Tilt right → ay>0. Nose down → ax>0.
 *
 * NOTE: Betaflight reports this board as "CW180°", but that
 * refers to BF's NED body frame, not NWU.  When converting
 * CW180°-in-NED to NWU the negations cancel out, yielding
 * identity alignment.
 * ============================================================ */
#define ALIGN_AX(raw)  ( (raw))
#define ALIGN_AY(raw)  ( -(raw))
#define ALIGN_AZ(raw)  ( (raw))
#define ALIGN_GX(raw)  ( -(raw))
#define ALIGN_GY(raw)  ( (raw))
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
#ifndef DEBUG_INTERVAL
#define DEBUG_INTERVAL     0          /* 0=off; set to 50 for 2 Hz telemetry (blocks loop at 115200!) */
#endif

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

/* 3-decimal-place float printer for obs/nn debug.
 * Output: "+1.234" or "-0.003" — always sign + digits.dot.3digits */
static void print_float3(float val)
{
    if (val < 0.0f) { uart2_write_char('-'); val = -val; }
    else             { uart2_write_char('+'); }
    uint32_t whole = (uint32_t)val;
    uint32_t frac  = (uint32_t)((val - (float)whole) * 1000.0f + 0.5f);
    if (frac >= 1000) { whole++; frac -= 1000; }
    uart2_write_uint32(whole);
    uart2_write_char('.');
    /* Zero-pad the fractional part to 3 digits */
    if (frac < 100) uart2_write_char('0');
    if (frac < 10)  uart2_write_char('0');
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
    delay_ms(500);  /* give terminal time to connect */


	#if LOOP_TIMING_DEBUG
		dwt_cycle_counter_init();
	#endif

    /* ── 4. SPI ─────────────────────────────────────────────── */
    delay_ms(100);
    spi1_init();
    delay_ms(10);

    /* ── 5. IMU detection & init ────────────────────────────── */
    uint8_t whoami = 0;
    imu_type_t imu = imu_detect(&whoami);

    switch (imu) {
        case IMU_MPU6000: break;
        case IMU_ICM42688P: break;
        case IMU_BMI270: break;
        default:
            uart2_write_string("IMU: NOT FOUND\r\n");
            while (1) { blink_error(); }
    }

    imu_init(imu);

    /* Set accel scale factor based on detected IMU chip */
    switch (imu) {
        case IMU_ICM42688P:
            accel_raw_to_g = 1.0f / 2048.0f;   /* ±16g → 2048 LSB/g */
            break;
        default:
            accel_raw_to_g = 1.0f / 4096.0f;   /* ±8g → 4096 LSB/g */
            break;
    }

    /* ── 6. Motor PWM init ──────────────────────────────────── */
    motor_init();

    /* ── 7. AHRS init ───────────────────────────────────────── */
    ahrs_init();

    /* ── 8. I2C + Barometer ─────────────────────────────────── */
    i2c1_init();

    uint8_t baro_id = 0;
    baro_type_t baro = baro_init(&baro_id);
    int baro_ok = (baro != BARO_NONE);

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
		/* Step 1: Full throttle — ESCs learn the maximum */
		motors_full_throttle();

		/* Blink LED rapidly while waiting — signals "active calibration" */
		for (int i = 0; i < 40; i++) {
			led_on();  delay_ms(50);
			led_off(); delay_ms(50);
		}

		/* Step 2: Drop to idle — ESCs learn the minimum */
		motor_all_stop();

		/* Slow blink while ESCs confirm */
		for (int i = 0; i < 8; i++) {
			led_on();  delay_ms(250);
			led_off(); delay_ms(250);
		}

		/* Done */
		led_on();
		while (1) { /* halt */ }

	#else /* Normal flight mode */

	/* ── 9. ESC arming (3 seconds of idle PWM) ──────────────── */
	delay_ms(3000);

	/* ── 9b. Motor identification test ──────────────────────
		* Fires each motor individually at ~30% throttle for 300 ms.
		* Order: FL → FR → RL → RR
		* Watch/listen to verify motor mapping is correct.
		* ** REMOVE ALL PROPELLERS BEFORE POWERING ON **
		*/
	{
		static const struct { int idx; const char *name; } motor_test[4] = {
			{ 3, "FL" },   /* idx3 = Front-Left  (PA9) */
			{ 2, "FR" },   /* idx2 = Front-Right (PA8) */
			{ 0, "RL" },   /* idx0 = Rear-Left   (PC8) */
			{ 1, "RR" },   /* idx1 = Rear-Right  (PC9) */
		};
		for (int m = 0; m < 4; m++) {
			uart2_write_string("MOTOR TEST: ");
			uart2_write_string(motor_test[m].name);
			uart2_write_string("\r\n");
			motor_set(motor_test[m].idx, 0.50f);
			delay_ms(300);
			motor_all_stop();
			delay_ms(300);
		}
		uart2_write_string("MOTOR TEST DONE\r\n");
	}

	/* ── 10. Barometer ground reference ─────────────────────── */
	if (baro_ok) {
		/* Let the BMP280 IIR filter settle by reading for 2 seconds */
		uart2_write_string("Barometer: settling...\r\n");
		for (int i = 0; i < 50; i++) {
			baro_update(0.04f);
			delay_ms(40);
		}

		baro_set_ground_reference();
	}

	/* ── 11. Start SysTick for 100 Hz ───────────────────────── */
	systick_init_hz(LOOP_HZ);
	led_on();

	/* ── 12. Main control loop ──────────────────────────────── */

	imu_raw_t  sample;
	float      motor_cmd[4] = {0};
	uint32_t   tick_count = 0;
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
			}
			continue;
		}

		/* ── E: Safety — tilt limit ───────────────────────── */
		if (my_fabsf(roll_deg) > TILT_LIMIT_DEG ||
			my_fabsf(pitch_deg) > TILT_LIMIT_DEG)
		{
			motor_all_stop();
			// continue;
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
		float alt_throttle_adj = 0.0f;
		float alt  = 0.0f;
		float vvel = 0.0f;
		if (baro_ok) {
			alt            = baro_get_altitude();
			float alt_err  = alt - TARGET_ALT;         /* positive = above target */
			vvel           = baro_get_vvel();           /* positive = going up */

			/* Feed the NN a tightly clamped altitude signal so it stays
			* in its ≈0 comfort zone (attitude-only regime). */
			obs[8]  = clampf(alt_err / ALT_SCALE, -ALT_OBS_CLAMP, ALT_OBS_CLAMP);
			obs[9]  = clampf(vvel / VVEL_SCALE, -1.0f, 1.0f);

			/* Simple P+D altitude hold — adjusts base throttle */
			alt_throttle_adj = clampf(
				ALT_KP * (TARGET_ALT - alt) + ALT_KD * (-vvel),
				-ALT_ADJ_MAX, ALT_ADJ_MAX
			);
		} else {
			obs[8]  = 0.0f;
			obs[9]  = 0.0f;
		}

		/* Horizontal velocity — not available without optical flow / GPS */
		obs[10] = 0.0f;
		obs[11] = 0.0f;

		/* Print obs at ~10 Hz */
		// {
		// 	static uint32_t obs_print_counter = 0;
		// 	if (++obs_print_counter >= 10) {
		// 		obs_print_counter = 0;
		// 		uart2_write_string("OBS:");
		// 		for (int k = 0; k < 12; k++) {
		// 			uart2_write_char(' ');
		// 			print_float3(obs[k]);
		// 		}
		// 		/* Raw gyro (rad/s), accel (g), baro (m, m/s) */
		// 		uart2_write_string(" | GYR:");
		// 		uart2_write_char(' '); print_float3(gx);
		// 		uart2_write_char(' '); print_float3(gy);
		// 		uart2_write_char(' '); print_float3(gz);
		// 		uart2_write_string(" | ACC:");
		// 		uart2_write_char(' '); print_float3(ax);
		// 		uart2_write_char(' '); print_float3(ay);
		// 		uart2_write_char(' '); print_float3(az);
		// 		uart2_write_string(" | ALT:");
		// 		uart2_write_char(' '); print_float3(alt);
		// 		uart2_write_string(" VV:");
		// 		uart2_write_char(' '); print_float3(vvel);
		// 		uart2_write_string("\r\n");
		// 	}
		// }

		

		#if LOOP_TIMING_DEBUG
			uint32_t t5 = dwt_cycle_now();
		#endif

		/* ── G: NN forward pass (~15 µs) ──────────────────── */
		float nn_out[4];
		nn_forward(obs, nn_out);

		/* Clip NN output to [-1,1] — matches training env np.clip(action,-1,1) */
		for (int k = 0; k < 4; k++) nn_out[k] = clampf(nn_out[k], -1.0f, 1.0f);

		/* Action smoothing — adaptive non-linear IIR filter.
			*
			* Two alphas:
			*   ALPHA_SLOW: used when the change is small (noise regime)
			*               → heavy smoothing, kills high-freq chatter
			*   ALPHA_FAST: used when the change is large (genuine control input)
			*               → fast tracking, no lag on real attitude corrections
			*
			* The threshold SNAP_THRESH separates the two regimes.
			* If |raw - smoothed| < SNAP_THRESH → use ALPHA_SLOW
			* If |raw - smoothed| >= SNAP_THRESH → use ALPHA_FAST
			*
			* Noise from vibration/sim-to-real gap is typically small (<0.15).
			* Real NN corrections during a tilt event are larger (>0.15).
			*/
		#define ACTION_ALPHA_SLOW   0.08f   /* heavy smoothing for noise */
		#define ACTION_ALPHA_FAST   0.60f   /* fast tracking for real inputs */
		#define ACTION_SNAP_THRESH  0.15f   /* boundary between regimes */

		static float nn_smoothed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		for (int k = 0; k < 4; k++) {
			float diff  = nn_out[k] - nn_smoothed[k];
			float alpha = (diff < 0.0f ? -diff : diff) >= ACTION_SNAP_THRESH
							? ACTION_ALPHA_FAST
							: ACTION_ALPHA_SLOW;
			nn_smoothed[k] += alpha * diff;
		}
		for (int k = 0; k < 4; k++)
			nn_out[k] = nn_smoothed[k];

		#if LOOP_TIMING_DEBUG
			uint32_t t6 = dwt_cycle_now();
		#endif

		/* ── H: Residual action → motor command ──────────── */
		float base_throttle = NN_HOVER_THROTTLE + alt_throttle_adj;
		for (int i = 0; i < 4; i++) {
			motor_cmd[i] = clampf(
				base_throttle + nn_out[i] * NN_ACTION_SCALE,
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
				/* Ramp the base throttle while preserving NN residual direction.
				* At ramp_frac=0 motors sit at 0; at ramp_frac=1 full command.
				* A minimum spin floor keeps all motors above ESC deadband
				* so slow-spinning motors don't stall during ramp-up. */
				float nn_residual = nn_out[i] * NN_ACTION_SCALE;
				float ramped_base = ramp_frac * NN_HOVER_THROTTLE;
				float floor       = ramp_frac * 0.06f; /* ~1060 µs min spin */
				float cmd = ramped_base + ramp_frac * nn_residual;
				motor_cmd[i] = clampf(cmd, floor, 1.0f);
			}
		}

		#if LOOP_TIMING_DEBUG
				uint32_t t7 = dwt_cycle_now();
		#endif

		/* ── I: Write to motors (NN→Betaflight mapping) ─── */
		for (int i = 0; i < 4; i++) {
			motor_set(NN_TO_MOTOR[i], motor_cmd[i]);
		}

		/* Print motor outputs at 10 Hz */
		{
			static uint32_t mot_print_counter = 0;
			if (++mot_print_counter >= 10) {
				mot_print_counter = 0;
				/* FL FR RR RL in µs */
				uart2_write_string("MOT FL:");
				uart2_write_uint32(1000 + (uint32_t)(motor_cmd[0] * 1000.0f));
				uart2_write_string(" FR:");
				uart2_write_uint32(1000 + (uint32_t)(motor_cmd[1] * 1000.0f));
				uart2_write_string(" RR:");
				uart2_write_uint32(1000 + (uint32_t)(motor_cmd[2] * 1000.0f));
				uart2_write_string(" RL:");
				uart2_write_uint32(1000 + (uint32_t)(motor_cmd[3] * 1000.0f));
				uart2_write_string("\r\n");
			}
		}
	}

	#endif

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

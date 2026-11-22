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
#include "flight_safety.h"
#include "nn.h"        /* Also provides NN_HOVER_THROTTLE, NN_ACTION_SCALE */
#include "i2c.h"
#include "baro.h"
#include "fc_config.h"  /* shared: axis alignment, obs scales, altitude gains, action filter */
#include "crsf.h"       /* ExpressLRS / CRSF decoder                                   */
#include "rc.h"         /* arming, kill switch, failsafe, modes, stick setpoints       */
#include "rc_input.h"   /* USART2 receiver driver                                      */


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

/*
 * RAW_LOG_MODE: 1 (default) streams a TEL telemetry line at 5 Hz from the
 *   control loop (see "Full telemetry log" below). Transmission is
 *   interrupt-driven and non-blocking.
 *
 *   Log 45 s with:  python fc-nn/log_imu.py --port /dev/cu.SLAB_USBtoUART --seconds 45
 *   Then analyse:   python fc-nn/analyze_imu.py imu_log.csv
 *
 *   ** This does NOT disable motors. Since 2026-11-21 the firmware only
 *      arms from the radio's ARM switch (see rc.h) and no longer runs the
 *      boot motor test (MOTOR_ID_TEST=0), but bench-test props-off anyway. **
 */
#ifndef RAW_LOG_MODE
#define RAW_LOG_MODE 1
#endif

/*
 * MOTOR_ID_TEST: 1 spins each motor to 50 % for 300 ms at boot (props OFF!)
 *   to check motor order. Default 0 since RC arming was added: the flight
 *   firmware must never spin a motor without the ARM switch.
 *   Harness TEST 10 does the same check.  Build with: make MOTOR_ID_TEST=1
 */
#ifndef MOTOR_ID_TEST
#define MOTOR_ID_TEST 0
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
                                 uint32_t iir_cycles,
                                 uint32_t map_cycles,
                                 uint32_t pwm_cycles)
{
    uart_write_string("LT us tot:");
    uart_write_uint32(cycles_to_us(total_cycles));
    uart_write_string(" imu:");
    uart_write_uint32(cycles_to_us(imu_cycles));
    uart_write_string(" conv:");
    uart_write_uint32(cycles_to_us(conv_cycles));
    uart_write_string(" ahrs:");
    uart_write_uint32(cycles_to_us(ahrs_cycles));
    uart_write_string(" baro:");
    uart_write_uint32(cycles_to_us(baro_cycles));
    uart_write_string(" obs:");
    uart_write_uint32(cycles_to_us(obs_cycles));
    uart_write_string(" nn:");
    uart_write_uint32(cycles_to_us(nn_cycles));
    uart_write_string(" iir:");
    uart_write_uint32(cycles_to_us(iir_cycles));
    uart_write_string(" map:");
    uart_write_uint32(cycles_to_us(map_cycles));
    uart_write_string(" pwm:");
    uart_write_uint32(cycles_to_us(pwm_cycles));
    uart_write_string("\r\n");
}


/* ============================================================
 * SENSOR CONVERSION / OBSERVATION / ALTITUDE CONSTANTS
 *
 * GYRO_RAW_TO_RADS, MAX_GYRO_RADS, ANGLE_OBS_SCALE, TARGET_ALT,
 * ALT_SCALE, VVEL_SCALE, HVEL_SCALE, ALT_OBS_CLAMP, ALT_KP, ALT_KD,
 * ALT_ADJ_MAX, the ALIGN_xx macros and the action filter all come from
 * fc_config.h so test_harness.c and the host tests use identical values.
 *
 * Altitude design (unchanged): a simple P+D loop adjusts base throttle;
 * the NN's obs[8] is clamped to ±ALT_OBS_CLAMP.
 *   alt_throttle_adj = ALT_KP*(target - alt) + ALT_KD*(-vvel)
 * ============================================================ */

/* Accel scale factor — set at runtime based on detected IMU:
 *   ICM42688P: ±16g → 2048 LSB/g → 1/2048
 *   MPU6000:   ±8g  → 4096 LSB/g → 1/4096
 *   BMI270:    ±8g  → 4096 LSB/g → 1/4096
 */
static float accel_raw_to_g;


/* ============================================================
 * AXIS ALIGNMENT — Sologood F405 IMU mounting
 *
 * Defined in fc_config.h (ALIGN_AX … ALIGN_GZ). The board is mounted
 * CW180, so chip → body is (-x, -y, +z) for BOTH gyro and accel.
 * Expected on the bench (training convention, drone_env.py):
 *   nose down  → ax < 0, pitch > 0
 *   right down → ay > 0, roll  > 0
 *   CW from above → gz < 0
 * ============================================================ */


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
    if (val < 0.0f) { uart_write_char('-'); val = -val; }
    else             { uart_write_char('+'); }
    uint32_t whole = (uint32_t)val;
    uint32_t frac  = (uint32_t)((val - (float)whole) * 10.0f) % 10;
    uart_write_uint32(whole);
    uart_write_char('.');
    uart_write_uint32(frac);
}

/* 3-decimal-place float printer for obs/nn debug.
 * Output: "+1.234" or "-0.003" — always sign + digits.dot.3digits */
static void print_float3(float val)
{
    if (val < 0.0f) { uart_write_char('-'); val = -val; }
    else             { uart_write_char('+'); }
    uint32_t whole = (uint32_t)val;
    uint32_t frac  = (uint32_t)((val - (float)whole) * 1000.0f + 0.5f);
    if (frac >= 1000) { whole++; frac -= 1000; }
    uart_write_uint32(whole);
    uart_write_char('.');
    /* Zero-pad the fractional part to 3 digits */
    if (frac < 100) uart_write_char('0');
    if (frac < 10)  uart_write_char('0');
    uart_write_uint32(frac);
}

static void print_telemetry(float roll, float pitch,
                            const float motor_cmd[4],
                            float alt_m, float vvel_mps,
                            float pressure_pa, float temp_c,
                            int baro_ok)
{
    static const char *labels[4] = {"FL:", "FR:", "RR:", "RL:"};

    uart_write_string("R:");
    print_fixed1(roll);
    uart_write_string(" P:");
    print_fixed1(pitch);
    uart_write_string(" | ");
    for (int i = 0; i < 4; i++) {
        uart_write_string(labels[i]);
        uart_write_uint32(1000 + (uint32_t)(motor_cmd[i] * 1000.0f));
        if (i < 3) uart_write_char(' ');
    }

    if (baro_ok) {
        uart_write_string(" | A:");
        print_fixed1(alt_m);
        uart_write_string(" V:");
        print_fixed1(vvel_mps);
        uart_write_string(" T:");
        print_fixed1(temp_c);
        uart_write_string(" Pa:");
        uart_write_uint32((uint32_t)pressure_pa);
    } else {
        uart_write_string(" | BARO:none");
    }

    uart_write_string("\r\n");
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
 * RC STATUS LINE (2 Hz, armed or not)
 *
 *   RCS,link,lq,armed,mode,block,cmd_roll,cmd_pitch,target_alt,alt,
 *       roll,pitch,ch1..ch6(us),rc_frames,crc_err,failsafes
 *
 * block = RC_BLOCK_* bitmask from rc.h (why arming is refused).
 * Non-blocking: skipped if the TX buffer cannot take the whole line.
 * ============================================================ */
static void print_rc_status(const crsf_t *c, const rc_state_t *r, int link_ok,
                            float alt, float roll, float pitch)
{
	static uint32_t div = 0;
	if (++div < 50) return;
	div = 0;
	if (uart_tx_free() < 200U) return;
	uart_write_string("RCS,");
	uart_write_uint32((uint32_t)link_ok);      uart_write_char(',');
	uart_write_uint32(c->lq);                  uart_write_char(',');
	uart_write_uint32((uint32_t)r->armed);     uart_write_char(',');
	uart_write_string(rc_mode_name(r->mode));  uart_write_char(',');
	uart_write_uint32((uint32_t)r->block);     uart_write_char(',');
	print_float3(r->cmd_roll_deg);             uart_write_char(',');
	print_float3(r->cmd_pitch_deg);            uart_write_char(',');
	print_float3(r->target_alt);               uart_write_char(',');
	print_float3(alt);                         uart_write_char(',');
	print_float3(roll);                        uart_write_char(',');
	print_float3(pitch);                       uart_write_char(',');
	for (int k = 0; k < 6; k++) {
		uart_write_uint32(crsf_to_us(c->ch[k])); uart_write_char(',');
	}
	uart_write_uint32(c->rc_frames);           uart_write_char(',');
	uart_write_uint32(c->crc_errors);          uart_write_char(',');
	uart_write_uint32(r->failsafe_count);
	uart_write_string("\r\n");
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
    uart_init(115200);
    delay_ms(500);  /* give terminal time to connect */

    /* ── 3b. RC receiver (ExpressLRS, CRSF on USART2) ───────── */
    rc_input_init();


	dwt_cycle_counter_init();

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
            uart_write_string("IMU: NOT FOUND\r\n");
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

	#if MOTOR_ID_TEST
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
			uart_write_string("MOTOR TEST: ");
			uart_write_string(motor_test[m].name);
			uart_write_string("\r\n");
			motor_set(motor_test[m].idx, 0.50f);
			delay_ms(300);
			motor_all_stop();
			delay_ms(300);
		}
		uart_write_string("MOTOR TEST DONE\r\n");
	}
	#endif /* MOTOR_ID_TEST */

	/* ── 10. Barometer ground reference ─────────────────────── */
	if (baro_ok) {
		/* Let the BMP280 IIR filter settle by reading for 2 seconds */
		uart_write_string("Barometer: settling...\r\n");
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
	int        calibrated = 0;     /* IMU bias calibration done (2 s settle) */
	uint32_t   ramp_counter = 0;

	/* RC link + pilot command state (see rc.h) */
	static crsf_t     crsf;
	static rc_state_t rc;
	crsf_init(&crsf);
	rc_init(&rc);
	uint32_t   rc_link_age = RC_LINK_TIMEOUT_TICKS;   /* ticks since last RC frame */

	/* Baro ground reference, re-taken at every arm (fc_config.h ALT_ZERO_ALPHA) */
	float      alt_zero = 0.0f;
	int        alt_zero_init = 0;

	/* NN action filter state — reset on every arm */
	float      nn_smoothed[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	float gyro_bias[3]  = {0.0f, 0.0f, 0.0f};
    float accel_bias[3] = {0.0f, 0.0f, 0.0f};   /* ax_bias, ay_bias (az handled separately) */
    float cal_sum_gx = 0, cal_sum_gy = 0, cal_sum_gz = 0;
    float cal_sum_ax = 0, cal_sum_ay = 0, cal_sum_az = 0;
    int   cal_count = 0;
	
	uint32_t   timing_counter = 0;

	while (1) {

		/* Wait for next 100 Hz tick (poll SysTick COUNTFLAG) */
		if (!(SYST_CSR & SYST_CSR_COUNTFLAG)) {
			continue;
		}

		tick_count++;

		uint32_t t0 = dwt_cycle_now();

		/* ── A: Read IMU (SPI burst, ~10 µs) ──────────────── */
		imu_read_raw(imu, &sample);

		uint32_t t1 = dwt_cycle_now();

		/* ── B: Axis alignment (CW180°) + unit conversion ──── */
		float gx = (float)ALIGN_GX(sample.gx) * GYRO_RAW_TO_RADS;
        float gy = (float)ALIGN_GY(sample.gy) * GYRO_RAW_TO_RADS;
        float gz = (float)ALIGN_GZ(sample.gz) * GYRO_RAW_TO_RADS;

        float ax = (float)ALIGN_AX(sample.ax) * accel_raw_to_g;
        float ay = (float)ALIGN_AY(sample.ay) * accel_raw_to_g;
        float az = (float)ALIGN_AZ(sample.az) * accel_raw_to_g;

        /* Apply bias correction (computed during settle phase).
         * Before arming, biases are all 0 so this is a no-op.
         * After arming, this removes persistent accel/gyro offsets.
         *
         * Gyro: subtract average at rest (should be ~0 rad/s)
         * Accel: subtract offset from expected gravity vector [0, 0, +1g]
         *   ax_corrected = ax - ax_avg     (expected 0)
         *   ay_corrected = ay - ay_avg     (expected 0)
         *   az_corrected = az - (az_avg - 1.0)  = az - az_avg + 1.0
         */
        gx -= gyro_bias[0];
        gy -= gyro_bias[1];
        gz -= gyro_bias[2];
        ax -= accel_bias[0];
        ay -= accel_bias[1];
        az -= accel_bias[2];

		uint32_t t2 = dwt_cycle_now();

		/* ── C: AHRS update (Madgwick, ~5 µs) ──────────────── */
		ahrs_update(gx, gy, gz, ax, ay, az, LOOP_DT);

		uint32_t t3 = dwt_cycle_now();

		float roll_deg  = ahrs_get_roll();
		float pitch_deg = ahrs_get_pitch();

		/* ── C2: Barometer update (~25 Hz, decimated internally) */
		if (baro_ok) {
			baro_update(LOOP_DT);
		}

		uint32_t t4 = dwt_cycle_now();

		/* ── D: Settling check + IMU bias calibration ──────────
         *
         * During the 2-second settle period:
         *   - AHRS runs (converges to approximate orientation)
         *   - We accumulate accel and gyro readings for bias cal
         *   - Skip the last 50 ticks (first 150 only) to avoid
         *     counting samples while AHRS is still way off
         *
         * At arming:
         *   - Compute gyro bias as simple average (should be ~0)
         *   - Compute accel bias: ax,ay average should be 0,
         *     az average should be +1g. We subtract the offset
         *     from the expected value.
         */
        /* Drain the receiver every tick, also during calibration, so its
         * ring buffer never overflows and the link state is current. */
        if (rc_input_poll(&crsf) > 0) rc_link_age = 0;
        else if (rc_link_age < RC_LINK_TIMEOUT_TICKS) rc_link_age++;
        int link_ok = rc_link_age < RC_LINK_TIMEOUT_TICKS;

        if (!calibrated) {
            /* Accumulate during ticks 50–199 (skip first 0.5s for IMU warmup) */
            if (tick_count >= 50 && tick_count < SETTLE_TICKS) {
                cal_sum_gx += gx;
                cal_sum_gy += gy;
                cal_sum_gz += gz;
                cal_sum_ax += ax;
                cal_sum_ay += ay;
                cal_sum_az += az;
                cal_count++;
            }

            if (tick_count >= SETTLE_TICKS) {
                /* Compute biases */
                if (cal_count > 0) {
                    float inv_n = 1.0f / (float)cal_count;
                    gyro_bias[0] = cal_sum_gx * inv_n;
                    gyro_bias[1] = cal_sum_gy * inv_n;
                    gyro_bias[2] = cal_sum_gz * inv_n;
                    accel_bias[0] = cal_sum_ax * inv_n;         /* expect 0, subtract offset */
                    accel_bias[1] = cal_sum_ay * inv_n;         /* expect 0, subtract offset */
                    accel_bias[2] = cal_sum_az * inv_n - 1.0f;  /* expect +1g, subtract offset from 1g */
                }

                calibrated = 1;
            }
            motor_all_stop();
            continue;
        }

		/* ── E: RC link, arming, kill switch, setpoints ───────
		 *
		 * Decode everything the receiver sent since the last tick.
		 * No valid RC frame for RC_LINK_TIMEOUT_TICKS (250 ms) = link lost.
		 * rc_update() disarms on ARM switch OFF (kill) or link loss
		 * (failsafe) on THIS tick, before any motor command is computed. */
		{
			rc_inputs_t in;
			in.link_ok    = link_ok;
			in.roll       = crsf_to_unit(crsf.ch[RC_CH_ROLL]);
			in.pitch      = crsf_to_unit(crsf.ch[RC_CH_PITCH]);
			in.throttle   = crsf_to_unit(crsf.ch[RC_CH_THROTTLE]);
			in.yaw        = crsf_to_unit(crsf.ch[RC_CH_YAW]);
			in.arm_sw     = crsf_to_unit(crsf.ch[RC_CH_ARM]);
			in.mode_sw    = crsf_to_unit(crsf.ch[RC_CH_MODE]);
			in.calibrated = calibrated;
			in.roll_deg   = roll_deg;
			in.pitch_deg  = pitch_deg;
			rc_update(&rc, &in, LOOP_DT);
		}

		/* Ground reference follows the baro while disarmed, frozen once armed */
		if (baro_ok && !rc.armed) {
			float a = baro_get_altitude();
			if (!alt_zero_init) { alt_zero = a; alt_zero_init = 1; }
			alt_zero += ALT_ZERO_ALPHA * (a - alt_zero);
		}

		print_rc_status(&crsf, &rc, link_ok,
		                baro_ok ? baro_get_altitude() - alt_zero : 0.0f,
		                roll_deg, pitch_deg);

		if (!rc.armed) {
			/* DISARMED: motors at 1000 µs, controller state reset so the
			 * next arm starts exactly like a training ground start. */
			motor_all_stop();
			ramp_counter = 0;
			for (int k = 0; k < 4; k++) nn_smoothed[k] = 0.0f;
			continue;
		}

		/* A fault must bypass this tick's computation AND all future outputs. */
		if (!ahrs_is_valid()) flight_safety_latch();
		if (!flight_safety_check(roll_deg, pitch_deg, TILT_LIMIT_DEG)) {
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
		/* Attitude ERROR vs the pilot setpoint (0 in HOVER mode) — the same
		 * roll − cmd_roll the training env uses (drone_env.py _get_obs). */
		obs[6] = clampf((roll_deg  - rc.cmd_roll_deg)  * ANGLE_OBS_SCALE, -1.0f, 1.0f);
		obs[7] = clampf((pitch_deg - rc.cmd_pitch_deg) * ANGLE_OBS_SCALE, -1.0f, 1.0f);

		/* Altitude error and vertical velocity from barometer */
		float alt_throttle_adj = 0.0f;
		float alt  = 0.0f;
		float vvel = 0.0f;
		if (baro_ok) {
			alt            = baro_get_altitude() - alt_zero;   /* above arm point */
			float target   = rc.target_alt;            /* TARGET_ALT in HOVER mode */
			float alt_err  = alt - target;             /* positive = above target */
			vvel           = baro_get_vvel();           /* positive = going up */

			/* Feed the NN a tightly clamped altitude signal so it stays
			* in its ≈0 comfort zone (attitude-only regime). */
			obs[8]  = clampf(alt_err / ALT_SCALE, -ALT_OBS_CLAMP, ALT_OBS_CLAMP);
			obs[9]  = clampf(vvel / VVEL_SCALE, -1.0f, 1.0f);

			/* Simple P+D altitude hold — adjusts base throttle */
			alt_throttle_adj = clampf(
				ALT_KP * (target - alt) + ALT_KD * (-vvel),
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
		// 		uart_write_string("OBS:");
		// 		for (int k = 0; k < 12; k++) {
		// 			uart_write_char(' ');
		// 			print_float3(obs[k]);
		// 		}
		// 		/* Raw gyro (rad/s), accel (g), baro (m, m/s) */
		// 		uart_write_string(" | GYR:");
		// 		uart_write_char(' '); print_float3(gx);
		// 		uart_write_char(' '); print_float3(gy);
		// 		uart_write_char(' '); print_float3(gz);
		// 		uart_write_string(" | ACC:");
		// 		uart_write_char(' '); print_float3(ax);
		// 		uart_write_char(' '); print_float3(ay);
		// 		uart_write_char(' '); print_float3(az);
		// 		uart_write_string(" | ALT:");
		// 		uart_write_char(' '); print_float3(alt);
		// 		uart_write_string(" VV:");
		// 		uart_write_char(' '); print_float3(vvel);
		// 		uart_write_string("\r\n");
		// 	}
		// }

		

		uint32_t t5 = dwt_cycle_now();

		/* ── G: NN forward pass (~15 µs) ──────────────────── */
		float nn_out[4];
		nn_forward(obs, nn_out);

		/* Clip NN output to [-1,1] — matches training env np.clip(action,-1,1) */
		for (int k = 0; k < 4; k++) nn_out[k] = clampf(nn_out[k], -1.0f, 1.0f);

		uint32_t t6 = dwt_cycle_now();  /* after NN inference */

		/* Action smoothing — fixed-alpha EMA, identical to training
		 * (drone_env.py: smooth = 0.3*action + 0.7*smooth).
		 * See fc_config.h for why the old adaptive 0.08/0.60 filter
		 * was removed (it was ~4x slower than training for small
		 * corrections). */
		action_filter_step(nn_smoothed, nn_out);
		for (int k = 0; k < 4; k++)
			nn_out[k] = nn_smoothed[k];

		uint32_t t7 = dwt_cycle_now();  /* after IIR smoothing */

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

		uint32_t t8 = dwt_cycle_now();  /* after motor command mapping */

		/* ── I: Write to motors (NN→Betaflight mapping) ─── */
		flight_safety_write_motors(NN_TO_MOTOR, motor_cmd);

		/* ── Full telemetry log (5 Hz) ───────────────────────
		 * Format: TEL,gx,gy,gz,ax,ay,az,roll,pitch,
		 *         obs[0..11],nn[0..3],mot[0..3]
		 * Total: 28 float values, comma-separated, ≤ ~260 bytes.
		 * 100 Hz / 20 = 5 Hz → ~1.3 kB/s (115200 baud carries ~11.5 kB/s).
		 *
		 * Non-blocking: uart_write_* only copies into the 1 KB background
		 * transmit buffer (see uart.c). We only start a line if the whole
		 * line fits, so the control loop never waits on the serial port.
		 * If a line is skipped, a "TELDROP,<count>" line is sent once
		 * there is room again (log_imu.py ignores non-TEL lines).
		 */
		#if RAW_LOG_MODE
		{
			#define TEL_MAX_BYTES 320U
			static uint32_t tel_div = 0;
			static uint32_t tel_dropped = 0;
			static uint32_t tel_dropped_reported = 0;
			if (++tel_div >= 20) {   /* 100 Hz / 20 = 5 Hz */
				tel_div = 0;
				if (uart_tx_free() < TEL_MAX_BYTES + 24U) {
					tel_dropped++;
					goto tel_done;
				}
				if (tel_dropped != tel_dropped_reported) {
					uart_write_string("TELDROP,");
					uart_write_uint32(tel_dropped);
					uart_write_string("\r\n");
					tel_dropped_reported = tel_dropped;
				}
				uart_write_string("TEL,");
				/* Raw IMU (6) */
				print_float3(gx); uart_write_char(',');
				print_float3(gy); uart_write_char(',');
				print_float3(gz); uart_write_char(',');
				print_float3(ax); uart_write_char(',');
				print_float3(ay); uart_write_char(',');
				print_float3(az); uart_write_char(',');
				/* AHRS attitude (2) */
				print_float3(roll_deg);  uart_write_char(',');
				print_float3(pitch_deg); uart_write_char(',');
				/* Obs vector (12) */
				for (int k = 0; k < 12; k++) {
					print_float3(obs[k]);
					uart_write_char(',');
				}
				/* NN output post-smoothing (4) */
				print_float3(nn_out[0]); uart_write_char(',');
				print_float3(nn_out[1]); uart_write_char(',');
				print_float3(nn_out[2]); uart_write_char(',');
				print_float3(nn_out[3]); uart_write_char(',');
				/* Final motor commands (4) */
				print_float3(motor_cmd[0]); uart_write_char(',');
				print_float3(motor_cmd[1]); uart_write_char(',');
				print_float3(motor_cmd[2]); uart_write_char(',');
				print_float3(motor_cmd[3]);
				uart_write_string("\r\n");
			}
		tel_done: ;
		}
		#endif

		uint32_t t9 = dwt_cycle_now();  /* after PWM write */
		// if (++timing_counter >= LOOP_TIMING_INTERVAL) {
		// 	timing_counter = 0;
		// 	print_loop_timing_us(
		// 		dwt_cycle_delta(t0, t9),  /* total */
		// 		dwt_cycle_delta(t0, t1),  /* imu   */
		// 		dwt_cycle_delta(t1, t2),  /* conv  */
		// 		dwt_cycle_delta(t2, t3),  /* ahrs  */
		// 		dwt_cycle_delta(t3, t4),  /* baro  */
		// 		dwt_cycle_delta(t4, t5),  /* obs   */
		// 		dwt_cycle_delta(t5, t6),  /* nn    */
		// 		dwt_cycle_delta(t6, t7),  /* iir   */
		// 		dwt_cycle_delta(t7, t8),  /* map   */
		// 		dwt_cycle_delta(t8, t9)   /* pwm   */
		// 	);
		// }

		// /* Print motor outputs at 10 Hz */
		// {
		// 	static uint32_t mot_print_counter = 0;
		// 	if (++mot_print_counter >= 10) {
		// 		mot_print_counter = 0;
		// 		/* FL FR RR RL in µs */
		// 		uart_write_string("MOT FL:");
		// 		uart_write_uint32(1000 + (uint32_t)(motor_cmd[0] * 1000.0f));
		// 		uart_write_string(" FR:");
		// 		uart_write_uint32(1000 + (uint32_t)(motor_cmd[1] * 1000.0f));
		// 		uart_write_string(" RR:");
		// 		uart_write_uint32(1000 + (uint32_t)(motor_cmd[2] * 1000.0f));
		// 		uart_write_string(" RL:");
		// 		uart_write_uint32(1000 + (uint32_t)(motor_cmd[3] * 1000.0f));
		// 		uart_write_string("\r\n");
		// 	}
		// }
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
 * 5. (done 2026-11-21: RC arming/kill/failsafe via ExpressLRS, rc.c)
 * 6. In-flight tuning of NN_HOVER_THROTTLE
 * 7. Complementary filter for vvel (fuse baro + accel Z)
 */

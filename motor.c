/*
 * motor.c — PWM motor output for STM32F405 (bare-metal)
 *
 * ── What Is PWM? ──────────────────────────────────────────────
 *
 * PWM = Pulse Width Modulation. The idea is dead simple:
 *
 *   A pin goes HIGH for some time, then LOW for the rest of a
 *   fixed period. The ratio of HIGH time to total period is
 *   the "duty cycle." By varying the HIGH time, we control
 *   how much power an ESC delivers to a motor.
 *
 *   For RC/drone ESCs, the standard is:
 *     Period:  2500 µs  (400 Hz — common for multirotor ESCs)
 *     HIGH time: 1000 µs = idle (motor off)
 *                2000 µs = full throttle
 *
 *   So the ESC only cares about the pulse width (1000–2000 µs),
 *   not the duty cycle percentage. 1500 µs = 50% throttle.
 *
 *   Timing diagram for one period at ~30% throttle:
 *
 *        1300µs HIGH           1200µs LOW
 *      |<---------->|<--------------------------->|
 *      ┌────────────┐
 *      │            │
 *   ───┘            └─────────────────────────────────
 *      |<------------- 2500µs period ------------>|
 *
 *
 * ── How STM32 Timers Do PWM ───────────────────────────────────
 *
 * A timer has a counter (CNT) that counts up from 0 to ARR
 * (Auto-Reload Register), then resets to 0 and repeats:
 *
 *   CNT: 0 → 1 → 2 → ... → ARR → 0 → 1 → ...
 *
 * Each timer channel has a CCR (Capture/Compare Register).
 * In PWM mode 1:
 *   - Output is HIGH while CNT < CCR
 *   - Output is LOW  while CNT >= CCR
 *
 * So the pulse width is directly controlled by CCR:
 *   CCR = 1000 → 1000 µs pulse → idle
 *   CCR = 1500 → 1500 µs pulse → 50% throttle
 *   CCR = 2000 → 2000 µs pulse → full throttle
 *
 * The counter must tick at 1 MHz for CCR values to directly
 * equal microseconds. We achieve this with:
 *   Timer clock = 168 MHz (APB2 timer clock)
 *   Prescaler = 167 → 168 MHz / (167+1) = 1 MHz
 *
 *
 * ── Advanced Timers (TIM1 and TIM8) ──────────────────────────
 *
 * TIM1 and TIM8 are "advanced-control" timers, designed for
 * motor control applications (how convenient). They have one
 * critical extra feature compared to general-purpose timers:
 *
 *   MOE (Main Output Enable) in the BDTR register.
 *
 * If MOE is not set, ALL timer outputs are forced inactive
 * regardless of CCER settings. This is a safety feature for
 * industrial motor controllers. We MUST set MOE to get any
 * output. Forgetting this is the #1 "why isn't my PWM working"
 * bug with advanced timers.
 *
 *
 * ── Pin Mapping ───────────────────────────────────────────────
 *
 *   Motor 1 → PC8  → TIM8 Channel 3  (AF3)
 *   Motor 2 → PC9  → TIM8 Channel 4  (AF3)
 *   Motor 3 → PA8  → TIM1 Channel 1  (AF1)
 *   Motor 4 → PA9  → TIM1 Channel 2  (AF1)
 *
 * Why these specific AF (Alternate Function) numbers?
 * The STM32F405 datasheet has a big table (Table 9) that lists
 * which peripheral connects to which pin at which AF number.
 * Each pin can do up to 16 different things (AF0–AF15). We
 * pick the AF that routes the timer channel to our pin.
 */

#include <stdint.h>
#include "motor.h"
#include "clock.h"

/* ============================================================
 * REGISTER DEFINITIONS
 * ============================================================ */

/* --- RCC registers --- */
#define RCC_AHB1ENR         (*(volatile uint32_t *)0x40023830)
#define RCC_APB2ENR         (*(volatile uint32_t *)0x40023844)
#define RCC_APB2RSTR        (*(volatile uint32_t *)0x40023824)  /* APB2 peripheral reset */
#define RCC_AHB1ENR_GPIOAEN (1U << 0)
#define RCC_AHB1ENR_GPIOCEN (1U << 2)
#define RCC_APB2ENR_TIM1EN  (1U << 0)
#define RCC_APB2ENR_TIM8EN  (1U << 1)
#define RCC_APB2RSTR_TIM1RST (1U << 0)  /* TIM1 peripheral reset */
#define RCC_APB2RSTR_TIM8RST (1U << 1)  /* TIM8 peripheral reset */

/* --- GPIOA registers (base 0x40020000) --- */
#define GPIOA_MODER     (*(volatile uint32_t *)0x40020000)
#define GPIOA_OTYPER    (*(volatile uint32_t *)0x40020004)
#define GPIOA_OSPEEDR   (*(volatile uint32_t *)0x40020008)
#define GPIOA_AFRH      (*(volatile uint32_t *)0x40020024)  /* Pins 8-15 */

/* --- GPIOC registers (base 0x40020800) --- */
#define GPIOC_MODER     (*(volatile uint32_t *)0x40020800)
#define GPIOC_OTYPER    (*(volatile uint32_t *)0x40020804)
#define GPIOC_OSPEEDR   (*(volatile uint32_t *)0x40020808)
#define GPIOC_AFRH      (*(volatile uint32_t *)0x40020824)  /* Pins 8-15 */

/* --- TIM1 registers (base 0x40010000) ---
 *
 * TIM1 is an advanced-control timer on APB2.
 * We use Channel 1 (PA8 = Motor 3) and Channel 2 (PA9 = Motor 4).
 *
 * Key registers:
 *   CR1    — Counter control (enable, auto-reload preload)
 *   CR2    — Control 2 (CCPC bit — see note below)
 *   CCMR1  — Output Compare mode for CH1 and CH2
 *   CCER   — Capture/Compare output enable
 *   PSC    — Prescaler (divides the timer clock)
 *   ARR    — Auto-Reload (sets the period)
 *   CCR1   — Compare value for CH1 (pulse width)
 *   CCR2   — Compare value for CH2 (pulse width)
 *   BDTR   — Break and Dead-Time register (has MOE bit)
 *   EGR    — Event Generation (force register update)
 *
 * CRITICAL NOTE — CR2.CCPC (bit 0):
 *   When CCPC=1, writes to CCER are redirected to a shadow/preload
 *   register and only take effect on a COM event. If a prior firmware
 *   left CCPC=1 and the MCU was only soft-reset (e.g. DFU — which does
 *   NOT reset peripherals), the timer will silently accept all CCER
 *   writes but never actually enable any output channels. The result
 *   looks exactly like MOE=0: PWM timer runs, counter counts, but zero
 *   output on the pins. CCPC defaults to 0 on power-on reset, but a
 *   DFU reflash does NOT guarantee power-on reset of peripherals.
 *   Fix: use RCC_APB2RSTR to force a peripheral reset before init.
 */
#define TIM1_CR1    (*(volatile uint32_t *)0x40010000)
#define TIM1_CR2    (*(volatile uint32_t *)0x40010004)  /* Has CCPC bit 0 */
#define TIM1_CCMR1  (*(volatile uint32_t *)0x40010018)
#define TIM1_CCER   (*(volatile uint32_t *)0x40010020)
#define TIM1_PSC    (*(volatile uint32_t *)0x40010028)
#define TIM1_ARR    (*(volatile uint32_t *)0x4001002C)
#define TIM1_CCR1   (*(volatile uint32_t *)0x40010034)
#define TIM1_CCR2   (*(volatile uint32_t *)0x40010038)
#define TIM1_BDTR   (*(volatile uint32_t *)0x40010044)
#define TIM1_EGR    (*(volatile uint32_t *)0x40010014)

/* --- TIM8 registers (base 0x40010400) ---
 *
 * Same layout as TIM1, just different base address.
 * We use Channel 3 (PC8 = Motor 1) and Channel 4 (PC9 = Motor 2).
 */
#define TIM8_CR1    (*(volatile uint32_t *)0x40010400)
#define TIM8_CR2    (*(volatile uint32_t *)0x40010404)  /* Has CCPC bit 0 */
#define TIM8_CCMR2  (*(volatile uint32_t *)0x4001041C)
#define TIM8_CCER   (*(volatile uint32_t *)0x40010420)
#define TIM8_PSC    (*(volatile uint32_t *)0x40010428)
#define TIM8_ARR    (*(volatile uint32_t *)0x4001042C)
#define TIM8_CCR3   (*(volatile uint32_t *)0x4001043C)
#define TIM8_CCR4   (*(volatile uint32_t *)0x40010440)
#define TIM8_BDTR   (*(volatile uint32_t *)0x40010444)
#define TIM8_EGR    (*(volatile uint32_t *)0x40010414)

/* --- Timer bit definitions --- */

/* CR1 */
#define TIM_CR1_CEN     (1U << 0)    /* Counter enable */
#define TIM_CR1_ARPE    (1U << 7)    /* Auto-reload preload enable:
                                        When set, writes to ARR are buffered
                                        and take effect at the next update event
                                        (prevents glitches mid-period) */

/* CCMR: Output Compare Mode
 *
 * Each CCMR register controls 2 channels (1&2 or 3&4).
 * Lower byte = odd channel, upper byte = even channel.
 *
 * OCxM (Output Compare x Mode) bits:
 *   110 = PWM mode 1 — output HIGH while CNT < CCR
 *   111 = PWM mode 2 — output LOW while CNT < CCR (inverted)
 *
 * OCxPE (Output Compare x Preload Enable):
 *   When set, CCR writes are buffered and applied at the next
 *   update event. This prevents a half-old, half-new pulse
 *   if you write CCR mid-period.
 */
#define TIM_CCMR_OC_PWM1_CH_LO  (6U << 4 | 1U << 3)    /* 0x0068: OC1/3 PWM1 + preload */
#define TIM_CCMR_OC_PWM1_CH_HI  (6U << 12 | 1U << 11)   /* 0x6800: OC2/4 PWM1 + preload */
#define TIM_CCMR_OC_PWM1_BOTH   (TIM_CCMR_OC_PWM1_CH_LO | TIM_CCMR_OC_PWM1_CH_HI) /* 0x6868 */

/* CCER: Capture/Compare Enable Register
 * Each channel has an Enable bit (CCxE):
 *   CC1E = bit 0,  CC2E = bit 4,  CC3E = bit 8,  CC4E = bit 12
 * Setting CCxE connects the timer output to the GPIO pin. */
#define TIM_CCER_CC1E   (1U << 0)
#define TIM_CCER_CC2E   (1U << 4)
#define TIM_CCER_CC3E   (1U << 8)
#define TIM_CCER_CC4E   (1U << 12)

/* BDTR: Break and Dead-Time Register
 * MOE (Main Output Enable, bit 15):
 *   MUST be set on advanced timers (TIM1/TIM8) for outputs to work.
 *   Without this, timer outputs stay inactive forever. */
#define TIM_BDTR_MOE    (1U << 15)

/* EGR: Event Generation Register
 * UG (Update Generation, bit 0):
 *   Writing 1 forces an update event — reloads PSC, ARR, CCR
 *   from their preload/shadow registers immediately. */
#define TIM_EGR_UG      (1U << 0)


/* ============================================================
 * PWM CONSTANTS
 * ============================================================ */

/* Timer prescaler: 168 MHz / (167+1) = 1 MHz → 1 tick = 1 µs */
#define PWM_PSC     167U

/* Auto-reload: 2500 ticks = 2500 µs = 400 Hz period */
#define PWM_ARR     2499U

/* CCR range: 1000 (idle) to 2000 (full throttle) */
#define PWM_MIN     1000U
#define PWM_MAX     2000U


/* ============================================================
 * motor_init — Configure TIM1 + TIM8 for PWM on motor pins
 * ============================================================
 *
 * After this function returns, all 4 motor pins output a 400 Hz
 * PWM signal with 1000 µs pulse width (ESC idle).
 *
 * Call order in main():
 *   clock_init()  →  motor_init()  →  delay 3s  →  control loop
 * The 3-second delay gives ESCs time to initialize.
 */
void motor_init(void)
{
    /* ── Step 1: Enable peripheral clocks ────────────────────── */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOCEN;
    RCC_APB2ENR |= RCC_APB2ENR_TIM1EN  | RCC_APB2ENR_TIM8EN;
    (void)RCC_APB2ENR;  /* flush — bus write must settle before first register access */

    /* ── Step 1b: Peripheral soft-reset ─────────────────────────
     *
     * DFU (and most soft-reset paths) do NOT reset peripheral registers
     * to their power-on-reset values. If a previous firmware left any
     * timer register in a non-default state — most critically
     * TIM1_CR2.CCPC=1 which silently redirects CCER writes to a shadow
     * register so output channels never actually enable — those stale
     * values will persist and break this init.
     *
     * The ONLY guaranteed fix is the RCC peripheral reset: assert reset,
     * flush, de-assert. After de-assert every register in both timers is
     * at its documented power-on-reset value.
     */
    RCC_APB2RSTR |=  (RCC_APB2RSTR_TIM1RST | RCC_APB2RSTR_TIM8RST);
    (void)RCC_APB2RSTR;  /* flush assert */
    RCC_APB2RSTR &= ~(RCC_APB2RSTR_TIM1RST | RCC_APB2RSTR_TIM8RST);
    (void)RCC_APB2RSTR;  /* flush de-assert */

    /* ── Step 2: Configure GPIO pins as Alternate Function ───── */

    /* --- PA8 (TIM1_CH1, Motor 3) and PA9 (TIM1_CH2, Motor 4) ---
     *
     * GPIOA_AFRH controls Alternate Function for pins 8-15.
     * Each pin gets 4 bits:
     *   PA8 → AFRH bits [3:0]  = AF1 (TIM1)
     *   PA9 → AFRH bits [7:4]  = AF1 (TIM1)
     *
     * GPIOA_MODER:
     *   PA8 → bits [17:16] = 10 (Alternate Function)
     *   PA9 → bits [19:18] = 10 (Alternate Function)
     */
    GPIOA_AFRH &= ~((0xFU << 0) | (0xFU << 4));
    GPIOA_AFRH |=  ((1U   << 0) | (1U   << 4));   /* AF1 = TIM1 */

    GPIOA_MODER &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOA_MODER |=  ((2U << (8 * 2)) | (2U << (9 * 2)));   /* AF mode */

    GPIOA_OSPEEDR |= ((3U << (8 * 2)) | (3U << (9 * 2)));  /* Very high speed */

    /* Ensure push-pull output type (OTYPER bit = 0).
     * DFU/soft-reset may not clear this if a prior firmware set open-drain. */
    GPIOA_OTYPER &= ~((1U << 8) | (1U << 9));

    /* --- PC8 (TIM8_CH3, Motor 1) and PC9 (TIM8_CH4, Motor 2) ---
     *
     * GPIOC_AFRH:
     *   PC8 → bits [3:0] = AF3 (TIM8)
     *   PC9 → bits [7:4] = AF3 (TIM8)
     */
    GPIOC_AFRH &= ~((0xFU << 0) | (0xFU << 4));
    GPIOC_AFRH |=  ((3U   << 0) | (3U   << 4));   /* AF3 = TIM8 */

    GPIOC_MODER &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOC_MODER |=  ((2U << (8 * 2)) | (2U << (9 * 2)));   /* AF mode */

    GPIOC_OSPEEDR |= ((3U << (8 * 2)) | (3U << (9 * 2)));  /* Very high speed */

    /* Ensure push-pull output type */
    GPIOC_OTYPER &= ~((1U << 8) | (1U << 9));

    /* ── Step 3: Configure TIM8 (Motor 1 = CH3, Motor 2 = CH4) ─ */

    /* Ensure CCPC=0 (no preloaded CCER — same fix as TIM1 below) */
    TIM8_CR2 = 0;

    /* CCMR2 controls channels 3 and 4.
     * Set both to PWM mode 1 with preload enabled.
     * CH3 = lower byte, CH4 = upper byte. */
    TIM8_CCMR2 = TIM_CCMR_OC_PWM1_BOTH;

    /* Enable CH3 and CH4 outputs */
    TIM8_CCER = TIM_CCER_CC3E | TIM_CCER_CC4E;

    /* Set prescaler and period */
    TIM8_PSC = PWM_PSC;
    TIM8_ARR = PWM_ARR;

    /* Set initial pulse width to 1000 µs (idle) */
    TIM8_CCR3 = PWM_MIN;
    TIM8_CCR4 = PWM_MIN;

    /* Enable Main Output — required for advanced timers! */
    TIM8_BDTR |= TIM_BDTR_MOE;

    /* Force update event to load all shadow registers */
    TIM8_EGR = TIM_EGR_UG;

    /* Enable counter with auto-reload preload */
    TIM8_CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;

    /* ── Step 4: Configure TIM1 (Motor 3 = CH1, Motor 4 = CH2) ─
     *
     * After the RCC peripheral reset above, all TIM1 registers are
     * guaranteed to be at power-on-reset values (CR2=0, CCER=0, etc.).
     * We still explicitly write CR2=0 here as belt-and-suspenders.
     */
    TIM1_CR2 = 0;  /* Ensure CCPC=0, MMS=0, CCDS=0 — no shadowed CCER */

    /* CCMR1 controls channels 1 and 2. */
    TIM1_CCMR1 = TIM_CCMR_OC_PWM1_BOTH;

    /* Enable CH1 and CH2 outputs */
    TIM1_CCER = TIM_CCER_CC1E | TIM_CCER_CC2E;

    /* Set prescaler and period */
    TIM1_PSC = PWM_PSC;
    TIM1_ARR = PWM_ARR;

    /* Set initial pulse width to 1000 µs (idle) */
    TIM1_CCR1 = PWM_MIN;
    TIM1_CCR2 = PWM_MIN;

    /* Enable Main Output — required for advanced timers! */
    TIM1_BDTR |= TIM_BDTR_MOE;

    /* Force update event to load all shadow registers */
    TIM1_EGR = TIM_EGR_UG;

    /* Enable counter with auto-reload preload */
    TIM1_CR1 = TIM_CR1_ARPE | TIM_CR1_CEN;
}


/* ============================================================
 * motor_set — Set one motor's throttle
 * ============================================================
 *
 * Parameters:
 *   index: 0–3 → Betaflight motor 1–4
 *          0 = M1 → PC8 (TIM8_CH3)
 *          1 = M2 → PC9 (TIM8_CH4)
 *          2 = M3 → PA8 (TIM1_CH1)
 *          3 = M4 → PA9 (TIM1_CH2)
 *   cmd:   0.0 = idle (1000 µs) to 1.0 = full (2000 µs)
 *
 * The function clamps cmd to [0, 1] and writes the corresponding
 * CCR value. The preload ensures the new value takes effect at
 * the start of the next PWM period (no mid-pulse glitch).
 */
void motor_set(int index, float cmd)
{
    /* Clamp command to [0, 1] */
    if (cmd < 0.0f) cmd = 0.0f;
    if (cmd > 1.0f) cmd = 1.0f;

    /* Convert to µs: 1000 + cmd * 1000 → [1000, 2000] */
    uint32_t ccr = (uint32_t)(cmd * 1000.0f) + PWM_MIN;

    switch (index) {
        case 0: TIM8_CCR3 = ccr; break;   /* M1 → PC8  TIM8_CH3 */
        case 1: TIM8_CCR4 = ccr; break;   /* M2 → PC9  TIM8_CH4 */
        case 2: TIM1_CCR1 = ccr; break;   /* M3 → PA8  TIM1_CH1 */
        case 3: TIM1_CCR2 = ccr; break;   /* M4 → PA9  TIM1_CH2 */
        default: break;
    }
}


/* ============================================================
 * motor_all_stop — Emergency stop: all motors to idle
 * ============================================================ */
void motor_all_stop(void)
{
    TIM8_CCR3 = PWM_MIN;
    TIM8_CCR4 = PWM_MIN;
    TIM1_CCR1 = PWM_MIN;
    TIM1_CCR2 = PWM_MIN;
}

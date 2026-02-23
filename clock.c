/*
 * clock.c — Clock configuration for STM32F405
 *
 * Configures the chip to run at 168MHz from the external 8MHz crystal.
 *
 * Clock path:
 *   8MHz HSE → ÷8 (PLLM) → 1MHz → ×336 (PLLN) → 336MHz VCO
 *   → ÷2 (PLLP) → 168MHz SYSCLK
 *   → ÷7 (PLLQ) → 48MHz USB
 *
 * Bus speeds after configuration:
 *   AHB  = 168MHz (GPIO, DMA)
 *   APB1 = 42MHz  (UART2/3, SPI2/3, I2C1, TIM2-7)
 *   APB2 = 84MHz  (UART1, SPI1, TIM1/8, ADC)
 *
 * Timer clocks (important for PWM/DShot):
 *   APB1 timers = 84MHz  (APB1 × 2, because prescaler > 1)
 *   APB2 timers = 168MHz (APB2 × 2, because prescaler > 1)
 */

#include <stdint.h>

/* ============================================================
 * REGISTER DEFINITIONS
 * ============================================================
 *
 * All addresses from RM0090 (STM32F405 Reference Manual).
 *
 * RCC (Reset and Clock Control) base: 0x40023800
 * Flash interface base: 0x40023C00
 */

/* --- RCC Registers --- */

/* CR — Clock Control Register (offset 0x00)
 *
 * Controls which clock sources are on and reports their status.
 *
 * Key bits:
 *   Bit 0:  HSION   — HSI (16MHz internal) enable
 *   Bit 1:  HSIRDY  — HSI ready (read-only, hardware sets when stable)
 *   Bit 16: HSEON   — HSE (external crystal) enable
 *   Bit 17: HSERDY  — HSE ready (read-only, hardware sets when crystal is stable)
 *   Bit 24: PLLON   — PLL enable
 *   Bit 25: PLLRDY  — PLL ready (read-only, hardware sets when PLL locks)
 *
 * The RDY bits are crucial — you MUST wait for them after enabling
 * a clock source. Enabling HSE starts the crystal oscillator, but it
 * takes milliseconds to stabilize. Using it before HSERDY = crash.
 */
#define RCC_CR          (*(volatile uint32_t *)0x40023800)
#define RCC_CR_HSION    (1U << 0)
#define RCC_CR_HSIRDY   (1U << 1)
#define RCC_CR_HSEON    (1U << 16)
#define RCC_CR_HSERDY   (1U << 17)
#define RCC_CR_PLLON    (1U << 24)
#define RCC_CR_PLLRDY   (1U << 25)

/* PLLCFGR — PLL Configuration Register (offset 0x04)
 *
 * Sets the PLL multipliers and dividers. Must be configured
 * BEFORE enabling the PLL (writing PLLON).
 *
 * Bit layout:
 *   [5:0]   PLLM   — Input divider (2-63). VCO_input = HSE / PLLM
 *   [14:6]  PLLN   — VCO multiplier (50-432). VCO_output = VCO_input × PLLN
 *   [17:16] PLLP   — Output divider for SYSCLK. Encoded: 00=÷2, 01=÷4, 10=÷6, 11=÷8
 *   [22]    PLLSRC — PLL source: 0=HSI, 1=HSE
 *   [27:24] PLLQ   — Output divider for USB (2-15). USB_clk = VCO / PLLQ
 *
 * Constraints (from datasheet, violating these = undefined behavior):
 *   - VCO input (HSE/PLLM) should be 1-2MHz (ST recommends 1MHz)
 *   - VCO output must be 100-432MHz
 *   - SYSCLK (VCO/PLLP) must not exceed 168MHz
 *   - USB clock (VCO/PLLQ) must be exactly 48MHz
 *
 * Our values:
 *   PLLM = 8:   8MHz / 8 = 1MHz input         ✓ (in 1-2MHz range)
 *   PLLN = 336: 1MHz × 336 = 336MHz VCO       ✓ (in 100-432MHz range)
 *   PLLP = 2:   336MHz / 2 = 168MHz SYSCLK     ✓ (at max)
 *   PLLQ = 7:   336MHz / 7 = 48MHz USB         ✓ (exact)
 *   PLLSRC = 1: HSE as source
 */
#define RCC_PLLCFGR     (*(volatile uint32_t *)0x40023804)

/* CFGR — Clock Configuration Register (offset 0x08)
 *
 * Controls the system clock source selection and bus prescalers.
 *
 * Key fields:
 *   [1:0]   SW    — System clock switch: 00=HSI, 01=HSE, 10=PLL
 *   [3:2]   SWS   — System clock switch status (read-only, confirms which source is active)
 *   [7:4]   HPRE  — AHB prescaler: 0xxx=÷1, 1000=÷2, 1001=÷4, ... 1111=÷512
 *   [12:10] PPRE1 — APB1 prescaler: 0xx=÷1, 100=÷2, 101=÷4, 110=÷8, 111=÷16
 *   [15:13] PPRE2 — APB2 prescaler: same encoding as PPRE1
 *
 * Bus prescaler values we need:
 *   HPRE  = 0000 (÷1):  AHB  = 168MHz
 *   PPRE1 = 101  (÷4):  APB1 = 42MHz   (max 42MHz per datasheet)
 *   PPRE2 = 100  (÷2):  APB2 = 84MHz   (max 84MHz per datasheet)
 *
 * IMPORTANT: When APB prescaler > 1, the timer clocks on that bus
 * are automatically doubled by hardware. So:
 *   APB1 timers get 42MHz × 2 = 84MHz
 *   APB2 timers get 84MHz × 2 = 168MHz
 * This is a hardware feature, not software — it's wired in the clock tree.
 * You need to know this for correct PWM frequency calculations.
 */
#define RCC_CFGR        (*(volatile uint32_t *)0x40023808)
#define RCC_CFGR_SW_PLL     (2U << 0)   /* Select PLL as system clock */
#define RCC_CFGR_SWS_PLL    (2U << 2)   /* Status: PLL is system clock */
#define RCC_CFGR_HPRE_DIV1  (0U << 4)   /* AHB = SYSCLK / 1 */
#define RCC_CFGR_PPRE1_DIV4 (5U << 10)  /* APB1 = AHB / 4 */
#define RCC_CFGR_PPRE2_DIV2 (4U << 13)  /* APB2 = AHB / 2 */

/* AHB1ENR — AHB1 peripheral clock enable (offset 0x30)
   We defined this in main.c for GPIO. Including here for completeness. */
#define RCC_AHB1ENR     (*(volatile uint32_t *)0x40023830)

/* --- Flash Interface Registers ---
 * Base: 0x40023C00
 *
 * ACR — Flash Access Control Register (offset 0x00)
 *
 * Key fields:
 *   [3:0]  LATENCY — Number of wait states for flash access (0-15)
 *   [8]    PRFTEN  — Prefetch enable
 *   [9]    ICEN    — Instruction cache enable
 *   [10]   DCEN    — Data cache enable
 *
 * Wait states are determined by SYSCLK frequency and supply voltage:
 *   At 3.3V (2.7-3.6V range):
 *     0 WS: up to 30MHz
 *     1 WS: up to 60MHz
 *     2 WS: up to 90MHz
 *     3 WS: up to 120MHz
 *     4 WS: up to 150MHz
 *     5 WS: up to 168MHz  ← we need this
 *
 * THE MOST CRITICAL STEP: Flash wait states MUST be set BEFORE
 * increasing the clock speed. If you switch to 168MHz with 0 wait
 * states, the CPU reads corrupted instructions from flash and crashes
 * immediately. This is the #1 mistake in STM32 clock configuration.
 */
#define FLASH_ACR       (*(volatile uint32_t *)0x40023C00)
#define FLASH_ACR_LATENCY_5WS  (5U << 0)   /* 5 wait states */
#define FLASH_ACR_PRFTEN       (1U << 8)    /* Prefetch enable */
#define FLASH_ACR_ICEN         (1U << 9)    /* Instruction cache enable */
#define FLASH_ACR_DCEN         (1U << 10)   /* Data cache enable */


/* ============================================================
 * CLOCK INITIALIZATION FUNCTION
 * ============================================================
 *
 * Call this at the beginning of main(), before any peripheral
 * that depends on accurate timing (SPI, UART, timers, etc.).
 *
 * Returns 0 on success, -1 if HSE or PLL failed to start.
 * On failure, the chip remains on HSI at 16MHz — still functional,
 * just slow and with inaccurate timing.
 */
int clock_init(void)
{
    /* ==========================================================
     * Step 1: Enable the HSE (external crystal) and wait for it
     * ==========================================================
     *
     * The Sologood board has an 8MHz crystal connected to the
     * HSE pins (OSC_IN / OSC_OUT). Enabling HSEON starts the
     * crystal oscillator circuit.
     *
     * The crystal needs time to start oscillating and reach a
     * stable frequency — typically 2-20ms depending on the crystal.
     * The hardware sets HSERDY when it's stable. We must wait
     * for this flag before using HSE as a clock source.
     *
     * If the crystal is missing, damaged, or not connected, HSERDY
     * will never set and we'll time out. This is why we have a
     * timeout counter instead of waiting forever.
     */
    RCC_CR |= RCC_CR_HSEON;

    /* Wait for HSE to become ready.
     * Timeout prevents hanging if crystal is bad.
     * At 16MHz HSI, ~100000 iterations ≈ ~20ms — more than enough
     * for any crystal to start. */
    uint32_t timeout = 100000;
    while (!(RCC_CR & RCC_CR_HSERDY)) {
        if (--timeout == 0) {
            /* HSE failed to start. Crystal might be:
             * - Not populated on the board
             * - Damaged
             * - Wrong frequency (shouldn't prevent oscillation though)
             * We stay on HSI. Caller should handle this error. */
            return -1;
        }
    }

    /* ==========================================================
     * Step 2: Configure flash wait states and caching
     * ==========================================================
     *
     * THIS MUST HAPPEN BEFORE WE SPEED UP THE CLOCK.
     *
     * Think of it like this: flash memory is a slow library.
     * The CPU is a fast reader. At 16MHz (current speed), the reader
     * is slow enough to wait for the library. At 168MHz, the reader
     * is 10x faster and needs to explicitly wait 5 "ticks" for each
     * book (instruction) from the library.
     *
     * Setting extra wait states when still at 16MHz is harmless
     * (the CPU just waits a bit longer than necessary). But having
     * too few wait states at high speed causes the CPU to read
     * incomplete/corrupted data from flash = crash.
     *
     * The ART (Adaptive Real-Time) accelerator hides these wait
     * states through caching and prefetch:
     *   - Instruction cache: 64 lines × 128 bits. Holds recently
     *     executed code. Cache hit = 0 wait states.
     *   - Data cache: similar, for const data reads from flash.
     *   - Prefetch: fetches the next instruction while current one
     *     executes. Effective for sequential code.
     *
     * With ART enabled, sequential code runs as if there were 0 WS.
     * Branches/jumps still incur the full 5 WS penalty (cache miss).
     */
    FLASH_ACR = FLASH_ACR_LATENCY_5WS   /* 5 wait states for 168MHz */
              | FLASH_ACR_PRFTEN         /* Enable prefetch buffer */
              | FLASH_ACR_ICEN           /* Enable instruction cache */
              | FLASH_ACR_DCEN;          /* Enable data cache */

    /* Verify wait states were applied.
     * The LATENCY field should read back as 5. If flash interface
     * is not responding (clock not enabled — shouldn't happen since
     * it's always on), this could fail. */
    if ((FLASH_ACR & 0xF) != 5) {
        return -1;  /* Flash configuration failed */
    }

    /* ==========================================================
     * Step 3: Configure the PLL
     * ==========================================================
     *
     * The PLL must be OFF when we change its configuration.
     * (It should already be off since we haven't enabled it yet,
     * but being explicit is good practice.)
     *
     * We build the PLLCFGR value by shifting each field into
     * its bit position:
     *
     *   Bits [5:0]   = PLLM = 8     → HSE / 8 = 1MHz
     *   Bits [14:6]  = PLLN = 336   → 1MHz × 336 = 336MHz VCO
     *   Bits [17:16] = PLLP = 0     → ÷2 (encoding: 00=÷2, 01=÷4, 10=÷6, 11=÷8)
     *   Bit  [22]    = PLLSRC = 1   → Use HSE (not HSI)
     *   Bits [27:24] = PLLQ = 7     → 336MHz / 7 = 48MHz USB
     *
     * Note: PLLP encoding is confusing. The division factor is (PLLP_reg + 1) × 2.
     *   Register value 0 → divide by 2
     *   Register value 1 → divide by 4
     *   Register value 2 → divide by 6
     *   Register value 3 → divide by 8
     * We want ÷2, so register value = 0.
     */

    /* Make sure PLL is off before configuring */
    RCC_CR &= ~RCC_CR_PLLON;

    uint32_t pllcfgr = 0;
    pllcfgr |= 8U;              /* PLLM = 8 (bits [5:0]) */
    pllcfgr |= (336U << 6);     /* PLLN = 336 (bits [14:6]) */
    pllcfgr |= (0U << 16);      /* PLLP = 0 → divide by 2 (bits [17:16]) */
    pllcfgr |= (1U << 22);      /* PLLSRC = HSE (bit [22]) */
    pllcfgr |= (7U << 24);      /* PLLQ = 7 (bits [27:24]) */

    RCC_PLLCFGR = pllcfgr;

    /* ==========================================================
     * Step 4: Enable the PLL and wait for lock
     * ==========================================================
     *
     * The PLL takes time to "lock" — the feedback loop in the PLL
     * circuit needs to stabilize so the output frequency is accurate.
     * Typically takes ~1-2ms. Hardware sets PLLRDY when locked.
     *
     * Using the PLL before it's locked gives you an unstable,
     * wrong-frequency clock = everything downstream breaks.
     */
    RCC_CR |= RCC_CR_PLLON;

    timeout = 100000;
    while (!(RCC_CR & RCC_CR_PLLRDY)) {
        if (--timeout == 0) {
            /* PLL failed to lock. Possible causes:
             * - PLLCFGR values out of valid range
             * - HSE not actually stable (shouldn't happen, we checked)
             * - Hardware issue */
            return -1;
        }
    }

    /* ==========================================================
     * Step 5: Configure bus prescalers
     * ==========================================================
     *
     * Set the AHB, APB1, and APB2 dividers BEFORE switching to
     * the faster PLL clock. This way, the moment we switch, the
     * bus speeds are already correct. If we switched first and
     * then set prescalers, there'd be a brief moment where APB1
     * runs at 168MHz (exceeding its 42MHz max = potential damage
     * or undefined behavior).
     *
     * We do a read-modify-write to preserve any other bits in CFGR
     * that might be set (like MCO output configuration).
     */
    uint32_t cfgr = RCC_CFGR;

    /* Clear the fields we're about to set */
    cfgr &= ~((0xF << 4)     /* HPRE  bits [7:4] */
             | (0x7 << 10)    /* PPRE1 bits [12:10] */
             | (0x7 << 13));  /* PPRE2 bits [15:13] */

    /* Set our prescaler values */
    cfgr |= RCC_CFGR_HPRE_DIV1     /* AHB  = SYSCLK / 1 = 168MHz */
          |  RCC_CFGR_PPRE1_DIV4    /* APB1 = AHB / 4 = 42MHz */
          |  RCC_CFGR_PPRE2_DIV2;   /* APB2 = AHB / 2 = 84MHz */

    RCC_CFGR = cfgr;

    /* ==========================================================
     * Step 6: Switch system clock source to PLL
     * ==========================================================
     *
     * This is the actual switchover. The SW (System clock Switch)
     * field tells the clock mux which source to use:
     *   00 = HSI (current)
     *   01 = HSE
     *   10 = PLL (what we want)
     *
     * The SWS (Switch Status) field confirms which source is
     * actually active. There's a brief transition period — we
     * must wait for SWS to confirm the switch before proceeding.
     *
     * After this write, the CPU is running at 168MHz. Every
     * instruction takes ~6ns instead of ~62ns. The delay loop
     * in blinky will run ~10x faster — the LED will appear
     * to be always-on (blinking too fast to see) unless we
     * increase the delay count.
     */
    cfgr = RCC_CFGR;
    cfgr &= ~(0x3 << 0);          /* Clear SW bits [1:0] */
    cfgr |= RCC_CFGR_SW_PLL;      /* Select PLL as system clock */
    RCC_CFGR = cfgr;

    /* Wait for the switch to complete.
     * SWS should become 10 (PLL). */
    timeout = 100000;
    while ((RCC_CFGR & (0x3 << 2)) != RCC_CFGR_SWS_PLL) {
        if (--timeout == 0) {
            /* Clock switch failed. CPU is still on HSI.
             * This really shouldn't happen if PLL is locked. */
            return -1;
        }
    }

    /* ==========================================================
     * DONE — System is now running at:
     *   SYSCLK = 168MHz
     *   AHB    = 168MHz
     *   APB1   = 42MHz  (timers: 84MHz)
     *   APB2   = 84MHz  (timers: 168MHz)
     *   Flash  = 5 WS + ART accelerator enabled
     *
     * We can optionally disable HSI now to save a tiny bit of
     * power, since we're running entirely from HSE → PLL.
     * ========================================================== */
    RCC_CR &= ~RCC_CR_HSION;

    return 0;  /* Success */
}
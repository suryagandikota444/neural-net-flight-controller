/*
 * uart.c — USART2 driver for STM32F405 (bare-metal, blocking TX)
 *
 * ── Why USART2? ────────────────────────────────────────────────
 * The STM32F405 has three USART peripherals relevant to us:
 *   USART1 — APB2 bus (84MHz), TX=PA9,  RX=PA10
 *   USART2 — APB1 bus (42MHz), TX=PA2,  RX=PA3   ← we use this
 *   USART3 — APB1 bus (42MHz), TX=PB10, RX=PB11
 *
 * USART2's pins (PA2/PA3) are on the same port as the SPI1 CS pin
 * (PA4) and are typically broken out on the Sologood F405 UART pads.
 * The slower APB1 bus (42MHz vs 84MHz) makes no practical difference
 * for 115200 baud — the timing error is <0.2%.
 *
 * ── Baud rate math ─────────────────────────────────────────────
 * USART oversamples by 16 (default). The BRR register stores the
 * divisor as a 12.4 fixed-point number:
 *   USARTDIV      = fCK / (16 × baud)
 *   BRR[15:4]     = integer part of USARTDIV
 *   BRR[3:0]      = fractional part × 16
 *
 * Equivalently, the integer formula with rounding:
 *   BRR = (fCK + baud/2) / baud
 *
 * For 115200 baud, APB1 = 42MHz:
 *   BRR = (42,000,000 + 57,600) / 115,200 = 365 = 0x016D
 *   Mantissa = 22 (0x16), Fraction = 13 (0xD)
 *   Actual baud = 42,000,000 / 365 = 115,068 → 0.11% error ✓
 *
 * ── Blocking TX ────────────────────────────────────────────────
 * This driver busy-waits on TXE (Transmit data register Empty).
 * TXE goes high when the CPU-facing DR has been copied into the
 * hardware shift register — the byte is still clocking out, but
 * DR is ready for the next byte. This allows back-to-back bytes
 * with no gap. TC (Transmission Complete) goes high only after the
 * last stop bit — you only need TC before disabling the USART.
 *
 * Blocking TX is fine for debug prints. It will NOT be used in
 * the fast gyro loop — DMA-based TX comes in a later step.
 */

#include <stdint.h>
#include "uart.h"
#include "clock.h"  /* For APB1_HZ */

/* ============================================================
 * REGISTER DEFINITIONS
 * All addresses from RM0090 (STM32F405 Reference Manual).
 * ============================================================ */

/* --- RCC clock enables --- */
#define RCC_AHB1ENR          (*(volatile uint32_t *)0x40023830)
#define RCC_APB1ENR          (*(volatile uint32_t *)0x40023840)
#define RCC_AHB1ENR_GPIOAEN  (1U << 0)    /* GPIOA clock enable */
#define RCC_APB1ENR_USART2EN (1U << 17)   /* USART2 clock enable */

/* --- GPIOA registers (base 0x40020000) ---
 *
 * We configure two pins:
 *   PA2 = USART2_TX  (AF7, output, high-speed)
 *   PA3 = USART2_RX  (AF7, input,  pull-up)
 *
 * MODER:   2 bits per pin — 00=input, 01=output, 10=AF, 11=analog
 * OSPEEDR: 2 bits per pin — 00=low, 01=medium, 10=high, 11=very high
 * PUPDR:   2 bits per pin — 00=none, 01=pull-up, 10=pull-down
 * AFR[0]:  4 bits per pin for pins 0-7 — AF7 = USART1/2/3
 */
#define GPIOA_MODER    (*(volatile uint32_t *)0x40020000)
#define GPIOA_OSPEEDR  (*(volatile uint32_t *)0x40020008)
#define GPIOA_PUPDR    (*(volatile uint32_t *)0x4002000C)
#define GPIOA_AFR0     (*(volatile uint32_t *)0x40020020)  /* Pins 0-7 */

/* --- USART2 registers (base 0x40004400) ---
 *
 * SR  — Status Register (offset 0x00)
 *   Bit 6: TC   — Transmission Complete (all bits sent incl. stop bit)
 *   Bit 7: TXE  — TX data register Empty (DR ready for next write)
 *
 * DR  — Data Register (offset 0x04)
 *   Write to transmit; read to receive. Only [7:0] used for 8-bit frames.
 *
 * BRR — Baud Rate Register (offset 0x08)
 *   [15:4] DIV_Mantissa, [3:0] DIV_Fraction — see baud math above.
 *
 * CR1 — Control Register 1 (offset 0x0C)
 *   Bit 2:  RE  — Receiver Enable
 *   Bit 3:  TE  — Transmitter Enable (also sends idle frame on the line)
 *   Bit 13: UE  — USART Enable (must be set last, after BRR/CR1 config)
 */
#define USART2_SR    (*(volatile uint32_t *)0x40004400)
#define USART2_DR    (*(volatile uint32_t *)0x40004404)
#define USART2_BRR   (*(volatile uint32_t *)0x40004408)
#define USART2_CR1   (*(volatile uint32_t *)0x4000440C)

#define USART_SR_TXE    (1U << 7)
#define USART_SR_TC     (1U << 6)
#define USART_CR1_RE    (1U << 2)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_UE    (1U << 13)


/* ============================================================
 * uart2_init
 * ============================================================ */
void uart2_init(uint32_t baud)
{
    /* ── Step 1: Enable peripheral clocks ──────────────────────
     *
     * Peripherals are clock-gated by default (saves power).
     * Writing to GPIOA or USART2 registers before enabling their
     * clocks has no effect — the write is silently ignored.
     * Enable GPIOA on AHB1 and USART2 on APB1. */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC_APB1ENR |= RCC_APB1ENR_USART2EN;

    /* Read RCC_APB1ENR back — this forces the AHB bus to complete
     * the write before we proceed. Without this, a speculative
     * read of USART2_CR1 could arrive before the clock is enabled. */
    (void)RCC_APB1ENR;

    /* ── Step 2: Configure PA2 (TX) and PA3 (RX) ───────────────
     *
     * MODER: set both to Alternate Function mode (0b10).
     * Clear 2 bits, then set. PA2 → bits[5:4], PA3 → bits[7:6]. */
    GPIOA_MODER &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_MODER |=  ((2U << (2 * 2)) | (2U << (3 * 2)));

    /* OSPEEDR: high speed (0b10) on both pins.
     * "High speed" here means the GPIO output slew rate — it allows
     * the pin to toggle fast enough for UART at 115200+.
     * Without this, the rise/fall times are too slow and the
     * waveform gets rounded off at higher baud rates. */
    GPIOA_OSPEEDR &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_OSPEEDR |=  ((2U << (2 * 2)) | (2U << (3 * 2)));

    /* PUPDR: pull-up on PA3 (RX), no pull on PA2 (TX).
     * The RX line should be held high when idle (UART idles mark=high).
     * A floating RX pin with no pull-up will read noise as data.
     * TX is driven by the USART peripheral, so no pull needed. */
    GPIOA_PUPDR &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_PUPDR |=  (1U << (3 * 2));  /* PA3 = pull-up (0b01) */

    /* AFR0: assign AF7 (USART1/2/3) to PA2 and PA3.
     * AFR0 covers pins 0-7, 4 bits each.
     * PA2 → bits[11:8], PA3 → bits[15:12].
     * AF7 = 0b0111. */
    GPIOA_AFR0 &= ~((0xFU << (2 * 4)) | (0xFU << (3 * 4)));
    GPIOA_AFR0 |=  ((7U  << (2 * 4)) | (7U  << (3 * 4)));

    /* ── Step 3: Set baud rate ──────────────────────────────────
     *
     * BRR = round(fCK / baud) where fCK = APB1_HZ (42MHz).
     * The +baud/2 term rounds to nearest instead of truncating,
     * which halves the maximum baud rate error. */
    USART2_BRR = (APB1_HZ + (baud / 2U)) / baud;

    /* ── Step 4: Enable USART ───────────────────────────────────
     *
     * Set UE (USART Enable), TE (TX enable), RE (RX enable) together.
     * TE causes the USART to send an idle frame (all 1s for one
     * frame period) before the first real byte — this is normal
     * UART behaviour and the CP2102 will ignore it. */
    USART2_CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}


/* ============================================================
 * uart2_write_char — blocking single-byte TX
 * ============================================================ */
void uart2_write_char(char c)
{
    /* Spin until the TX data register is empty (previous byte has
     * been moved into the shift register and is clocking out). */
    while (!(USART2_SR & USART_SR_TXE));
    USART2_DR = (uint8_t)c;
}


/* ============================================================
 * uart2_write_string — transmit a null-terminated string
 * ============================================================ */
void uart2_write_string(const char *s)
{
    while (*s) {
        uart2_write_char(*s++);
    }
}


/* ============================================================
 * uart2_write_uint32 — print uint32 in decimal
 * ============================================================
 *
 * No printf, no libc. Build digits right-to-left in a 10-char
 * buffer (2^32 - 1 = 4,294,967,295 is 10 digits), then print
 * forward from the first populated slot.
 */
void uart2_write_uint32(uint32_t val)
{
    char buf[10];
    int  i = 9;

    if (val == 0) {
        uart2_write_char('0');
        return;
    }

    while (val > 0 && i >= 0) {
        buf[i--] = '0' + (val % 10);
        val /= 10;
    }

    while (++i < 10) {
        uart2_write_char(buf[i]);
    }
}


/* ============================================================
 * uart2_write_int32 — print int32 in decimal with sign
 * ============================================================
 *
 * Prints a leading '+' or '-' then the absolute value.
 * Handles INT32_MIN correctly by casting through uint32_t.
 */
void uart2_write_int32(int32_t val)
{
    if (val < 0) {
        uart2_write_char('-');
        uart2_write_uint32((uint32_t)(-(val + 1)) + 1U);
    } else {
        uart2_write_char('+');
        uart2_write_uint32((uint32_t)val);
    }
}


/* ============================================================
 * uart2_write_hex8 — print uint8_t as "0xXX"
 * ============================================================
 *
 * Always exactly 4 characters. Useful for WHO_AM_I values,
 * raw register dumps, anything where decimal is misleading.
 */
void uart2_write_hex8(uint8_t val)
{
    const char hex[] = "0123456789ABCDEF";
    uart2_write_char('0');
    uart2_write_char('x');
    uart2_write_char(hex[(val >> 4) & 0xF]);
    uart2_write_char(hex[val & 0xF]);
}

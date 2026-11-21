/*
 * uart.c — configurable USART driver for STM32F405 (bare-metal, blocking TX)
 *
 * Set UART_PERIPH in uart.h to choose the peripheral:
 *   2 → USART2  TX=PA2   RX=PA3   (APB1, 42 MHz)
 *   3 → USART3  TX=PB10  RX=PB11  (APB1, 42 MHz)
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
 * REGISTER DEFINITIONS  (selected by UART_PERIPH in uart.h)
 * All addresses from RM0090 (STM32F405 Reference Manual).
 * ============================================================ */

/* --- RCC clock enables --- */
#define RCC_AHB1ENR    (*(volatile uint32_t *)0x40023830)
#define RCC_APB1ENR    (*(volatile uint32_t *)0x40023840)

/* ── Per-peripheral configuration ──────────────────────────── */
#if UART_PERIPH == 2
/*  USART2 — TX=PA2, RX=PA3, GPIOA, APB1 bit 17, base 0x40004400 */
#define UART_RCC_GPIO_EN     (1U << 0)       /* GPIOA on AHB1       */
#define UART_RCC_USART_EN    (1U << 17)      /* USART2 on APB1      */
#define UART_GPIO_BASE       0x40020000U     /* GPIOA               */
#define UART_TX_PIN          2
#define UART_RX_PIN          3
#define UART_AF              7               /* AF7 for USART2      */
#define UART_AFR_IDX         0               /* AFR[0] (pins 0-7)   */
#define USART_BASE           0x40004400U

#elif UART_PERIPH == 3
/*  USART3 — TX=PB10, RX=PB11, GPIOB, APB1 bit 18, base 0x40004800 */
#define UART_RCC_GPIO_EN     (1U << 1)       /* GPIOB on AHB1       */
#define UART_RCC_USART_EN    (1U << 18)      /* USART3 on APB1      */
#define UART_GPIO_BASE       0x40020400U     /* GPIOB               */
#define UART_TX_PIN          10
#define UART_RX_PIN          11
#define UART_AF              7               /* AF7 for USART3      */
#define UART_AFR_IDX         1               /* AFR[1] (pins 8-15)  */
#define USART_BASE           0x40004800U

#else
#error "UART_PERIPH must be 2 or 3"
#endif

/* --- GPIO registers (generic, computed from UART_GPIO_BASE) --- */
#define GPIO_MODER     (*(volatile uint32_t *)(UART_GPIO_BASE + 0x00))
#define GPIO_OSPEEDR   (*(volatile uint32_t *)(UART_GPIO_BASE + 0x08))
#define GPIO_PUPDR     (*(volatile uint32_t *)(UART_GPIO_BASE + 0x0C))
#define GPIO_AFR       (*(volatile uint32_t *)(UART_GPIO_BASE + 0x20 + 4*UART_AFR_IDX))

/* Pin number within the AFR register (pins 0-7 in AFR[0], 8-15 → subtract 8 in AFR[1]) */
#define UART_TX_AFR_PIN  (UART_TX_PIN - 8 * UART_AFR_IDX)
#define UART_RX_AFR_PIN  (UART_RX_PIN - 8 * UART_AFR_IDX)

/* --- USART registers (generic, computed from USART_BASE) --- */
#define USART_SR     (*(volatile uint32_t *)(USART_BASE + 0x00))
#define USART_DR     (*(volatile uint32_t *)(USART_BASE + 0x04))
#define USART_BRR    (*(volatile uint32_t *)(USART_BASE + 0x08))
#define USART_CR1    (*(volatile uint32_t *)(USART_BASE + 0x0C))

#define USART_SR_TXE    (1U << 7)
#define USART_SR_TC     (1U << 6)
#define USART_CR1_RE    (1U << 2)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_UE    (1U << 13)


/* ============================================================
 * uart_init
 * ============================================================ */
void uart_init(uint32_t baud)
{
    /* ── Step 1: Enable peripheral clocks ──────────────────────
     *
     * Peripherals are clock-gated by default (saves power).
     * Enable the GPIO port on AHB1 and the USART on APB1. */
    RCC_AHB1ENR |= UART_RCC_GPIO_EN;
    RCC_APB1ENR |= UART_RCC_USART_EN;

    /* Read RCC_APB1ENR back — this forces the AHB bus to complete
     * the write before we proceed. Without this, a speculative
     * read of USART registers could arrive before the clock is enabled. */
    (void)RCC_APB1ENR;

    /* ── Step 2: Configure TX and RX pins ──────────────────────
     *
     * MODER: set both to Alternate Function mode (0b10). */
    GPIO_MODER &= ~((3U << (UART_TX_PIN * 2)) | (3U << (UART_RX_PIN * 2)));
    GPIO_MODER |=  ((2U << (UART_TX_PIN * 2)) | (2U << (UART_RX_PIN * 2)));

    /* OSPEEDR: high speed (0b10) on both pins. */
    GPIO_OSPEEDR &= ~((3U << (UART_TX_PIN * 2)) | (3U << (UART_RX_PIN * 2)));
    GPIO_OSPEEDR |=  ((2U << (UART_TX_PIN * 2)) | (2U << (UART_RX_PIN * 2)));

    /* PUPDR: pull-up on RX, no pull on TX. */
    GPIO_PUPDR &= ~((3U << (UART_TX_PIN * 2)) | (3U << (UART_RX_PIN * 2)));
    GPIO_PUPDR |=  (1U << (UART_RX_PIN * 2));

    /* AFR: assign the selected AF to TX and RX pins. */
    GPIO_AFR &= ~((0xFU << (UART_TX_AFR_PIN * 4)) | (0xFU << (UART_RX_AFR_PIN * 4)));
    GPIO_AFR |=  (((uint32_t)UART_AF << (UART_TX_AFR_PIN * 4)) |
                  ((uint32_t)UART_AF << (UART_RX_AFR_PIN * 4)));

    /* ── Step 3: Set baud rate ──────────────────────────────────
     *
     * BRR = round(fCK / baud) where fCK = APB1_HZ (42MHz). */
    USART_BRR = (APB1_HZ + (baud / 2U)) / baud;

    /* ── Step 4: Enable USART ───────────────────────────────────
     *
     * Set UE (USART Enable), TE (TX enable), RE (RX enable). */
    USART_CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}


/* ============================================================
 * uart_write_char — blocking single-byte TX
 * ============================================================ */
void uart_write_char(char c)
{
    /* Spin until the TX data register is empty (previous byte has
     * been moved into the shift register and is clocking out). */
    while (!(USART_SR & USART_SR_TXE));
    USART_DR = (uint8_t)c;
}


/* ============================================================
 * uart_write_string — transmit a null-terminated string
 * ============================================================ */
void uart_write_string(const char *s)
{
    while (*s) {
        uart_write_char(*s++);
    }
}


/* ============================================================
 * uart_write_uint32 — print uint32 in decimal
 * ============================================================
 *
 * No printf, no libc. Build digits right-to-left in a 10-char
 * buffer (2^32 - 1 = 4,294,967,295 is 10 digits), then print
 * forward from the first populated slot.
 */
void uart_write_uint32(uint32_t val)
{
    char buf[10];
    int  i = 9;

    if (val == 0) {
        uart_write_char('0');
        return;
    }

    while (val > 0 && i >= 0) {
        buf[i--] = '0' + (val % 10);
        val /= 10;
    }

    while (++i < 10) {
        uart_write_char(buf[i]);
    }
}


/* ============================================================
 * uart_write_int32 — print int32 in decimal with sign
 * ============================================================
 *
 * Prints a leading '+' or '-' then the absolute value.
 * Handles INT32_MIN correctly by casting through uint32_t.
 */
void uart_write_int32(int32_t val)
{
    if (val < 0) {
        uart_write_char('-');
        uart_write_uint32((uint32_t)(-(val + 1)) + 1U);
    } else {
        uart_write_char('+');
        uart_write_uint32((uint32_t)val);
    }
}


/* ============================================================
 * uart_write_hex8 — print uint8_t as "0xXX"
 * ============================================================
 *
 * Always exactly 4 characters. Useful for WHO_AM_I values,
 * raw register dumps, anything where decimal is misleading.
 */
void uart_write_hex8(uint8_t val)
{
    const char hex[] = "0123456789ABCDEF";
    uart_write_char('0');
    uart_write_char('x');
    uart_write_char(hex[(val >> 4) & 0xF]);
    uart_write_char(hex[val & 0xF]);
}

/*
 * rc_input.c — USART2 receive driver for the ExpressLRS receiver (CRSF).
 *
 *   PA3 = USART2_RX  ← receiver TX   (AF7)
 *   PA2 = USART2_TX  → receiver RX   (AF7; configured, not used yet —
 *                                      reserved for telemetry back to the radio)
 *   420 000 baud, 8N1. APB1 = 42 MHz → BRR = 42 000 000 / 420 000 = 100
 *   exactly (0 % baud error).
 *
 * Bytes arrive at up to ~42 per ms. The ISR stores them in a 512-byte
 * ring (≈5 ms of continuous data; one 10 ms tick of a 500 Hz ELRS link is
 * ~130 bytes) and the main loop drains it every tick.
 */
#include "rc_input.h"
#include "clock.h"     /* APB1_HZ */
#include "fc_config.h" /* RC_BAUD */
#include "uart.h"      /* UART_PERIPH: debug UART must not also be USART2 */

#if UART_PERIPH == 2
#error "USART2 is the RC receiver port; set UART_PERIPH to 3 in uart.h"
#endif

#define RCC_AHB1ENR     (*(volatile uint32_t *)0x40023830)
#define RCC_APB1ENR     (*(volatile uint32_t *)0x40023840)

#define GPIOA_BASE      0x40020000U
#define GPIOA_MODER     (*(volatile uint32_t *)(GPIOA_BASE + 0x00))
#define GPIOA_OSPEEDR   (*(volatile uint32_t *)(GPIOA_BASE + 0x08))
#define GPIOA_PUPDR     (*(volatile uint32_t *)(GPIOA_BASE + 0x0C))
#define GPIOA_AFRL      (*(volatile uint32_t *)(GPIOA_BASE + 0x20))

#define USART2_BASE     0x40004400U
#define USART2_SR       (*(volatile uint32_t *)(USART2_BASE + 0x00))
#define USART2_DR       (*(volatile uint32_t *)(USART2_BASE + 0x04))
#define USART2_BRR      (*(volatile uint32_t *)(USART2_BASE + 0x08))
#define USART2_CR1      (*(volatile uint32_t *)(USART2_BASE + 0x0C))
#define USART2_CR2      (*(volatile uint32_t *)(USART2_BASE + 0x10))
#define USART2_CR3      (*(volatile uint32_t *)(USART2_BASE + 0x14))

#define SR_ORE          (1U << 3)
#define SR_RXNE         (1U << 5)
#define CR1_RE          (1U << 2)
#define CR1_TE          (1U << 3)
#define CR1_RXNEIE      (1U << 5)
#define CR1_UE          (1U << 13)

#define USART2_IRQN     38
#define NVIC_ISER(n)    (*(volatile uint32_t *)(0xE000E100U + 4U * (n)))

#define RX_BUF_SIZE     512U                 /* power of two */
#define RX_BUF_MASK     (RX_BUF_SIZE - 1U)

static volatile uint8_t  rx_buf[RX_BUF_SIZE];
static volatile uint32_t rx_head;            /* written by ISR only  */
static volatile uint32_t rx_tail;            /* written by main only */
static volatile uint32_t rx_overruns;
static volatile uint32_t rx_bytes;      /* every byte the USART received (diagnostics) */

void USART2_IRQHandler(void)
{
    uint32_t sr = USART2_SR;
    if (sr & (SR_RXNE | SR_ORE)) {
        /* Reading DR after SR clears RXNE and the overrun flag. */
        uint8_t b = (uint8_t)USART2_DR;
        rx_bytes++;
        if (sr & SR_ORE) rx_overruns++;
        uint32_t next = (rx_head + 1U) & RX_BUF_MASK;
        if (next != rx_tail) {
            rx_buf[rx_head] = b;
            rx_head = next;
        } else {
            rx_overruns++;                   /* main loop fell behind */
        }
    }
}

void rc_input_init(void)
{
    RCC_AHB1ENR |= (1U << 0);                /* GPIOA */
    RCC_APB1ENR |= (1U << 17);               /* USART2 */
    (void)RCC_APB1ENR;

    /* PA2, PA3 → alternate function 7, high speed, pull-up on RX */
    GPIOA_MODER   &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_MODER   |=  ((2U << (2 * 2)) | (2U << (3 * 2)));
    GPIOA_OSPEEDR &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_OSPEEDR |=  ((2U << (2 * 2)) | (2U << (3 * 2)));
    GPIOA_PUPDR   &= ~((3U << (2 * 2)) | (3U << (3 * 2)));
    GPIOA_PUPDR   |=  (1U << (3 * 2));
    GPIOA_AFRL    &= ~((0xFU << (2 * 4)) | (0xFU << (3 * 4)));
    GPIOA_AFRL    |=  ((7U << (2 * 4)) | (7U << (3 * 4)));

    USART2_CR1 = 0;
    USART2_CR2 = 0;                          /* 1 stop bit */
    USART2_CR3 = 0;
    USART2_BRR = (APB1_HZ + RC_BAUD / 2U) / RC_BAUD;
    rx_head = rx_tail = 0;
    rx_overruns = 0;
    rx_bytes = 0;
    USART2_CR1 = CR1_UE | CR1_RE | CR1_TE | CR1_RXNEIE;
    NVIC_ISER(USART2_IRQN / 32U) = 1U << (USART2_IRQN % 32U);
}

int rc_input_poll(crsf_t *c)
{
    int frames = 0;
    uint32_t head = rx_head;                 /* snapshot */
    while (rx_tail != head) {
        uint8_t b = rx_buf[rx_tail];
        rx_tail = (rx_tail + 1U) & RX_BUF_MASK;
        if (crsf_feed(c, b) == 1) frames++;
    }
    return frames;
}

uint32_t rc_input_overruns(void)
{
    return rx_overruns;
}

uint32_t rc_input_bytes(void)
{
    return rx_bytes;
}

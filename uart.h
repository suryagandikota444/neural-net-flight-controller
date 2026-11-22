/*
 * uart.h — USART driver for STM32F405
 *
 * Change UART_PERIPH to select which USART to use:
 *   2 → USART2: TX=PA2,  RX=PA3   (APB1, 42 MHz)
 *   3 → USART3: TX=PB10, RX=PB11  (APB1, 42 MHz)
 *
 * Call clock_init() before uart_init() or the baud rate will be wrong.
 *
 * On your Mac: screen /dev/cu.SLAB_USBtoUART 115200
 */

#ifndef UART_H
#define UART_H

#include <stdint.h>

/* ── Which USART peripheral to use (2 or 3) ──────────────── */
#define UART_PERIPH  3

/* Initialize the selected USART.
 * Configures TX/RX pins as AF7, sets baud rate.
 * Typical use: uart_init(115200) */
void uart_init(uint32_t baud);

/* Queue a single character for background (interrupt-driven) transmission.
 * Returns immediately unless the 1 KB transmit buffer is full, in which
 * case it waits for space — so no output is ever lost. */
void uart_write_char(char c);

/* Queue a null-terminated string (same rules as uart_write_char). */
void uart_write_string(const char *s);

/* Bytes that can be queued right now without waiting. Real-time code
 * (the flight loop) checks this before printing and skips the message
 * if it would not fit, so it never blocks. */
uint32_t uart_tx_free(void);

/* Wait until every queued byte has been transmitted (e.g. before a
 * reset or before a timing measurement). */
void uart_tx_flush(void);

/* Print an unsigned 32-bit integer in decimal (no leading zeros). */
void uart_write_uint32(uint32_t val);

/* Print a uint8_t as "0xXX" hex — useful for WHO_AM_I / register dumps. */
void uart_write_hex8(uint8_t val);

/* Print a signed 32-bit integer in decimal with leading sign (+/-). */
void uart_write_int32(int32_t val);

#endif /* UART_H */

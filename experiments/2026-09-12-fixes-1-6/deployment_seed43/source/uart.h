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

/* Transmit a single character (blocking — spins until TX register is empty). */
void uart_write_char(char c);

/* Transmit a null-terminated string (blocking). */
void uart_write_string(const char *s);

/* Print an unsigned 32-bit integer in decimal (no leading zeros). */
void uart_write_uint32(uint32_t val);

/* Print a uint8_t as "0xXX" hex — useful for WHO_AM_I / register dumps. */
void uart_write_hex8(uint8_t val);

/* Print a signed 32-bit integer in decimal with leading sign (+/-). */
void uart_write_int32(int32_t val);

#endif /* UART_H */

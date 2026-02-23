/*
 * uart.h — USART2 driver for STM32F405
 *
 * USART2 pinout (Sologood F405 header pins):
 *   PA2 = TX  →  connect to CP2102 RXD
 *   PA3 = RX  →  connect to CP2102 TXD
 *
 * USART2 is on the APB1 bus (42MHz after clock_init).
 * Call clock_init() before uart2_init() or the baud rate will be wrong.
 *
 * On your Mac: screen /dev/cu.SLAB_USBtoUART 115200
 */

#ifndef UART_H
#define UART_H

#include <stdint.h>

/* Initialize USART2.
 * Configures PA2 (TX) and PA3 (RX) as AF7, sets baud rate.
 * Typical use: uart2_init(115200) */
void uart2_init(uint32_t baud);

/* Transmit a single character (blocking — spins until TX register is empty). */
void uart2_write_char(char c);

/* Transmit a null-terminated string (blocking). */
void uart2_write_string(const char *s);

/* Print an unsigned 32-bit integer in decimal (no leading zeros). */
void uart2_write_uint32(uint32_t val);

/* Print a uint8_t as "0xXX" hex — useful for WHO_AM_I / register dumps. */
void uart2_write_hex8(uint8_t val);

/* Print a signed 32-bit integer in decimal with leading sign (+/-). */
void uart2_write_int32(int32_t val);

#endif /* UART_H */

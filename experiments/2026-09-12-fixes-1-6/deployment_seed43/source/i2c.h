/*
 * i2c.h — I2C1 driver for STM32F405 (bare-metal, polling)
 *
 * Pinout (common for F405 flight controllers):
 *   PB8 = SCL  (AF4, I2C1)
 *   PB9 = SDA  (AF4, I2C1)
 *
 * I2C1 is on the APB1 bus (42 MHz after clock_init).
 * Configured for Fast Mode (400 kHz).
 *
 * Usage:
 *   i2c1_init();
 *   uint8_t id;
 *   i2c1_read_reg(0x76, 0xD0, &id);   // read BMP280 chip ID
 */

#ifndef I2C_H
#define I2C_H

#include <stdint.h>

/* Initialize I2C1 in Fast Mode (400 kHz) on PB8 (SCL) / PB9 (SDA).
 * Call clock_init() first. */
void i2c1_init(void);

/* Write one byte to a register.
 * Returns 0 on success, -1 on timeout/error. */
int i2c1_write_reg(uint8_t dev_addr, uint8_t reg, uint8_t val);

/* Read one byte from a register.
 * Returns 0 on success, -1 on timeout/error. */
int i2c1_read_reg(uint8_t dev_addr, uint8_t reg, uint8_t *val);

/* Burst-read 'len' bytes starting from 'reg'.
 * Returns 0 on success, -1 on timeout/error. */
int i2c1_read_burst(uint8_t dev_addr, uint8_t reg, uint8_t *buf, int len);

/* Probe an I2C address. Returns 0 if device ACKs, -1 if NACK/timeout.
 * Useful for bus scanning to find connected devices. */
int i2c1_probe(uint8_t dev_addr);

#endif /* I2C_H */

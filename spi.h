/*
 * spi.h — SPI1 driver for IMU communication
 *
 * SPI1 pinout on Sologood F405:
 *   SCK  = PA5
 *   MISO = PA6
 *   MOSI = PA7
 *   CS   = PA4 (gyro chip select)
 */

#ifndef SPI_H
#define SPI_H

#include <stdint.h>

/* Initialize SPI1 peripheral and GPIO pins.
 * Configures: PA4=CS output, PA5/PA6/PA7=AF5 (SPI1)
 * SPI Mode 3, master, 5.25MHz initial clock speed. */
void spi1_init(void);

/* Read a single register from the IMU.
 * Sends (0x80 | reg) then clocks out one byte. */
uint8_t spi1_read_register(uint8_t reg);

/* Write a single register on the IMU.
 * Sends (reg & 0x7F) then the value byte. */
void spi1_write_register(uint8_t reg, uint8_t value);

/* Burst read: read 'count' consecutive registers starting at 'start_reg'.
 * Data is stored in 'buffer'. Used for fast gyro/accel data reads. */
void spi1_read_registers(uint8_t start_reg, uint8_t *buffer, uint8_t count);

#endif /* SPI_H */
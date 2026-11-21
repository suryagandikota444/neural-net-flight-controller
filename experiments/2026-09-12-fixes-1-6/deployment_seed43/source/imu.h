/*
 * imu.h — IMU detection and interface
 */

#ifndef IMU_H
#define IMU_H

#include <stdint.h>

/* Supported IMU types on the Sologood F405 */
typedef enum {
    IMU_NONE = 0,      /* Detection failed */
    IMU_MPU6000,       /* InvenSense MPU6000 (WHO_AM_I = 0x68) */
    IMU_ICM42688P,     /* InvenSense ICM-42688-P (WHO_AM_I = 0x47) */
    IMU_BMI270         /* Bosch BMI270 (CHIP_ID = 0x24) */
} imu_type_t;

/* Detect which IMU is present on SPI1.
 * Reads WHO_AM_I / CHIP_ID registers and identifies the chip.
 * If 'whoami' is not NULL, stores the raw register value there.
 * Returns the detected IMU type. */
imu_type_t imu_detect(uint8_t *whoami);

/* Raw sensor counts straight off the ADC — no scaling, no filtering.
 * Units depend on chip config (set in imu_init):
 *   accel: ±4g  → ±8192 LSB/g  (MPU6000 / ICM42688P at ±4g / ±16g)
 *   gyro:  ±2000 dps → ±16.4 LSB/(°/s)
 * For BMI270: accel ±8g → 4096 LSB/g, gyro ±2000 dps → 16.4 LSB/(°/s) */
typedef struct {
    int16_t ax, ay, az;   /* Accelerometer X, Y, Z */
    int16_t gx, gy, gz;   /* Gyroscope     X, Y, Z */
    int16_t temp;         /* Raw temperature (chip-specific formula) */
} imu_raw_t;

/* Configure the detected IMU for normal streaming operation.
 * Must be called after imu_detect() succeeds and before imu_read_raw().
 * Sets output data rate, full-scale range, and wakes the chip from sleep. */
void imu_init(imu_type_t imu);

/* Burst-read one set of raw accel + gyro + temp samples.
 * Results are stored in *out. Returns 0 on success, -1 on failure. */
int imu_read_raw(imu_type_t imu, imu_raw_t *out);

#endif /* IMU_H */
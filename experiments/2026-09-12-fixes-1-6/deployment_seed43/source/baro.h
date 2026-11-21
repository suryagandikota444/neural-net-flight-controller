/*
 * baro.h — Barometer driver (BMP280 / DPS310) for altitude estimation
 *
 * Provides:
 *   - Altitude relative to a reference point (metres, positive = up)
 *   - Vertical velocity (m/s, positive = up)
 *
 * These feed directly into the NN observation vector:
 *   obs[8]  = altitude_error / ALT_SCALE     (ALT_SCALE = 2.0)
 *   obs[9]  = vertical_vel  / VVEL_SCALE     (VVEL_SCALE = 5.0)
 *
 * Hardware: BMP280 or DPS310/SPL06 on I2C1 (address 0x76 or 0x77).
 *
 * Usage:
 *   baro_init();                    // detect + configure barometer
 *   baro_set_ground_reference();    // capture current pressure as "ground"
 *   // In control loop (call every tick, internally decimates to ~25 Hz):
 *   baro_update(0.01f);            // dt = 10 ms
 *   float alt  = baro_get_altitude();   // metres above ground ref
 *   float vvel = baro_get_vvel();       // m/s vertical velocity
 */

#ifndef BARO_H
#define BARO_H

#include <stdint.h>

/* Detection result */
typedef enum {
    BARO_NONE = 0,
    BARO_BMP280,
    BARO_DPS310,    /* Also covers SPL06-001 (same register set) */
} baro_type_t;

/* Detect and initialize the barometer.
 * Tries DPS310 first (more common on budget FCs), then BMP280.
 * Stores the chip ID in *chip_id if not NULL. */
baro_type_t baro_init(uint8_t *chip_id);

/* Capture current altitude as the ground reference (call once on
 * the ground, after barometer has been running for a few seconds
 * to let the IIR filter settle). */
void baro_set_ground_reference(void);

/* Read the barometer and update internal state.
 * Call every control loop tick — internally decimates to ~25 Hz.
 * dt = time since last call (seconds), typically 0.01 for 100 Hz.
 * Returns 0 on success, -1 if no barometer detected. */
int baro_update(float dt);

/* Get altitude above ground reference in metres (positive = up).
 * Valid after baro_set_ground_reference() has been called. */
float baro_get_altitude(void);

/* Get vertical velocity in m/s (positive = going up).
 * Computed by filtered differentiation of altitude. */
float baro_get_vvel(void);

/* Get raw pressure in Pa (useful for debugging). */
float baro_get_pressure(void);

/* Get temperature in °C (useful for debugging). */
float baro_get_temperature(void);

#endif /* BARO_H */

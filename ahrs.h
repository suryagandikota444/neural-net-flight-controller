/*
 * ahrs.h — Attitude and Heading Reference System (Madgwick filter)
 *
 * Fuses gyroscope and accelerometer data to estimate the drone's
 * orientation (roll, pitch, yaw) in space.
 *
 * Why we need this:
 *   The gyroscope measures rotation RATE (degrees/second), not angle.
 *   If you just integrate gyro rate to get angle, errors accumulate
 *   (drift) — after a few minutes, your "level" drifts by tens of
 *   degrees even though the drone hasn't moved.
 *
 *   The accelerometer can measure the gravity direction, which tells
 *   you the absolute tilt angle. But it's noisy and gets confused
 *   by linear acceleration during flight.
 *
 *   The Madgwick filter combines both:
 *     - Gyro provides fast, smooth angle updates (short-term accurate)
 *     - Accel corrects the drift (long-term accurate)
 *
 *   Think of it like a GPS + speedometer: the speedometer (gyro) tells
 *   you exactly how fast you're turning right now, but accumulates
 *   error over distance. The GPS (accel) is noisy but doesn't drift.
 *   Fusing them gives you the best of both.
 *
 * Usage:
 *   ahrs_init();                     // once, at boot
 *   ahrs_update(gx, gy, gz,         // called every loop iteration
 *               ax, ay, az, dt);     // gyro in rad/s, accel in g
 *   float roll  = ahrs_get_roll();   // returns degrees
 *   float pitch = ahrs_get_pitch();
 */

#ifndef AHRS_H
#define AHRS_H

/* Initialize the filter — sets quaternion to identity (flat, nose forward).
 * Call once before the main loop starts. */
void ahrs_init(void);

/* Run one filter iteration.
 *
 *   gx, gy, gz  — gyroscope angular rates in rad/s (body frame)
 *   ax, ay, az  — accelerometer readings in g (body frame)
 *                 (the filter normalizes these internally, so the
 *                  exact magnitude doesn't matter — only the direction)
 *   dt          — timestep in seconds (e.g. 0.01 for 100 Hz)
 */
void ahrs_update(float gx, float gy, float gz,
                 float ax, float ay, float az,
                 float dt);

/* Get current roll angle in degrees.
 * Positive = right wing down. Range: -180 to +180. */
float ahrs_get_roll(void);

/* Get current pitch angle in degrees.
 * Positive = nose up. Range: -90 to +90. */
float ahrs_get_pitch(void);

/* Get current yaw angle in degrees.
 * Positive = clockwise from above. Range: -180 to +180.
 * NOTE: Yaw drifts without a magnetometer — only trust gyro-integrated yaw
 * for short periods. For hover, yaw accuracy is not critical. */
float ahrs_get_yaw(void);

#endif /* AHRS_H */

/*
 * ahrs.c — Madgwick AHRS filter for STM32F405 (bare-metal)
 *
 * ── What Is the Madgwick Filter? ──────────────────────────────
 *
 * Sebastian Madgwick published this algorithm in 2011 as a
 * computationally efficient alternative to the Kalman filter
 * for orientation estimation. Betaflight, INAV, PX4, and most
 * open-source flight controllers use some variant of it.
 *
 * The core idea in plain English:
 *
 *   1. The filter maintains a QUATERNION — a 4-number representation
 *      of orientation [q0, q1, q2, q3]. Think of it as a compact
 *      way to describe "I'm tilted 10° right and 5° forward."
 *
 *   2. Each loop iteration:
 *      a) PREDICT: Use the gyroscope rate to update the quaternion.
 *         "I was tilted X°. The gyro says I'm rotating at Y°/s.
 *          So now I must be tilted X + Y*dt degrees."
 *
 *      b) CORRECT: Use the accelerometer to fix gyro drift.
 *         "Gravity should point straight down. The accelerometer
 *          says gravity is coming from *this* direction. If my
 *          predicted orientation disagrees with that, nudge the
 *          quaternion toward the accelerometer's answer."
 *
 *   3. The 'beta' parameter controls how much we trust the accel:
 *      - Small beta (0.01): Trust gyro more. Smooth but drifts slowly.
 *      - Large beta (0.1):  Trust accel more. No drift but noisier.
 *      - Good starting value: 0.04–0.06 for a quadrotor.
 *
 *
 * ── Why Quaternions Instead of Euler Angles? ──────────────────
 *
 * Euler angles (roll/pitch/yaw) have a problem called "gimbal lock"
 * where two axes align and you lose a degree of freedom. Quaternions
 * don't have this problem. Also, quaternion math is just multiply
 * and add — no trig functions needed for the filter itself.
 *
 * We only convert to Euler angles at the END, when we need roll/pitch
 * in degrees for the NN observation vector. The filter runs entirely
 * in quaternion space.
 *
 *
 * ── Math Functions ────────────────────────────────────────────
 *
 * Since we compile with -nostdlib (no standard C library), we
 * implement sqrtf, atan2f, and asinf ourselves:
 *
 *   sqrtf:  Use the Cortex-M4 hardware FPU VSQRT.F32 instruction.
 *           Single-cycle square root. Free!
 *
 *   atan2f: Polynomial approximation, valid for all quadrants.
 *           Max error ~0.22° — plenty accurate for flight control.
 *
 *   asinf:  Derived from atan2f using the identity
 *           asin(x) = atan2(x, sqrt(1-x²)).
 *
 *
 * ── Reference ─────────────────────────────────────────────────
 * S. Madgwick, "An efficient orientation filter for inertial and
 * magnetic measurement units," 2011.
 */

#include <stdint.h>
#include "ahrs.h"

/* ============================================================
 * FILTER TUNING
 * ============================================================ */

/* Beta — filter gain. Controls how aggressively the accelerometer
 * corrects the gyroscope drift. Higher = faster correction, more
 * noise. MWC Lower = smoother, more drift.
 *
 * Typical values:
 *   0.033 — very smooth, slow convergence (Madgwick's default)
 *   0.05  — good balance for multirotor hover
 *   0.1   — aggressive correction, use if attitude is sluggish
 *
 * This should be tuned on the actual drone. Start at 0.05, increase
 * if the attitude response feels laggy, decrease if it's jittery. */
#define AHRS_BETA   0.05f

/* Degrees-to-radians and radians-to-degrees */
#define DEG_TO_RAD  0.01745329f    /* π / 180 */
#define RAD_TO_DEG  57.2957795f    /* 180 / π */


/* ============================================================
 * MATH UTILITIES — no libm needed
 * ============================================================ */

/* Hardware square root — the Cortex-M4 FPU has a dedicated VSQRT
 * instruction. The compiler builtin ensures it's used directly
 * without linking against libm. Single-precision, ~14 cycles. */
static inline float sqrtf_hw(float x)
{
#if defined(__arm__) || defined(__thumb__)
    float result;
    __asm__ volatile ("vsqrt.f32 %0, %1" : "=t"(result) : "t"(x));
    return result;
#else
    /* Native regression tests execute the same estimator arithmetic. */
    return __builtin_sqrtf(x);
#endif
}

/* Inverse square root: 1/sqrt(x).
 * Could use the fast Quake trick, but the HW FPU is fast enough:
 * VSQRT + VDIV = ~28 cycles. At 168MHz that's 170 ns — nothing. */
static inline float inv_sqrtf(float x)
{
    return 1.0f / sqrtf_hw(x);
}

/* atan(x) for x in [0, 1]. Returns result in [0, π/4].
 *
 * Uses the approximation from Rajan et al.:
 *   atan(x) ≈ (π/4)*x + x*(1-x)*(0.2447 + 0.0663*x)
 *
 * At x=0: returns 0                        (exact)
 * At x=1: returns π/4 ≈ 0.7854            (exact)
 * At x=0.5: returns 0.4622 vs true 0.4636 (error 0.08°)
 * Maximum error: ~0.22° across [0, 1].
 *
 * Why this works:
 *   The first term (π/4 * x) is the linear approximation of atan.
 *   The second term adds a parabolic correction that's zero at the
 *   endpoints (x=0 and x=1) and peaks in the middle, fixing the
 *   belly of the curve where the linear approximation is worst. */
static float atan_unit(float x)
{
    return x * (0.7853982f + (1.0f - x) * (0.2447f + 0.0663f * x));
}

/* Full 4-quadrant atan2(y, x) using the atan_unit helper.
 *
 * Strategy:
 *   1. Compute atan of the smaller/larger ratio (always in [0,1])
 *   2. Adjust for the correct quadrant based on signs of x and y
 *
 * This avoids the need for a full-range atan approximation. */
static float atan2_approx(float y, float x)
{
    float ax = x < 0.0f ? -x : x;
    float ay = y < 0.0f ? -y : y;

    /* Both zero — undefined, return 0 */
    if (ax < 1e-10f && ay < 1e-10f) return 0.0f;

    float a;
    if (ax >= ay) {
        /* |x| >= |y|: ratio = ay/ax is in [0, 1] */
        a = atan_unit(ay / ax);
    } else {
        /* |y| > |x|: use identity atan(r) = π/2 - atan(1/r) */
        a = 1.5707963f - atan_unit(ax / ay);
    }

    /* Adjust for quadrant:
     *   x >= 0, y >= 0 → angle is a           (first quadrant)
     *   x <  0, y >= 0 → angle is π - a       (second quadrant)
     *   x <  0, y <  0 → angle is -(π - a)    (third quadrant)
     *   x >= 0, y <  0 → angle is -a          (fourth quadrant)  */
    if (x < 0.0f) a = 3.1415927f - a;
    if (y < 0.0f) a = -a;

    return a;
}

/* asin(x) via the identity: asin(x) = atan2(x, sqrt(1 - x²)).
 * Clamps input to [-1, 1] to avoid NaN in sqrt. */
static float asin_approx(float x)
{
    if (x >=  1.0f) return  1.5707963f;
    if (x <= -1.0f) return -1.5707963f;
    return atan2_approx(x, sqrtf_hw(1.0f - x * x));
}


/* ============================================================
 * FILTER STATE — quaternion representing current orientation
 * ============================================================
 *
 * Convention: [q0, q1, q2, q3] = [w, x, y, z]
 *
 * q = [1, 0, 0, 0] means "no rotation" — sensor frame is aligned
 * with the Earth frame (X=North, Y=West, Z=Up).
 *
 * For our drone:
 *   q0=1 means flat and level, nose pointing in reference direction.
 *   As the drone tilts, the quaternion components change.
 */
static float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
static int update_valid = 1;

int ahrs_is_valid(void)
{
    return update_valid;
}


/* ============================================================
 * ahrs_init — Reset quaternion to identity
 * ============================================================
 *
 * The filter starts assuming the drone is flat and level.
 * The accelerometer correction in ahrs_update() will quickly
 * converge to the true orientation (within ~0.5 seconds with
 * beta = 0.05, faster with higher beta).
 *
 * For best startup accuracy: keep the drone stationary for
 * the first second after boot so the accel readings are pure
 * gravity with no linear acceleration confusing the filter.
 */
void ahrs_init(void)
{
    update_valid = 1;
    q0 = 1.0f;
    q1 = 0.0f;
    q2 = 0.0f;
    q3 = 0.0f;
}


/* ============================================================
 * ahrs_update — Run one Madgwick filter iteration
 * ============================================================
 *
 * This is the heart of the filter. Called once per control loop
 * iteration (100 Hz → dt = 0.01 s).
 *
 * Inputs:
 *   gx, gy, gz — gyro rates in rad/s (body frame, axis-aligned)
 *   ax, ay, az — accel readings in any unit (body frame, axis-aligned)
 *                (they get normalized internally, so g or m/s² both work)
 *   dt         — timestep in seconds
 *
 * The algorithm has two phases:
 *
 * Phase 1: GYRO PREDICTION
 *   Update the quaternion using the gyro rate.
 *   This is a quaternion derivative: q̇ = 0.5 * q ⊗ [0, gx, gy, gz]
 *
 * Phase 2: ACCEL CORRECTION (gradient descent)
 *   Define an objective function:
 *     "Given my current quaternion, which direction should gravity
 *      point in the sensor frame?"
 *   Compare that to where the accelerometer ACTUALLY measures gravity.
 *   Compute the gradient (which direction to push the quaternion
 *   to reduce the error) and subtract it from the quaternion rate.
 *
 *   The gradient descent step is what makes Madgwick elegant:
 *   instead of a full Kalman filter (matrix inversions, covariance
 *   tracking), it uses a simple gradient of a cost function.
 *   One step of gradient descent per iteration is enough because
 *   the loop runs fast (100+ Hz) and the orientation changes slowly.
 */
void ahrs_update(float gx, float gy, float gz,
                 float ax, float ay, float az,
                 float dt)
{
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;

    update_valid = 0;
    if (!__builtin_isfinite(gx) || !__builtin_isfinite(gy) || !__builtin_isfinite(gz) ||
        !__builtin_isfinite(ax) || !__builtin_isfinite(ay) || !__builtin_isfinite(az) ||
        !__builtin_isfinite(dt) || dt <= 0.0f) return;
    const float accel_norm_sq = ax * ax + ay * ay + az * az;
    if (!__builtin_isfinite(accel_norm_sq)) return;

    /* ── Phase 1: Quaternion rate from gyroscope ────────────────
     *
     * The quaternion derivative for pure rotation is:
     *   q̇ = 0.5 * q ⊗ ω
     * where ω = [0, gx, gy, gz] (the angular velocity as a pure quaternion).
     *
     * Expanding the quaternion product:
     *   qDot1 = 0.5 * (-q1*gx - q2*gy - q3*gz)
     *   qDot2 = 0.5 * ( q0*gx + q2*gz - q3*gy)
     *   qDot3 = 0.5 * ( q0*gy - q1*gz + q3*gx)
     *   qDot4 = 0.5 * ( q0*gz + q1*gy - q2*gx)
     *
     * If we just integrated this, we'd get pure gyro integration
     * (accurate short-term, drifts long-term).
     */
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * ( q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * ( q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * ( q0 * gz + q1 * gy - q2 * gx);

    /* ── Phase 2: Gradient descent correction from accel ────────
     *
     * Only apply correction if accelerometer data is valid.
     * If accel magnitude is zero, we can't determine gravity direction.
     */
    if (accel_norm_sq > 1e-20f) {

        /* Normalize accelerometer — we only care about DIRECTION,
         * not magnitude. After normalization, [ax, ay, az] is a
         * unit vector pointing in the direction of gravity (as
         * measured in the sensor frame). */
        recipNorm = inv_sqrtf(accel_norm_sq);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        /* Pre-compute commonly used terms (saves ~20 multiplications
         * compared to expanding everything inline). */
        float _2q0 = 2.0f * q0;
        float _2q1 = 2.0f * q1;
        float _2q2 = 2.0f * q2;
        float _2q3 = 2.0f * q3;
        float _4q0 = 4.0f * q0;
        float _4q1 = 4.0f * q1;
        float _4q2 = 4.0f * q2;
        float _8q1 = 8.0f * q1;
        float _8q2 = 8.0f * q2;
        float q0q0 = q0 * q0;
        float q1q1 = q1 * q1;
        float q2q2 = q2 * q2;
        float q3q3 = q3 * q3;

        /* Gradient of the objective function.
         *
         * The objective function measures the difference between:
         *   - Where gravity SHOULD point (given our quaternion)
         *   - Where gravity ACTUALLY points (accelerometer)
         *
         * In the earth frame, gravity is [0, 0, 1] (pointing up).
         * Rotating [0, 0, 1] into the sensor frame using our quaternion
         * gives the expected accelerometer reading. The gradient tells
         * us which direction to push the quaternion to minimize the
         * difference between expected and measured.
         *
         * s0..s3 is the gradient vector ∇f = Jᵀ · f, where:
         *   f = predicted_gravity - measured_gravity  (3×1)
         *   J = Jacobian of the rotation function      (3×4)
         *   Jᵀ·f = gradient direction                 (4×1)
         *
         * These expressions come from analytically computing J^T * f
         * with the quaternion rotation formula. They look ugly but
         * they're just a bunch of multiplies and adds — the FPU
         * eats these for breakfast at 168 MHz.
         */
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1
           - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
        s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3
           - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
        s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;

        /* Normalize the gradient direction.
         * We don't need the gradient MAGNITUDE (that's controlled
         * by beta), just the DIRECTION. */
        const float gradient_norm_sq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
        if (!__builtin_isfinite(gradient_norm_sq)) return;
        if (gradient_norm_sq > 1e-20f) {
            recipNorm = inv_sqrtf(gradient_norm_sq);
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;
        } else {
            /* Exact agreement with gravity needs no correction, not 0/0. */
            s0 = s1 = s2 = s3 = 0.0f;
        }

        /* Subtract the correction from the gyro-predicted rate.
         * beta controls how fast the accel correction pulls the
         * quaternion toward the "true" gravity direction.
         *
         *   q̇_corrected = q̇_gyro - beta * gradient
         *
         * This is the key fusion step: the gyro provides fast
         * updates and the gradient descent slowly corrects drift. */
        qDot1 -= AHRS_BETA * s0;
        qDot2 -= AHRS_BETA * s1;
        qDot3 -= AHRS_BETA * s2;
        qDot4 -= AHRS_BETA * s3;
    }

    /* ── Integrate the quaternion rate ──────────────────────────
     *
     * Simple Euler integration: q_new = q_old + q̇ * dt
     *
     * At 100 Hz (dt = 0.01), Euler integration is accurate enough.
     * Higher-order methods (RK4) would only matter at much lower
     * update rates or extreme rotation speeds. */
    const float n0 = q0 + qDot1 * dt;
    const float n1 = q1 + qDot2 * dt;
    const float n2 = q2 + qDot3 * dt;
    const float n3 = q3 + qDot4 * dt;

    /* ── Normalize the quaternion ──────────────────────────────
     *
     * Quaternions must have unit length to represent pure rotations.
     * Numerical integration causes the magnitude to drift slightly
     * from 1.0 each iteration. Re-normalizing keeps it valid.
     * Without this, the quaternion gradually grows or shrinks and
     * the orientation estimate becomes meaningless. */
    const float quat_norm_sq = n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3;
    if (!__builtin_isfinite(quat_norm_sq) || quat_norm_sq <= 1e-20f) return;
    recipNorm = inv_sqrtf(quat_norm_sq);
    q0 = n0 * recipNorm;
    q1 = n1 * recipNorm;
    q2 = n2 * recipNorm;
    q3 = n3 * recipNorm;
    update_valid = 1;
}


/* ============================================================
 * QUATERNION → EULER ANGLE CONVERSION
 * ============================================================
 *
 * Roll, pitch, yaw from the quaternion, matching the same
 * convention used in the MuJoCo training environment:
 *
 *   roll  = atan2(2(q0·q1 + q2·q3), 1 - 2(q1² + q2²))
 *   pitch = asin(2(q0·q2 - q3·q1))
 *   yaw   = atan2(2(q0·q3 + q1·q2), 1 - 2(q2² + q3²))
 *
 * Convention:
 *   Roll  = rotation about X axis (+X = front of drone)
 *   Pitch = rotation about Y axis (+Y = left of drone)
 *   Yaw   = rotation about Z axis (+Z = up)
 *
 * These match drone_env.py's _quat_to_euler() exactly.
 * CRITICAL: If these don't match the training convention, the NN
 * will get wrong attitude inputs and fly erratically.
 */

float ahrs_get_roll(void)
{
    float sinr = 2.0f * (q0 * q1 + q2 * q3);
    float cosr = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
    return atan2_approx(sinr, cosr) * RAD_TO_DEG;
}

float ahrs_get_pitch(void)
{
    float sinp = 2.0f * (q0 * q2 - q3 * q1);
    return asin_approx(sinp) * RAD_TO_DEG;
}

float ahrs_get_yaw(void)
{
    float siny = 2.0f * (q0 * q3 + q1 * q2);
    float cosy = 1.0f - 2.0f * (q2 * q2 + q3 * q3);
    return atan2_approx(siny, cosy) * RAD_TO_DEG;
}

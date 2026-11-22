/*
 * fc_config.h — single source of truth for constants that main.c,
 * test_harness.c and the host tests must agree on.
 *
 * Before this header existed, main.c and test_harness.c each kept their
 * own copies of the axis alignment, altitude gains and action filter.
 * They drifted apart (e.g. the harness used ALT_KD = 0.08 while main.c
 * used 0.02), so harness results did not describe the real firmware.
 * Anything both files need now lives here.
 *
 * Every value here must also match the training environment
 * (fc-nn/drone_env.py) and the Python firmware replica (fc-nn/fw_common.py).
 */

#ifndef FC_CONFIG_H
#define FC_CONFIG_H

/* ============================================================
 * IMU AXIS ALIGNMENT  (chip frame → body frame)
 *
 * Body frame = training/MuJoCo NWU: X forward, Y left, Z up.
 * Sign conventions the NN was trained with (verified in drone_env.py):
 *   nose DOWN        → pitch > 0, ax < 0, gy > 0 while tilting
 *   right side DOWN  → roll  > 0, ay > 0, gx > 0 while tilting
 *   CW seen from above → gz < 0, yaw decreasing
 *
 * The Sologood F405 mounts its IMU rotated 180° about Z (Betaflight
 * reports "CW180"). A 180° rotation about Z maps chip → body as
 *   (x, y, z) → (-x, -y, +z)
 * and the SAME rotation must be applied to both gyro and accel, because
 * both sensors share one silicon die and one frame.
 *
 * History (fixed 2026-11-21): the previous macros were
 *   accel (x, -y, z)   gyro (-x, y, z)
 * That pair is a mirror image, not a rotation. It was self-consistent
 * for the AHRS but reported pitch with the opposite sign from training,
 * so the NN pushed the nose further in whichever direction it was
 * already tilted. Roll and yaw were already correct and are unchanged;
 * only ALIGN_AX and ALIGN_GY flipped.
 * ============================================================ */
#define ALIGN_AX(raw)  (-(raw))
#define ALIGN_AY(raw)  (-(raw))
#define ALIGN_AZ(raw)  ( (raw))
#define ALIGN_GX(raw)  (-(raw))
#define ALIGN_GY(raw)  (-(raw))
#define ALIGN_GZ(raw)  ( (raw))

/* ============================================================
 * SENSOR SCALING / OBSERVATION NORMALISATION
 * (matches drone_env.py _get_obs)
 * ============================================================ */
#define GYRO_RAW_TO_RADS   (1.0f / 939.65f)  /* ±2000 dps: 16.4 LSB/dps */
#define MAX_GYRO_RADS      34.907f           /* obs[0..2] = gyro / this  */
#define ANGLE_OBS_SCALE    (1.0f / 45.0f)    /* obs[6..7] = deg / 45     */
#define ACCEL_OBS_SCALE_G  2.0f              /* obs[3..5] = g / 2        */

/* ============================================================
 * ALTITUDE
 * ============================================================ */
#define TARGET_ALT         0.3f     /* m above baro ground reference        */
#define ALT_SCALE          2.0f     /* obs[8] = alt_err / ALT_SCALE         */
#define VVEL_SCALE         5.0f     /* obs[9] = vvel / VVEL_SCALE           */
#define HVEL_SCALE         3.0f     /* obs[10..11] (always 0 on hardware)   */
#define ALT_OBS_CLAMP      0.05f    /* obs[8] clamp (firmware-only design)  */
#define ALT_KP             0.15f    /* throttle per metre                   */
#define ALT_KD             0.02f    /* throttle per m/s                     */
#define ALT_ADJ_MAX        0.10f    /* cap on the PD throttle adjustment    */

/* ============================================================
 * ACTION FILTER (NN output smoothing)
 *
 * Fixed-alpha EMA, identical to training:
 *   drone_env.py:  smooth = 0.3 * action + 0.7 * smooth
 *
 * History (changed 2026-11-21): firmware previously used an adaptive
 * filter (alpha 0.08 when |change| < 0.15, else 0.60). Small hover
 * corrections always fell in the 0.08 branch, giving a ~120 ms time
 * constant instead of the ~28 ms the NN was trained with — corrections
 * landed ~4x later than the policy expected.
 *
 * At 100 Hz: tau = -dt / ln(1 - 0.3) ≈ 28 ms; 90 % of a step in 7 ticks.
 * ============================================================ */
#define ACTION_FILTER_ALPHA  0.3f

/* ============================================================
 * RC LINK — ExpressLRS receiver, CRSF on USART2 (TX2 = PA2, RX2 = PA3)
 *
 * Channel numbers are 0-based (CH1 = 0). Default ELRS / Betaflight AETR
 * order: CH1 roll, CH2 pitch, CH3 throttle, CH4 yaw, CH5 = AUX1 (arm —
 * ELRS sends this channel in every packet), CH6 = AUX2 (mode switch).
 * Verify with harness TEST 17 (channel finder) and change here if your
 * radio differs.
 *
 * 2026-11-21: this radio's chosen ARM switch is on CH6 (TEST 17: 1503 ->
 * 2000 us), so ARM = CH6 and MODE moved to CH5 (the two must differ:
 * with both on CH6, ARM ON would also select LAND, which blocks arming).
 * Note: ELRS sends CH5 in every packet but CH6+ in rotation, so a CH6
 * kill can arrive a few packets later than a CH5 one.
 *
 * 2026-11-22: radio is an APEX G-129 gamepad (buttons, no switches).
 * No button was found on CH5, so MODE moved to CH8 (Anti-Turtle button,
 * AUX4 per the G-129 manual). Check its two states with TEST 17.
 * ============================================================ */
#define RC_BAUD                 420000U
#define RC_CH_ROLL              0
#define RC_CH_PITCH             1
#define RC_CH_THROTTLE          2
#define RC_CH_YAW               3
#define RC_CH_ARM               5        /* CH6 */
#define RC_CH_MODE              7        /* CH8 — APEX G-129 "Anti-Turtle" button (AUX4) */
#if RC_CH_ARM == RC_CH_MODE
#error "RC_CH_ARM and RC_CH_MODE must be different channels"
#endif

#define RC_LINK_TIMEOUT_TICKS   25       /* 250 ms without a frame → failsafe   */
#define RC_ARM_ON_THRESHOLD     0.5f     /* arm switch > this (≈1700 µs) = ON   */
#define RC_THROTTLE_LOW         (-0.70f) /* throttle below this to arm (≈1141 µs). Was -0.85 (≈1065);
                                          * APEX G-129 throttle bottoms at 1107 µs and its
                                          * stick calibration would not start (2026-11-22). */
#define RC_ARM_MAX_TILT_DEG     15.0f    /* refuse to arm unless roughly level  */
#define RC_MODE_LOW_MAX         (-0.33f) /* mode switch: < this HOVER          */
#define RC_MODE_HIGH_MIN        0.33f    /*              > this LAND, else STICK*/

/* Lean commands from the right stick (STICK mode). 0 until a policy trained
 * WITH attitude setpoints is flashed: in sim the hover-only policy follows a
 * commanded lean with 3–7° error and wobbles 5–10° after the stick is
 * centred (fc-nn/sim_stick_commands.py). With 0, STICK mode = throttle →
 * altitude only; roll/pitch setpoints stay 0. */
#define RC_STICK_LEAN_ENABLE    0

#define RC_STICK_DEADBAND       0.05f
#define RC_MAX_ANGLE_DEG        8.0f     /* full stick = 8° lean (start small)  */
#define RC_ANGLE_RATE_DPS       30.0f    /* setpoint slew limit                 */
#define RC_ALT_MIN              0.20f    /* STICK mode: throttle low  → 0.20 m  */
#define RC_ALT_MAX              0.60f    /*             throttle high → 0.60 m  */
#define RC_ALT_RATE             0.20f    /* m/s target slew limit               */
#define RC_LAND_RATE            0.15f    /* m/s descent of the target in LAND   */
#define RC_LAND_TARGET          (-0.30f) /* LAND target floor (below ground)    */

/* Baro ground reference: low-passed altitude while DISARMED, frozen at the
 * moment of arming, so weather/temperature drift between power-on and
 * take-off does not shift the 0.3 m target.  alpha per 100 Hz tick. */
#define ALT_ZERO_ALPHA          0.01f

static inline void action_filter_step(float state[4], const float in[4])
{
    for (int k = 0; k < 4; k++)
        state[k] += ACTION_FILTER_ALPHA * (in[k] - state[k]);
}

#endif /* FC_CONFIG_H */

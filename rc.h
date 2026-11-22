/*
 * rc.h — pilot command logic: arming, kill switch, failsafe, flight modes,
 * stick → setpoint mapping.  Pure logic (no hardware), unit-tested on the
 * Mac by host_test/test_rc_host.c.  Constants live in fc_config.h.
 *
 * ── Arming (all must hold on the tick the ARM switch goes ON) ──
 *   link OK, IMU calibration done, throttle stick low, roll/pitch within
 *   RC_ARM_MAX_TILT_DEG, mode not LAND.
 *   If the switch is turned ON while something blocks arming (or was
 *   already ON at power-up, or after a failsafe), it must be switched OFF
 *   and ON again — the drone never arms "by itself" later.
 *
 * ── Disarm = kill ──
 *   ARM switch OFF → disarmed on that same 10 ms tick (motors to 1000 µs).
 *   Link lost for RC_LINK_TIMEOUT_TICKS (250 ms) → disarmed (failsafe).
 *
 * ── Modes (MODE switch) ──
 *   HOVER : roll/pitch setpoint 0, altitude target TARGET_ALT (0.3 m).
 *           Exactly what the current policy was trained for.
 *   STICK : throttle stick → altitude target RC_ALT_MIN..RC_ALT_MAX;
 *           right stick → roll/pitch setpoint (±RC_MAX_ANGLE_DEG) only if
 *           RC_STICK_LEAN_ENABLE (off until a setpoint-trained policy).
 *   LAND  : setpoints 0, altitude target slides down at RC_LAND_RATE to
 *           RC_LAND_TARGET (below ground) → the drone settles; then disarm.
 *   All setpoints are rate-limited so a stick flick is a ramp, not a step.
 *   The yaw stick is read but unused (the network has no yaw input yet).
 */
#ifndef RC_H
#define RC_H

typedef enum { RC_MODE_HOVER = 0, RC_MODE_STICK = 1, RC_MODE_LAND = 2 } rc_mode_t;

/* Why arming is currently refused (bitmask, shown by telemetry / TEST 15) */
#define RC_BLOCK_NO_LINK      (1 << 0)
#define RC_BLOCK_NOT_CAL      (1 << 1)
#define RC_BLOCK_THROTTLE     (1 << 2)
#define RC_BLOCK_TILT         (1 << 3)
#define RC_BLOCK_MODE_LAND    (1 << 4)
#define RC_BLOCK_SWITCH_CYCLE (1 << 5)   /* turn ARM off, then on again */

typedef enum {
    RC_DISARM_NONE = 0, RC_DISARM_SWITCH, RC_DISARM_FAILSAFE
} rc_disarm_reason_t;

typedef struct {
    int   link_ok;          /* a valid RC frame arrived within the timeout  */
    float roll, pitch;      /* sticks, -1..+1 (right / forward = +)         */
    float throttle;         /* -1 (low) .. +1 (high)                        */
    float yaw;              /* -1..+1 (unused for now)                      */
    float arm_sw;           /* -1..+1                                        */
    float mode_sw;          /* -1..+1                                        */
    int   calibrated;       /* IMU bias calibration finished                 */
    float roll_deg;         /* current attitude (for the arming tilt check)  */
    float pitch_deg;
} rc_inputs_t;

typedef struct {
    int       lean_enabled;     /* RC_STICK_LEAN_ENABLE at init              */
    int       armed;
    int       just_armed;       /* 1 only on the tick arming happened       */
    int       need_switch_cycle;
    int       block;            /* RC_BLOCK_* bits, evaluated every tick     */
    rc_mode_t mode;
    rc_disarm_reason_t last_disarm;
    float     cmd_roll_deg;     /* setpoints fed to the controller           */
    float     cmd_pitch_deg;
    float     target_alt;       /* m above the arm-time ground reference     */
    unsigned  arm_count, failsafe_count;
} rc_state_t;

void      rc_init(rc_state_t *s);
void      rc_update(rc_state_t *s, const rc_inputs_t *in, float dt);
rc_mode_t rc_mode_from_switch(float sw);
const char *rc_mode_name(rc_mode_t m);

#endif /* RC_H */

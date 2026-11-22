/*
 * rc.c — arming / kill / failsafe / mode / setpoint logic. See rc.h.
 */
#include "rc.h"
#include "fc_config.h"

static float absf(float x) { return x < 0.0f ? -x : x; }

static float clampf_rc(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* Move cur toward target by at most step. */
static float slew(float cur, float target, float step)
{
    if (target > cur + step) return cur + step;
    if (target < cur - step) return cur - step;
    return target;
}

static float deadband(float x)
{
    if (absf(x) <= RC_STICK_DEADBAND) return 0.0f;
    /* rescale so the output still reaches ±1 at full stick */
    float s = (absf(x) - RC_STICK_DEADBAND) / (1.0f - RC_STICK_DEADBAND);
    return x < 0.0f ? -s : s;
}

rc_mode_t rc_mode_from_switch(float sw)
{
    if (sw < RC_MODE_LOW_MAX)  return RC_MODE_HOVER;
    if (sw > RC_MODE_HIGH_MIN) return RC_MODE_LAND;
    return RC_MODE_STICK;
}

const char *rc_mode_name(rc_mode_t m)
{
    switch (m) {
        case RC_MODE_HOVER: return "HOVER";
        case RC_MODE_STICK: return "STICK";
        case RC_MODE_LAND:  return "LAND";
    }
    return "?";
}

void rc_init(rc_state_t *s)
{
    s->lean_enabled = RC_STICK_LEAN_ENABLE;
    s->armed = 0;
    s->just_armed = 0;
    s->need_switch_cycle = 1;      /* must see the ARM switch OFF after boot */
    s->block = 0;
    s->mode = RC_MODE_HOVER;
    s->last_disarm = RC_DISARM_NONE;
    s->cmd_roll_deg = 0.0f;
    s->cmd_pitch_deg = 0.0f;
    s->target_alt = TARGET_ALT;
    s->arm_count = 0;
    s->failsafe_count = 0;
}

/* Setpoints the current mode asks for (before slew limiting). */
static void desired_setpoints(const rc_state_t *s, const rc_inputs_t *in,
                              float *roll, float *pitch, float *alt)
{
    *roll = 0.0f;
    *pitch = 0.0f;
    *alt = TARGET_ALT;
    if (s->mode == RC_MODE_STICK) {
        if (s->lean_enabled) {
            *roll  = deadband(clampf_rc(in->roll,  -1.0f, 1.0f)) * RC_MAX_ANGLE_DEG;
            *pitch = deadband(clampf_rc(in->pitch, -1.0f, 1.0f)) * RC_MAX_ANGLE_DEG;
        }
        float thr01 = (clampf_rc(in->throttle, -1.0f, 1.0f) + 1.0f) * 0.5f;
        *alt = RC_ALT_MIN + thr01 * (RC_ALT_MAX - RC_ALT_MIN);
    } else if (s->mode == RC_MODE_LAND) {
        *alt = RC_LAND_TARGET;
    }
}

void rc_update(rc_state_t *s, const rc_inputs_t *in, float dt)
{
    int arm_on = in->arm_sw > RC_ARM_ON_THRESHOLD;
    s->just_armed = 0;

    /* Mode only follows the switch while the link is up; on link loss the
     * failsafe below disarms anyway. */
    if (in->link_ok) s->mode = rc_mode_from_switch(in->mode_sw);

    /* ── What would block arming right now? ── */
    int block = 0;
    if (!in->link_ok)                                  block |= RC_BLOCK_NO_LINK;
    if (!in->calibrated)                               block |= RC_BLOCK_NOT_CAL;
    if (in->throttle > RC_THROTTLE_LOW)                block |= RC_BLOCK_THROTTLE;
    if (absf(in->roll_deg)  > RC_ARM_MAX_TILT_DEG ||
        absf(in->pitch_deg) > RC_ARM_MAX_TILT_DEG)     block |= RC_BLOCK_TILT;
    if (s->mode == RC_MODE_LAND)                       block |= RC_BLOCK_MODE_LAND;
    if (s->need_switch_cycle)                          block |= RC_BLOCK_SWITCH_CYCLE;
    s->block = block;

    if (s->armed) {
        if (!in->link_ok) {                     /* failsafe */
            s->armed = 0;
            s->last_disarm = RC_DISARM_FAILSAFE;
            s->failsafe_count++;
            s->need_switch_cycle = 1;
        } else if (!arm_on) {                   /* kill switch */
            s->armed = 0;
            s->last_disarm = RC_DISARM_SWITCH;
        }
    } else {
        if (!arm_on) {
            if (in->link_ok) s->need_switch_cycle = 0;   /* saw a real OFF */
        } else if (block) {
            s->need_switch_cycle = 1;           /* flipped ON while blocked */
        } else {
            s->armed = 1;
            s->just_armed = 1;
            s->arm_count++;
        }
    }

    float roll, pitch, alt;
    desired_setpoints(s, in, &roll, &pitch, &alt);

    if (!s->armed || s->just_armed) {
        /* On the ground: hold setpoints at the mode's values with no slew,
         * so take-off always starts from a level, consistent command. */
        s->cmd_roll_deg = 0.0f;
        s->cmd_pitch_deg = 0.0f;
        s->target_alt = alt;
        return;
    }

    float ang_step = RC_ANGLE_RATE_DPS * dt;
    float alt_step = (s->mode == RC_MODE_LAND ? RC_LAND_RATE : RC_ALT_RATE) * dt;
    s->cmd_roll_deg  = slew(s->cmd_roll_deg,  roll,  ang_step);
    s->cmd_pitch_deg = slew(s->cmd_pitch_deg, pitch, ang_step);
    s->target_alt    = slew(s->target_alt,    alt,   alt_step);
}

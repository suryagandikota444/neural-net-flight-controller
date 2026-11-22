/*
 * test_rc_host.c — host tests for crsf.c (decoder) and rc.c (arming, kill,
 * failsafe, modes, setpoints). Compiles the REAL firmware sources.
 *
 *   make host-regression
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../crsf.c"
#include "../rc.c"

#define NEAR(a, b, tol) (fabs((double)(a) - (double)(b)) <= (tol))

static uint16_t us_to_ch(int us) { return (uint16_t)((us - 1500) * 8 / 5 + 992); }

/* ---------------------------------------------------------------- CRSF */
static void test_crc(void)
{
    /* CRC-8/DVB-S2 check value */
    assert(crsf_crc8((const uint8_t *)"123456789", 9) == 0xBC);
}

static void test_roundtrip_and_resync(void)
{
    crsf_t c; crsf_init(&c);
    uint16_t ch[16];
    for (int i = 0; i < 16; i++) ch[i] = (uint16_t)(172 + i * 100);
    uint8_t f[26];
    assert(crsf_build_rc_frame(ch, f) == 26);

    /* garbage first, then a valid frame split over arbitrary chunks */
    const uint8_t junk[] = {0x00, 0xFF, 0x13, 0xC8, 0x00, 0x55};
    for (unsigned i = 0; i < sizeof junk; i++) assert(crsf_feed(&c, junk[i]) == 0);
    int got = 0;
    for (int i = 0; i < 26; i++) got += crsf_feed(&c, f[i]) == 1;
    assert(got == 1 && c.rc_frames == 1);
    for (int i = 0; i < 16; i++) assert(c.ch[i] == ch[i]);

    /* corrupted CRC: rejected, channels unchanged */
    uint16_t ch2[16]; for (int i = 0; i < 16; i++) ch2[i] = 992;
    crsf_build_rc_frame(ch2, f);
    f[10] ^= 0x01;
    for (int i = 0; i < 26; i++) assert(crsf_feed(&c, f[i]) != 1);
    assert(c.crc_errors == 1 && c.ch[0] == ch[0]);

    /* decoder recovers on the very next good frame */
    crsf_build_rc_frame(ch2, f);
    got = 0;
    for (int i = 0; i < 26; i++) got += crsf_feed(&c, f[i]) == 1;
    assert(got == 1 && c.ch[5] == 992);

    /* back-to-back frames */
    got = 0;
    for (int r = 0; r < 50; r++) for (int i = 0; i < 26; i++) got += crsf_feed(&c, f[i]) == 1;
    assert(got == 50);
}

static void test_link_stats(void)
{
    crsf_t c; crsf_init(&c);
    uint8_t f[14] = {0xC8, 12, 0x14, 67, 70, 98, 10, 0, 2, 3, 80, 100, 5, 0};
    f[13] = crsf_crc8(&f[2], 11);
    int r = 0;
    for (int i = 0; i < 14; i++) r = crsf_feed(&c, f[i]);
    assert(r == 2 && c.lq == 98 && c.rssi_dbm == -67);
}

static void test_scaling(void)
{
    assert(crsf_to_us(172) == 988 && crsf_to_us(992) == 1500 && crsf_to_us(1811) == 2011);
    assert(NEAR(crsf_to_unit(992), 0, 1e-6) && crsf_to_unit(172) == -1.0f
           && NEAR(crsf_to_unit(1811), 1, 1e-3) && crsf_to_unit(2047) == 1.0f);
    assert(NEAR(crsf_to_unit(us_to_ch(2000)), 0.976, 0.005));  /* full scale is 2012 us */
}

/* ------------------------------------------------------------ arming */
static rc_inputs_t ground(void)
{
    rc_inputs_t in;
    memset(&in, 0, sizeof in);
    in.link_ok = 1; in.calibrated = 1;
    in.throttle = -1.0f; in.arm_sw = -1.0f; in.mode_sw = -1.0f;   /* HOVER */
    return in;
}

static void tick(rc_state_t *s, rc_inputs_t *in, int n)
{
    for (int i = 0; i < n; i++) rc_update(s, in, 0.01f);
}

static void test_arming(void)
{
    rc_state_t s; rc_init(&s);
    rc_inputs_t in = ground();

    /* switch already ON at power-up → must be cycled */
    in.arm_sw = 1.0f; tick(&s, &in, 10);
    assert(!s.armed && (s.block & RC_BLOCK_SWITCH_CYCLE));
    in.arm_sw = -1.0f; tick(&s, &in, 1);
    in.arm_sw = 1.0f;  tick(&s, &in, 1);
    assert(s.armed && s.arm_count == 1);

    /* kill: OFF disarms on the same tick */
    in.arm_sw = -1.0f; rc_update(&s, &in, 0.01f);
    assert(!s.armed && s.last_disarm == RC_DISARM_SWITCH);

    /* each blocker refuses arming, and later clearing it does NOT arm */
    struct { const char *what; int bit; } cases[] = {
        {"throttle", RC_BLOCK_THROTTLE}, {"tilt", RC_BLOCK_TILT},
        {"cal", RC_BLOCK_NOT_CAL}, {"land", RC_BLOCK_MODE_LAND},
        {"link", RC_BLOCK_NO_LINK}};
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        rc_init(&s); in = ground(); tick(&s, &in, 2);       /* switch seen OFF */
        if (cases[k].bit == RC_BLOCK_THROTTLE)  in.throttle = -0.5f;
        if (cases[k].bit == RC_BLOCK_TILT)      in.roll_deg = 20.0f;
        if (cases[k].bit == RC_BLOCK_NOT_CAL)   in.calibrated = 0;
        if (cases[k].bit == RC_BLOCK_MODE_LAND) in.mode_sw = 1.0f;
        if (cases[k].bit == RC_BLOCK_NO_LINK)   in.link_ok = 0;
        in.arm_sw = 1.0f; tick(&s, &in, 5);
        assert(!s.armed && (s.block & cases[k].bit));
        rc_inputs_t ok = ground(); ok.arm_sw = 1.0f;         /* blocker cleared */
        tick(&s, &ok, 50);
        assert(!s.armed);                                    /* still needs a cycle */
        ok.arm_sw = -1.0f; tick(&s, &ok, 1);
        ok.arm_sw = 1.0f;  tick(&s, &ok, 1);
        assert(s.armed);
    }
}

static void test_failsafe(void)
{
    rc_state_t s; rc_init(&s);
    rc_inputs_t in = ground(); tick(&s, &in, 1);
    in.arm_sw = 1.0f; tick(&s, &in, 1); assert(s.armed);
    in.throttle = 0.5f; tick(&s, &in, 100); assert(s.armed);   /* flying */
    in.link_ok = 0; rc_update(&s, &in, 0.01f);
    assert(!s.armed && s.last_disarm == RC_DISARM_FAILSAFE && s.failsafe_count == 1);
    /* link returns with switch still ON and throttle up: stays disarmed */
    in.link_ok = 1; tick(&s, &in, 100); assert(!s.armed);
    in.throttle = -1.0f; tick(&s, &in, 100); assert(!s.armed);
    in.arm_sw = -1.0f; tick(&s, &in, 1); in.arm_sw = 1.0f; tick(&s, &in, 1);
    assert(s.armed);
}

static void test_lean_disabled_by_default(void)
{
    rc_state_t s; rc_init(&s);
    assert(s.lean_enabled == RC_STICK_LEAN_ENABLE);
    s.lean_enabled = 0;
    rc_inputs_t in = ground(); tick(&s, &in, 1);
    in.arm_sw = 1.0f; tick(&s, &in, 1);
    in.mode_sw = 0.0f; in.roll = 1.0f; in.pitch = 1.0f; in.throttle = 1.0f;
    tick(&s, &in, 300);
    assert(s.mode == RC_MODE_STICK && s.cmd_roll_deg == 0 && s.cmd_pitch_deg == 0);
    assert(NEAR(s.target_alt, RC_ALT_MAX, 1e-4));            /* altitude still works */
}

static void test_modes_and_setpoints(void)
{
    rc_state_t s; rc_init(&s);
    s.lean_enabled = 1;                                       /* test the lean path */
    rc_inputs_t in = ground(); tick(&s, &in, 1);
    in.arm_sw = 1.0f; tick(&s, &in, 1);
    assert(s.armed && s.mode == RC_MODE_HOVER && s.target_alt == TARGET_ALT);

    /* HOVER ignores sticks */
    in.roll = 1.0f; in.pitch = -1.0f; in.throttle = 1.0f; tick(&s, &in, 200);
    assert(s.cmd_roll_deg == 0 && s.cmd_pitch_deg == 0 && s.target_alt == TARGET_ALT);

    /* STICK: full right/back stick → +8° roll, −8° pitch, slewed at 30°/s */
    in.mode_sw = 0.0f; tick(&s, &in, 1);
    assert(s.mode == RC_MODE_STICK && NEAR(s.cmd_roll_deg, 0.3, 1e-4));
    tick(&s, &in, 10);
    assert(NEAR(s.cmd_roll_deg, 3.3, 1e-3));                 /* 11 ticks × 0.3° */
    tick(&s, &in, 200);                                       /* alt 0.3→0.6 m takes 1.5 s */
    assert(NEAR(s.cmd_roll_deg, RC_MAX_ANGLE_DEG, 1e-4) &&
           NEAR(s.cmd_pitch_deg, -RC_MAX_ANGLE_DEG, 1e-4));
    assert(NEAR(s.target_alt, RC_ALT_MAX, 1e-4));           /* throttle high */

    /* deadband: small stick → exactly 0 */
    in.roll = 0.04f; in.pitch = -0.04f; tick(&s, &in, 200);
    assert(s.cmd_roll_deg == 0 && s.cmd_pitch_deg == 0);

    /* throttle mid → middle of the altitude range */
    in.throttle = 0.0f; tick(&s, &in, 300);
    assert(NEAR(s.target_alt, 0.5f * (RC_ALT_MIN + RC_ALT_MAX), 1e-4));

    /* LAND: level, target slides down at 0.15 m/s to the floor value */
    in.roll = 1.0f; in.mode_sw = 1.0f; tick(&s, &in, 100);
    assert(s.mode == RC_MODE_LAND && s.armed);
    assert(NEAR(s.target_alt, 0.40 - 0.15, 1e-3));
    tick(&s, &in, 1000);
    assert(NEAR(s.target_alt, RC_LAND_TARGET, 1e-4));
    tick(&s, &in, 200);
    assert(NEAR(s.cmd_roll_deg, 0, 1e-4));

    /* disarm resets setpoints */
    in.arm_sw = -1.0f; tick(&s, &in, 1);
    assert(!s.armed && s.cmd_roll_deg == 0);
}

int main(void)
{
    test_crc();
    test_roundtrip_and_resync();
    test_link_stats();
    test_scaling();
    test_arming();
    test_failsafe();
    test_lean_disabled_by_default();
    test_modes_and_setpoints();
    puts("PASS: CRSF decode/CRC/resync/link-stats, arming blockers + switch cycle, "
         "kill on same tick, failsafe, HOVER/STICK/LAND setpoints");
    return 0;
}

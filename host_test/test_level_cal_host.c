/*
 * test_level_cal_host.c — host test for level_cal.h maths (issue 003).
 *
 *   make host-regression
 *
 * Simulates the two-position calibration: a known sensor offset plus a
 * sloped surface, measured at heading 0 and 180°, and checks that the
 * offset is recovered independent of the slope, and that implausible
 * results are rejected.
 */
#include <math.h>
#include <stdio.h>
#include "../level_cal.h"

static int fails = 0;

static void check(int ok, const char *what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

/* Body-frame accel of a board with sensor offset `off` standing on a surface
 * tilted by (sx, sy) radians, rotated by heading psi about vertical. */
static void reading(const float off[3], float sx, float sy, float psi, float a[3])
{
    /* gravity direction in the surface frame (small slopes) */
    float gx = sinf(sx), gy = sinf(sy), gz = cosf(sx) * cosf(sy);
    float c = cosf(psi), s = sinf(psi);
    a[0] =  c * gx + s * gy + off[0];
    a[1] = -s * gx + c * gy + off[1];
    a[2] = gz + off[2];
}

int main(void)
{
    printf("level_cal host test\n");
    const float off[3] = {0.021f, -0.034f, 0.012f};
    const float slopes[][2] = {{0, 0}, {0.026f, -0.017f}, {-0.05f, 0.04f}};

    for (unsigned i = 0; i < sizeof slopes / sizeof slopes[0]; i++) {
        float a0[3], a180[3];
        reading(off, slopes[i][0], slopes[i][1], 0.0f, a0);
        reading(off, slopes[i][0], slopes[i][1], (float)M_PI, a180);
        level_cal_t c;
        level_cal_from_two(a0, a180, &c);
        char msg[120];
        snprintf(msg, sizeof msg, "slope (%.3f, %.3f) rad: offset x/y recovered within 1e-4 g",
                 slopes[i][0], slopes[i][1]);
        check(fabsf(c.offset[0] - off[0]) < 1e-4f && fabsf(c.offset[1] - off[1]) < 1e-4f, msg);
        snprintf(msg, sizeof msg, "slope (%.3f, %.3f) rad: surface slope reported", slopes[i][0], slopes[i][1]);
        check(fabsf(c.surface[0] - sinf(slopes[i][0])) < 1e-4f &&
              fabsf(c.surface[1] - sinf(slopes[i][1])) < 1e-4f, msg);
        check(fabsf(c.offset[2] - off[2]) < 1e-4f, "z offset recovered within 0.1 mg on the slope");
        check(level_cal_plausible(&c), "plausible");

        /* after correction, the board on THIS surface reads its slope, not 0 */
        float corr_y = a0[1] - c.offset[1];
        check(fabsf(corr_y - sinf(slopes[i][1])) < 1e-4f,
              "corrected reading shows the true surface slope");
    }

    /* 10° rotation error leaks only a fraction of the slope */
    {
        float a0[3], a190[3];
        reading(off, 0.035f, 0.0f, 0.0f, a0);                       /* 2° slope */
        reading(off, 0.035f, 0.0f, (float)M_PI + 0.1745f, a190);
        level_cal_t c;
        level_cal_from_two(a0, a190, &c);
        check(fabsf(c.offset[0] - off[0]) < 0.035f * 0.05f &&
              fabsf(c.offset[1] - off[1]) < 0.035f * 0.1f,
              "190 deg instead of 180: error < 10 % of the slope");
    }

    level_cal_t bad = {{0.15f, 0, 0}, {0, 0}};
    check(!level_cal_plausible(&bad), "0.15 g offset rejected");
    level_cal_t nan = {{NAN, 0, 0}, {0, 0}};
    check(!level_cal_plausible(&nan), "NaN rejected");
    check(level_cal_roll_deg(0.0175f) > 0.99f && level_cal_roll_deg(0.0175f) < 1.01f,
          "ay +0.0175 g = roll +1 deg (right down)");
    check(level_cal_pitch_deg(-0.0175f) > 0.99f, "ax -0.0175 g = pitch +1 deg (nose down)");

    printf(fails ? "LEVEL_CAL: %d FAILED\n" : "LEVEL_CAL: ALL PASS\n", fails);
    return fails != 0;
}

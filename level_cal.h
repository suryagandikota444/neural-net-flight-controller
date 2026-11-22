/*
 * level_cal.h — stored accelerometer level calibration (issue 003).
 *
 * WHY
 *   main.c used to zero the accelerometer at every power-up, assuming the
 *   surface the drone booted on was perfectly level. Any slope (floor,
 *   grass, a hand) became the drone's idea of "level" for that flight:
 *   it then flew leaning by that much and drifted (≈1 m in 3 s per degree).
 *
 * WHAT
 *   Harness TEST 18 measures the accelerometer twice on the same spot, the
 *   second time rotated 180° about vertical. Rotating flips the surface
 *   slope as seen by the drone but not the sensor's own offset
 *   (chip bias + how the board sits in the frame):
 *
 *       a0   = offset + slope          a180 = offset − slope      (x, y)
 *       offset = (a0 + a180) / 2       slope = (a0 − a180) / 2
 *
 *   so the surface does NOT need to be perfectly level. "Level" then means
 *   the plane the landing gear stands on. The result is stored in flash
 *   sector 10 and main.c uses it on every boot; only the gyro is still
 *   re-zeroed at boot (a resting gyro reads ~0 at any tilt).
 *
 * All values are body frame (fc_config.h ALIGN_*), in g. Subtracting an
 * offset rather than rotating is the small-angle approximation (< 0.1 %
 * error for a few degrees).
 */
#ifndef LEVEL_CAL_H
#define LEVEL_CAL_H

#include <stdint.h>

#define LEVEL_CAL_SECTOR      10U
#define LEVEL_CAL_ADDR        0x080C0000U   /* sector 10, 128 KB, reserved in linker.ld */
#define LEVEL_CAL_MAGIC       0x314C564CU   /* "LVL1" little-endian */
#define LEVEL_CAL_VERSION     1U
#define LEVEL_CAL_WORDS       8U

/* A good calibration has small offsets. Larger means the board is badly
 * mounted or the measurement went wrong (moved, not rotated 180°). */
#define LEVEL_CAL_MAX_XY_G    0.10f         /* ≈ 5.7° */
#define LEVEL_CAL_MAX_Z_G     0.10f

typedef struct {
    float offset[3];    /* subtract from body-frame accel (z: reads 1 g + offset when level) */
    float surface[2];   /* slope of the calibration surface at position 1, g (info only)   */
} level_cal_t;

static inline float level_cal_absf(float x) { return x < 0.0f ? -x : x; }

/* Two-position calibration (see top). a0, a180: mean body-frame accel (g)
 * at heading 0 and after rotating 180° in place on the same surface. */
static inline void level_cal_from_two(const float a0[3], const float a180[3],
                                      level_cal_t *c)
{
    c->offset[0]  = 0.5f * (a0[0] + a180[0]);
    c->offset[1]  = 0.5f * (a0[1] + a180[1]);
    c->surface[0] = 0.5f * (a0[0] - a180[0]);
    c->surface[1] = 0.5f * (a0[1] - a180[1]);
    /* on a slope, gravity on z is sqrt(1 − sx² − sy²) ≈ 1 − (sx² + sy²)/2,
     * not 1 (error of the approximation < 1e-5 g below 5°) */
    float s2 = c->surface[0] * c->surface[0] + c->surface[1] * c->surface[1];
    c->offset[2]  = 0.5f * (a0[2] + a180[2]) - (1.0f - 0.5f * s2);
}

/* 1 if every value is finite and within the plausible limits. */
static inline int level_cal_plausible(const level_cal_t *c)
{
    for (int k = 0; k < 3; k++)
        if (!(c->offset[k] == c->offset[k])) return 0;          /* NaN */
    for (int k = 0; k < 2; k++)
        if (!(c->surface[k] == c->surface[k])) return 0;
    return level_cal_absf(c->offset[0]) <= LEVEL_CAL_MAX_XY_G &&
           level_cal_absf(c->offset[1]) <= LEVEL_CAL_MAX_XY_G &&
           level_cal_absf(c->offset[2]) <= LEVEL_CAL_MAX_Z_G;
}

/* Approximate angles (deg, training convention) for printing.
 *   right down → ay > 0 → roll > 0;  nose down → ax < 0 → pitch > 0 */
static inline float level_cal_roll_deg(float ay_g)  { return  ay_g * 57.2958f; }
static inline float level_cal_pitch_deg(float ax_g) { return -ax_g * 57.2958f; }

/* Flash record (level_cal.c). Words: magic, version, offset[3], surface[2],
 * checksum (sum of words 1..6 + magic). Magic is written last. */
int level_cal_load(level_cal_t *out);       /* 1 = valid stored calibration     */
int level_cal_save(const level_cal_t *c);   /* 1 = written and read back OK     */

#endif /* LEVEL_CAL_H */

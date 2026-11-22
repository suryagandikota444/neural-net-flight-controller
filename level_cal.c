/*
 * level_cal.c — store / load the accelerometer level calibration in flash
 * sector 10 (0x080C0000). See level_cal.h for the method.
 *
 * Flashing new firmware with `make flash` / `make flash-test` does not
 * touch sector 10 (dfu-util erases only the sectors the ~25 KB image
 * covers), so the calibration survives reflashing and power cycles.
 * Saving erases the 128 KB sector (CPU stalls ~1-3 s): harness only.
 */
#include "level_cal.h"
#include "flash.h"

static volatile const uint32_t *const lc_flash = (volatile const uint32_t *)LEVEL_CAL_ADDR;

typedef union { float f; uint32_t u; } fu_t;

static uint32_t f2u(float f) { fu_t x; x.f = f; return x.u; }
static float    u2f(uint32_t u) { fu_t x; x.u = u; return x.f; }

static void pack(const level_cal_t *c, uint32_t w[LEVEL_CAL_WORDS])
{
    w[0] = LEVEL_CAL_MAGIC;
    w[1] = LEVEL_CAL_VERSION;
    w[2] = f2u(c->offset[0]);
    w[3] = f2u(c->offset[1]);
    w[4] = f2u(c->offset[2]);
    w[5] = f2u(c->surface[0]);
    w[6] = f2u(c->surface[1]);
    uint32_t sum = LEVEL_CAL_MAGIC;
    for (uint32_t k = 1; k < 7; k++) sum += w[k];
    w[7] = sum;
}

int level_cal_load(level_cal_t *out)
{
    uint32_t w[LEVEL_CAL_WORDS];
    for (uint32_t k = 0; k < LEVEL_CAL_WORDS; k++) w[k] = lc_flash[k];
    if (w[0] != LEVEL_CAL_MAGIC || w[1] != LEVEL_CAL_VERSION) return 0;
    uint32_t sum = LEVEL_CAL_MAGIC;
    for (uint32_t k = 1; k < 7; k++) sum += w[k];
    if (sum != w[7]) return 0;

    level_cal_t c;
    c.offset[0]  = u2f(w[2]);
    c.offset[1]  = u2f(w[3]);
    c.offset[2]  = u2f(w[4]);
    c.surface[0] = u2f(w[5]);
    c.surface[1] = u2f(w[6]);
    if (!level_cal_plausible(&c)) return 0;
    *out = c;
    return 1;
}

int level_cal_save(const level_cal_t *c)
{
    if (!level_cal_plausible(c)) return 0;
    uint32_t w[LEVEL_CAL_WORDS];
    pack(c, w);

    flash_unlock();
    int err = flash_erase_sector(LEVEL_CAL_SECTOR);
    if (!err) {
        flash_program_begin();
        for (uint32_t k = 1; k < LEVEL_CAL_WORDS && !err; k++)
            err = flash_program_word(LEVEL_CAL_ADDR + 4U * k, w[k]);
        if (!err) err = flash_program_word(LEVEL_CAL_ADDR, w[0]);   /* magic last */
        flash_program_end();
    }
    flash_lock();
    flash_dcache_flush();
    if (err) return 0;

    level_cal_t back;
    if (!level_cal_load(&back)) return 0;
    for (int k = 0; k < 3; k++) if (f2u(back.offset[k]) != f2u(c->offset[k])) return 0;
    return 1;
}

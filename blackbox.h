/*
 * blackbox.h — 100 Hz flight recorder: ONE log of the last 30 s of flight
 *
 * ── Flow ───────────────────────────────────────────────────────
 *   1. Fly. While ARMED, every control tick appends a 40-byte record to a
 *      RAM ring holding the most recent 3000 ticks (30 s at 100 Hz). Each
 *      arm starts a fresh recording.
 *   2. When the motors STOP (ARM switch off, failsafe, or the tilt/fault
 *      safety cut) the ring is written to flash sector 11 (0x080E0000).
 *      It survives power-off. Only one log exists: a new flight replaces
 *      the old one. Flights shorter than 1 s are not saved.
 *   3. To read it: hold BOOT, plug in USB (DFU mode, same as flashing) and
 *      run   make pull-log
 *      fc-nn/pull_log.py reads the sector with dfu-util, saves
 *      experiments/blackbox/<time>/{flight.csv, flight.png, summary.txt},
 *      then erases the log from the drone. Unplug/replug to run normally.
 *
 * ── Cost ───────────────────────────────────────────────────────
 *   In flight: one 40-byte copy per tick (~1 µs).
 *   On motor stop: flash erase + write, ~1–3 s with the CPU stalled
 *   (motors already at 1000 µs; cycle the ARM switch before re-arming).
 *   Flash wear: 1 erase per saved flight + 1 per pull; the sector is
 *   rated ≥ 10 000 erases → ~5 000 flights.
 *
 * ── Lost if ────────────────────────────────────────────────────
 *   the battery is pulled while still ARMED (the RAM copy never reaches
 *   flash) — flip ARM off first.
 *
 * Record layout must match fc-nn/pull_log.py (struct "<H3h3h2h4h4H2hH").
 * Flash layout: 8 header words (magic 'BBX1', version, record size, count,
 * checksum, armed ticks, 2 reserved), then the records oldest first.
 */
#ifndef BLACKBOX_H
#define BLACKBOX_H

#include <stdint.h>

#define BB_RATE_HZ        100
#define BB_RECORDS        3000          /* 30 s at 100 Hz                     */
#define BB_MIN_SAVE       100           /* don't save flights shorter than 1 s */

/* flags bits */
#define BB_F_RAMPING      (1U << 0)     /* post-arm throttle ramp active      */
#define BB_F_SAFETY_CUT   (1U << 1)     /* flight_safety cut the motors        */
#define BB_F_LINK_OK      (1U << 2)     /* RC link up                         */
#define BB_F_MODE_SHIFT   3             /* bits 3-4: rc_mode_t                 */
#define BB_F_BARO_OK      (1U << 5)
#define BB_F_LEVEL_CAL    (1U << 6)     /* accel level from stored cal (TEST 18) */
#define BB_F_LOOP_SHIFT   8             /* bits 8-15: loop time / 50 µs        */

typedef struct {
    uint16_t tick;        /* loop tick counter (low 16 bits) — gaps = stalls */
    int16_t  gyro[3];     /* body rates, rad/s × 1000  (bias-corrected)     */
    int16_t  acc[3];      /* accel, g × 1000                                */
    int16_t  roll;        /* AHRS roll, deg × 100                           */
    int16_t  pitch;       /* AHRS pitch, deg × 100                          */
    int16_t  nn[4];       /* raw NN output (clipped ±1, before filter) ×10000 */
    uint16_t pwm[4];      /* motor command, µs, NN order FL FR RR RL         */
    int16_t  alt_mm;      /* baro altitude above arm point, mm              */
    int16_t  vvel_mms;    /* baro vertical speed, mm/s                       */
    uint16_t flags;       /* BB_F_* bits                                    */
} bb_record_t;

/* Arming: start a fresh 30 s recording. */
void blackbox_begin_flight(void);

/* Every armed tick. */
void blackbox_record(const bb_record_t *r);

/* Motors have stopped: write the recording to flash (blocks ~1–3 s),
 * replacing any previous log. Call ONLY with all motors at 1000 µs.
 * Returns 1 if saved. */
int  blackbox_save(void);

#endif /* BLACKBOX_H */

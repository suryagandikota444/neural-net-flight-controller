/*
 * crsf.h — CRSF (Crossfire / ExpressLRS) receiver protocol decoder.
 *
 * Pure byte-stream parser: no hardware access, so the same file is
 * compiled into the firmware and into host_test/test_rc_host.c.
 *
 * ── Wire format ────────────────────────────────────────────────
 * 420 000 baud, 8N1, non-inverted. Every frame is
 *
 *   [addr] [len] [type] [payload ...] [crc8]
 *
 *   addr : 0xC8 (flight controller) — ELRS receivers use this; 0xEA/0xEE
 *          are also accepted.
 *   len  : number of bytes that follow (type + payload + crc), 2..62
 *   crc8 : CRC-8/DVB-S2 (poly 0xD5) over type + payload
 *
 * Frame types we decode:
 *   0x16 RC_CHANNELS_PACKED — 16 channels × 11 bits = 22 bytes payload.
 *        Channel value 172 ↔ 988 µs, 992 ↔ 1500 µs, 1811 ↔ 2012 µs.
 *   0x14 LINK_STATISTICS     — we keep uplink RSSI / link quality (LQ %).
 */
#ifndef CRSF_H
#define CRSF_H

#include <stdint.h>

#define CRSF_NUM_CHANNELS        16
#define CRSF_MAX_FRAME           64

#define CRSF_ADDR_FC             0xC8
#define CRSF_TYPE_LINK_STATS     0x14
#define CRSF_TYPE_RC_CHANNELS    0x16

#define CRSF_CH_MIN              172
#define CRSF_CH_MID              992
#define CRSF_CH_MAX              1811

typedef struct {
    /* decoder state */
    uint8_t  buf[CRSF_MAX_FRAME];
    uint8_t  pos;

    /* latest decoded data */
    uint16_t ch[CRSF_NUM_CHANNELS];   /* raw 11-bit channel values        */
    uint8_t  lq;                      /* uplink link quality, %           */
    int8_t   rssi_dbm;                /* uplink RSSI (antenna 1), dBm     */

    /* counters (for diagnostics) */
    uint32_t rc_frames;               /* valid RC channel frames           */
    uint32_t other_frames;            /* valid frames of other types       */
    uint32_t crc_errors;
    uint32_t bad_headers;             /* bytes discarded while resyncing   */
} crsf_t;

void     crsf_init(crsf_t *c);

/* Feed one received byte. Returns 1 when that byte completed a valid
 * RC_CHANNELS frame (c->ch[] updated), 2 for another valid frame type,
 * 0 otherwise. */
int      crsf_feed(crsf_t *c, uint8_t byte);

uint8_t  crsf_crc8(const uint8_t *data, int len);

/* 172..1811 → -1..+1 (clamped). */
float    crsf_to_unit(uint16_t v);

/* 172..1811 → 988..2012 µs (the numbers Betaflight's receiver tab shows). */
uint16_t crsf_to_us(uint16_t v);

/* Pack channels into a complete RC frame (26 bytes). Used by the host
 * test to generate realistic input; returns the frame length. */
int      crsf_build_rc_frame(const uint16_t ch[CRSF_NUM_CHANNELS], uint8_t out[26]);

#endif /* CRSF_H */

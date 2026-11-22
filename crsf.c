/*
 * crsf.c — CRSF byte-stream decoder (see crsf.h for the wire format).
 *
 * The decoder is a tiny state machine over a 64-byte buffer:
 *   pos 0      expect an address byte, otherwise drop it (resync)
 *   pos 1      length byte, must be 2..62, otherwise restart
 *   pos 2..    collect until we have addr + len + len bytes, then check
 *              the CRC. A corrupted frame is simply dropped; the next
 *              address byte starts a new attempt.
 * At 420 kbaud a 26-byte RC frame takes ~0.6 ms, so a corrupted frame
 * costs at most one packet; the link-timeout logic in rc.c decides when
 * "no packets" becomes a failsafe.
 */
#include "crsf.h"

static int is_address(uint8_t b)
{
    return b == CRSF_ADDR_FC || b == 0xEA || b == 0xEE;
}

void crsf_init(crsf_t *c)
{
    uint8_t *p = (uint8_t *)c;
    for (unsigned i = 0; i < sizeof(*c); i++) p[i] = 0;
    for (int i = 0; i < CRSF_NUM_CHANNELS; i++) c->ch[i] = CRSF_CH_MID;
    c->ch[2] = CRSF_CH_MIN;          /* throttle low until we hear otherwise */
}

uint8_t crsf_crc8(const uint8_t *data, int len)
{
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5) : (uint8_t)(crc << 1);
    }
    return crc;
}

/* 22 payload bytes → 16 little-endian 11-bit values */
static void unpack_channels(const uint8_t *p, uint16_t ch[CRSF_NUM_CHANNELS])
{
    uint32_t acc = 0;
    int bits = 0, k = 0;
    for (int i = 0; i < 22 && k < CRSF_NUM_CHANNELS; i++) {
        acc |= (uint32_t)p[i] << bits;
        bits += 8;
        while (bits >= 11 && k < CRSF_NUM_CHANNELS) {
            ch[k++] = (uint16_t)(acc & 0x7FF);
            acc >>= 11;
            bits -= 11;
        }
    }
}

int crsf_feed(crsf_t *c, uint8_t byte)
{
    if (c->pos == 0) {
        if (!is_address(byte)) { c->bad_headers++; return 0; }
        c->buf[c->pos++] = byte;
        return 0;
    }
    if (c->pos == 1) {
        if (byte < 2 || byte > CRSF_MAX_FRAME - 2) {
            c->bad_headers++;
            c->pos = 0;
            if (is_address(byte)) c->buf[c->pos++] = byte;   /* maybe a new start */
            return 0;
        }
        c->buf[c->pos++] = byte;
        return 0;
    }

    c->buf[c->pos++] = byte;
    uint8_t len = c->buf[1];
    if (c->pos < (uint8_t)(len + 2)) return 0;

    /* complete frame: buf[2 .. len] = type + payload, buf[len+1] = crc */
    c->pos = 0;
    if (crsf_crc8(&c->buf[2], len - 1) != c->buf[len + 1]) {
        c->crc_errors++;
        return 0;
    }
    uint8_t type = c->buf[2];
    if (type == CRSF_TYPE_RC_CHANNELS && len == 24) {
        unpack_channels(&c->buf[3], c->ch);
        c->rc_frames++;
        return 1;
    }
    if (type == CRSF_TYPE_LINK_STATS && len >= 12) {
        c->rssi_dbm = (int8_t)(-(int)c->buf[3]);   /* sent as positive dBm magnitude */
        c->lq       = c->buf[5];
    }
    c->other_frames++;
    return 2;
}

float crsf_to_unit(uint16_t v)
{
    float x = ((float)v - (float)CRSF_CH_MID) / 819.5f;
    if (x < -1.0f) return -1.0f;
    if (x >  1.0f) return  1.0f;
    return x;
}

uint16_t crsf_to_us(uint16_t v)
{
    /* Betaflight: us = (v - 992) * 5/8 + 1500 */
    return (uint16_t)(((int32_t)v - CRSF_CH_MID) * 5 / 8 + 1500);
}

int crsf_build_rc_frame(const uint16_t ch[CRSF_NUM_CHANNELS], uint8_t out[26])
{
    out[0] = CRSF_ADDR_FC;
    out[1] = 24;
    out[2] = CRSF_TYPE_RC_CHANNELS;
    uint32_t acc = 0;
    int bits = 0, n = 3;
    for (int k = 0; k < CRSF_NUM_CHANNELS; k++) {
        acc |= (uint32_t)(ch[k] & 0x7FF) << bits;
        bits += 11;
        while (bits >= 8) {
            out[n++] = (uint8_t)(acc & 0xFF);
            acc >>= 8;
            bits -= 8;
        }
    }
    out[25] = crsf_crc8(&out[2], 23);
    return 26;
}

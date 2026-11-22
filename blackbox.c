/*
 * blackbox.c — 100 Hz flight recorder. See blackbox.h for the flow.
 *
 * Storage
 *   RAM ring: 3000 × 40 B = 120 000 B, split across CCM (64 KB, 1600
 *   records) and main SRAM (1400 records) so the stack keeps plenty of
 *   room. CCM is CPU-only memory, which is all this needs.
 *
 *   Flash: sector 11 = 0x080E0000 … 0x080FFFFF (128 KB). linker.ld limits
 *   the program to the first 896 KB so code can never land there.
 *     word 0  magic 'BBX1'      word 4  checksum (sum of record words)
 *     word 1  version (1)       word 5  total armed ticks of the flight
 *     word 2  record size (40)  word 6-7 reserved
 *     word 3  record count
 *     +32     records, oldest first
 *   The magic is programmed LAST, so an interrupted save reads as "no log".
 *   pull_log.py erases the sector (via DFU) after a successful pull.
 *
 * Flash programming (RM0090 §3.6): unlock with KEY1/KEY2, sector-erase with
 * PSIZE = x32 (valid for 2.7–3.6 V), program word by word, lock. The F405
 * has a single flash bank, so the CPU stalls while erasing/programming —
 * that is why saving only happens with the motors already stopped.
 */
#include "blackbox.h"
#include "uart.h"
#include "flash.h"

/* ── layout ─────────────────────────────────────────────────────── */
typedef char bb_record_is_40_bytes[(sizeof(bb_record_t) == 40) ? 1 : -1];

#define BB_CCM_RECORDS  1600
#define BB_RAM_RECORDS  (BB_RECORDS - BB_CCM_RECORDS)

static bb_record_t bb_ccm[BB_CCM_RECORDS] __attribute__((section(".ccmram")));
static bb_record_t bb_ram[BB_RAM_RECORDS];

static uint32_t bb_head;          /* next slot to write             */
static uint32_t bb_count;         /* valid records in the ring      */
static uint32_t bb_flight_ticks;  /* armed ticks this flight        */

static bb_record_t *slot(uint32_t i)
{
    return (i < BB_CCM_RECORDS) ? &bb_ccm[i] : &bb_ram[i - BB_CCM_RECORDS];
}

/* ── flash layout (programming primitives live in flash.c) ─────── */
#define BB_SECTOR        11U
#define BB_FLASH_ADDR    0x080E0000U
#define BB_MAGIC         0x31584242U    /* "BBX1" little-endian */
#define BB_VERSION       1U
#define BB_HDR_WORDS     8U

static volatile uint32_t *const bb_flash = (volatile uint32_t *)BB_FLASH_ADDR;

/* ── recording ──────────────────────────────────────────────────── */
void blackbox_begin_flight(void)
{
    bb_head = 0;
    bb_count = 0;
    bb_flight_ticks = 0;
}

void blackbox_record(const bb_record_t *r)
{
    *slot(bb_head) = *r;
    bb_head = (bb_head + 1U) % BB_RECORDS;
    if (bb_count < BB_RECORDS) bb_count++;
    bb_flight_ticks++;
}

/* ── save (replaces the previous log) ───────────────────────────── */
int blackbox_save(void)
{
    if (bb_count < BB_MIN_SAVE) {
        bb_count = 0;
        return 0;
    }

    flash_unlock();
    int err = flash_erase_sector(BB_SECTOR);

    uint32_t sum = 0;
    if (!err) {
        flash_program_begin();
        uint32_t addr  = BB_FLASH_ADDR + BB_HDR_WORDS * 4U;
        uint32_t first = (bb_head + BB_RECORDS - bb_count) % BB_RECORDS;
        for (uint32_t n = 0; n < bb_count && !err; n++) {
            const uint32_t *w = (const uint32_t *)slot((first + n) % BB_RECORDS);
            for (uint32_t k = 0; k < sizeof(bb_record_t) / 4U && !err; k++) {
                sum += w[k];
                err = flash_program_word(addr, w[k]);
                addr += 4U;
            }
        }
        /* header last; the magic word goes in only once everything is written */
        const uint32_t hdr[BB_HDR_WORDS] = {
            BB_MAGIC, BB_VERSION, sizeof(bb_record_t), bb_count,
            sum, bb_flight_ticks, 0xFFFFFFFFU, 0xFFFFFFFFU };
        for (uint32_t k = 1; k < BB_HDR_WORDS && !err; k++)
            err = flash_program_word(BB_FLASH_ADDR + 4U * k, hdr[k]);
        if (!err) err = flash_program_word(BB_FLASH_ADDR, hdr[0]);
        flash_program_end();
    }
    flash_lock();
    flash_dcache_flush();

    uint32_t saved = bb_count;
    bb_count = 0;
    if (err || bb_flash[0] != BB_MAGIC) {
        uart_write_string("LOG: save FAILED (flash error)\r\n");
        return 0;
    }
    uart_write_string("LOG: saved last ");
    uart_write_uint32(saved / BB_RATE_HZ);
    uart_write_string(" s of flight - pull with: make pull-log\r\n");
    return 1;
}

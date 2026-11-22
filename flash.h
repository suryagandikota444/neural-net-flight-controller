/*
 * flash.h — minimal STM32F405 internal-flash programming (sector erase,
 * 32-bit word program). Shared by the flight recorder (blackbox.c,
 * sector 11) and the stored level calibration (level_cal.c, sector 10).
 *
 * Usage:
 *   flash_unlock();
 *   if (flash_erase_sector(n) == 0) {
 *       flash_program_begin();
 *       err = flash_program_word(addr, value);   ... repeat
 *       flash_program_end();
 *   }
 *   flash_lock();
 *   flash_dcache_flush();      // so reads see the new contents
 *
 * Erasing a 128 KB sector stalls the CPU for ~1-3 s: only call with the
 * motors stopped.
 */
#ifndef FLASH_H
#define FLASH_H

#include <stdint.h>

void flash_unlock(void);
void flash_lock(void);
int  flash_erase_sector(uint32_t sector);              /* 0 = ok, -1 = error */
void flash_program_begin(void);                        /* PG on, x32 words   */
int  flash_program_word(uint32_t addr, uint32_t value); /* 0 = ok, -1 = error */
void flash_program_end(void);
void flash_dcache_flush(void);

#endif /* FLASH_H */

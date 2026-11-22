/*
 * flash.c — STM32F405 internal-flash sector erase / word program.
 * See flash.h. (Moved out of blackbox.c so level_cal.c can share it.)
 */
#include "flash.h"

#define FLASH_BASE_REG   0x40023C00U
#define FLASH_ACR        (*(volatile uint32_t *)(FLASH_BASE_REG + 0x00))
#define FLASH_KEYR       (*(volatile uint32_t *)(FLASH_BASE_REG + 0x04))
#define FLASH_SR         (*(volatile uint32_t *)(FLASH_BASE_REG + 0x0C))
#define FLASH_CR         (*(volatile uint32_t *)(FLASH_BASE_REG + 0x10))

#define FLASH_KEY1       0x45670123U
#define FLASH_KEY2       0xCDEF89ABU
#define SR_BSY           (1U << 16)
#define SR_ERRORS        0xF2U          /* OPERR|WRPERR|PGAERR|PGPERR|PGSERR */
#define CR_PG            (1U << 0)
#define CR_SER           (1U << 1)
#define CR_SNB_SHIFT     3
#define CR_PSIZE_X32     (2U << 8)
#define CR_STRT          (1U << 16)
#define CR_LOCK          (1U << 31)
#define ACR_DCEN         (1U << 10)
#define ACR_DCRST        (1U << 12)

static void flash_wait(void) { while (FLASH_SR & SR_BSY) { } }

void flash_unlock(void)
{
    if (FLASH_CR & CR_LOCK) {
        FLASH_KEYR = FLASH_KEY1;
        FLASH_KEYR = FLASH_KEY2;
    }
}

void flash_lock(void)
{
    FLASH_CR |= CR_LOCK;
}

int flash_erase_sector(uint32_t sector)
{
    flash_wait();
    FLASH_SR = SR_ERRORS;                                  /* clear old errors */
    FLASH_CR = CR_PSIZE_X32 | CR_SER | (sector << CR_SNB_SHIFT);
    FLASH_CR |= CR_STRT;
    flash_wait();
    FLASH_CR &= ~CR_SER;
    return (FLASH_SR & SR_ERRORS) ? -1 : 0;
}

void flash_program_begin(void)
{
    FLASH_CR = CR_PSIZE_X32 | CR_PG;
}

int flash_program_word(uint32_t addr, uint32_t value)
{
    *(volatile uint32_t *)addr = value;
    flash_wait();
    return (FLASH_SR & SR_ERRORS) ? -1 : 0;
}

void flash_program_end(void)
{
    FLASH_CR &= ~CR_PG;
}

void flash_dcache_flush(void)
{
    /* Stale cached copies of the old contents must not be read back. */
    uint32_t acr = FLASH_ACR;
    if (acr & ACR_DCEN) {
        FLASH_ACR = acr & ~ACR_DCEN;
        FLASH_ACR = (acr & ~ACR_DCEN) | ACR_DCRST;
        FLASH_ACR = acr & ~ACR_DCEN;
        FLASH_ACR = acr;
    }
}

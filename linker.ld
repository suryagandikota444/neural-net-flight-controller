ENTRY(Reset_Handler)

MEMORY
{
    /* Top 256 KB is reserved — code must never be placed there:
     *   sector 10 (0x080C0000) stored level calibration (level_cal.c)
     *   sector 11 (0x080E0000) flight recorder log (blackbox.c) */
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 768K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 128K
    CCM (rwx)   : ORIGIN = 0x10000000, LENGTH = 64K
}

SECTIONS
{
    .isr_vector : {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH 

    .text : {
        *(.text*) 
        *(.rodata*)
    } > FLASH

    _sidata = LOADADDR(.data);  /* Flash address — source for copying */

    .data : {
        _sdata = .;              /* RAM start address */
        *(.data*)
        _edata = .;              /* RAM end address */
    } > RAM AT> FLASH            /* VMA in RAM, LMA in Flash */

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM

    _estack = ORIGIN(RAM) + LENGTH(RAM);  /* Top of RAM = 0x2001FFFF + 1 */

    /* Define a section for CCM */
    .ccmram (NOLOAD) :   /* not zeroed/copied by startup; keeps .bin small */
    {
        . = ALIGN(4);
        *(.ccmram)
        *(.ccmram*)
        . = ALIGN(4);
    } > CCM
}



#include <stdint.h>

extern uint32_t _estack;    /* Top of stack (top of RAM) */
extern uint32_t _sidata;    /* Start of .data initializers in flash */
extern uint32_t _sdata;     /* Start of .data section in RAM */
extern uint32_t _edata;     /* End of .data section in RAM */
extern uint32_t _sbss;      /* Start of .bss section in RAM */
extern uint32_t _ebss;      /* End of .bss section in RAM */

/* Forward declarations */
void Reset_Handler(void);
void Default_Handler(void);
extern int main(void); /* main is defined in main.c */

void Reset_Handler(void)
{
    /* ----- Step 1: Copy .data from flash to RAM -----
     *
     * Initialized globals like `int cal_offset = 42;` have their values
     * stored in flash (the only non-volatile storage). But the code was
     * compiled expecting these variables to be at RAM addresses. So we
     * copy the initial values from flash (at _sidata) to RAM (at _sdata).
     *
     * Without this, every initialized global variable would be garbage.
     */
    uint32_t *src = &_sidata;  /* Source: flash */
    uint32_t *dst = &_sdata;   /* Destination: RAM */
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    /* ----- Step 2: Zero the .bss section -----
     *
     * The C standard guarantees that global/static variables without
     * explicit initializers are zero. For example:
     *   uint32_t sensor_count;  // MUST be 0 at startup
     *
     * RAM is random at power-on, so we enforce this ourselves.
     * .bss doesn't need flash storage because 0 doesn't need to be "stored."
     */
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    /* ----- Step 3: Enable the Floating Point Unit -----
     *
     * The Cortex-M4F has a hardware single-precision FPU, but it's OFF
     * by default. Any float instruction (even a simple float assignment)
     * will trigger a UsageFault if the FPU isn't enabled.
     *
     * The FPU is controlled by the CPACR register in the System Control Block:
     *   Address: 0xE000ED88
     *   Bits 20-21 (CP10) and 22-23 (CP11): Set to 11 for full access
     *
     * CP10 and CP11 together control the FPU coprocessor.
     * We're writing directly to the register address here instead of using
     * CMSIS headers — this is the raw reality of what's happening.
     */
    volatile uint32_t *cpacr = (volatile uint32_t *)0xE000ED88;
    *cpacr |= ((3UL << 20) | (3UL << 22));

    /* Data Synchronization Barrier + Instruction Synchronization Barrier
     * Ensures the FPU is actually enabled before any float instructions execute.
     * Without this, the pipeline might execute a float instruction before
     * the CPACR write takes effect. */
    __asm volatile ("dsb");  /* Complete all memory accesses */
    __asm volatile ("isb");  /* Flush the pipeline */

    /* ----- Step 4: Call main -----
     * 
     * At this point:
     *   - Stack is set up (CPU did this automatically)
     *   - .data is initialized (we just copied it)
     *   - .bss is zeroed (we just did that)
     *   - FPU is enabled
     *   - We're running at 16MHz on HSI (we'll configure 168MHz in main)
     *
     * The C runtime environment is now valid. main() can use global
     * variables, floating point, function calls — everything.
     */
    main();

    /* ----- Step 5: Trap if main() returns -----
     * 
     * main() should never return in an embedded system (it runs a loop).
     * If it does, hang here rather than executing random memory.
     */
    while (1);
}

void Default_Handler(void)
{
    while (1);
}

void NMI_Handler(void)          __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)    __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)    __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void)   __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)          __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void)     __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)       __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)      __attribute__((weak, alias("Default_Handler")));

void EXTI4_IRQHandler(void)             __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream0_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream1_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream2_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream3_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream4_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream5_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream6_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA1_Stream7_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void SPI1_IRQHandler(void)              __attribute__((weak, alias("Default_Handler")));
void USART1_IRQHandler(void)            __attribute__((weak, alias("Default_Handler")));
void USART2_IRQHandler(void)            __attribute__((weak, alias("Default_Handler")));
void USART3_IRQHandler(void)            __attribute__((weak, alias("Default_Handler")));
void TIM1_CC_IRQHandler(void)           __attribute__((weak, alias("Default_Handler")));
void TIM3_IRQHandler(void)              __attribute__((weak, alias("Default_Handler")));
void TIM8_CC_IRQHandler(void)           __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream0_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream1_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream2_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));
void DMA2_Stream3_IRQHandler(void)      __attribute__((weak, alias("Default_Handler")));

/* Define a type for Interrupt Service Routines (void functions with no args) */
typedef void (*ISR_Type)(void);

__attribute__((section(".isr_vector")))
ISR_Type vector_table[] = {
    /* --- Cortex-M4 System Exceptions (entries 0-15) --- */
    (ISR_Type)&_estack,              /*  0: Initial Stack Pointer (Cast required) */
    Reset_Handler,                   /*  1: Reset */
    NMI_Handler,                     /*  2: NMI */
    HardFault_Handler,               /*  3: Hard Fault */
    MemManage_Handler,               /*  4: Memory Management */
    BusFault_Handler,                /*  5: Bus Fault */
    UsageFault_Handler,              /*  6: Usage Fault */
    0,                               /*  7: Reserved */
    0,                               /*  8: Reserved */
    0,                               /*  9: Reserved */
    0,                               /* 10: Reserved */
    SVC_Handler,                     /* 11: SVCall */
    DebugMon_Handler,                /* 12: Debug Monitor */
    0,                               /* 13: Reserved */
    PendSV_Handler,                  /* 14: PendSV */
    SysTick_Handler,                 /* 15: SysTick */

    /* --- STM32F405 Peripheral Interrupts (IRQ 0-81) --- */
    /* No casting needed anymore for functions! */

    /* IRQ  0 */ Default_Handler,        /* WWDG */
    /* IRQ  1 */ Default_Handler,        /* PVD */
    /* IRQ  2 */ Default_Handler,        /* TAMP_STAMP */
    /* IRQ  3 */ Default_Handler,        /* RTC_WKUP */
    /* IRQ  4 */ Default_Handler,        /* FLASH */
    /* IRQ  5 */ Default_Handler,        /* RCC */
    /* IRQ  6 */ Default_Handler,        /* EXTI0 */
    /* IRQ  7 */ Default_Handler,        /* EXTI1 */
    /* IRQ  8 */ Default_Handler,        /* EXTI2 */
    /* IRQ  9 */ Default_Handler,        /* EXTI3 */
    /* IRQ 10 */ EXTI4_IRQHandler,       /* EXTI4 */
    /* IRQ 11 */ DMA1_Stream0_IRQHandler,
    /* IRQ 12 */ DMA1_Stream1_IRQHandler,
    /* IRQ 13 */ DMA1_Stream2_IRQHandler,
    /* IRQ 14 */ DMA1_Stream3_IRQHandler,
    /* IRQ 15 */ DMA1_Stream4_IRQHandler,
    /* IRQ 16 */ DMA1_Stream5_IRQHandler,
    /* IRQ 17 */ DMA1_Stream6_IRQHandler,
    /* IRQ 18 */ Default_Handler,        /* ADC */
    /* IRQ 19 */ Default_Handler,        /* CAN1_TX */
    /* IRQ 20 */ Default_Handler,        /* CAN1_RX0 */
    /* IRQ 21 */ Default_Handler,        /* CAN1_RX1 */
    /* IRQ 22 */ Default_Handler,        /* CAN1_SCE */
    /* IRQ 23 */ Default_Handler,        /* EXTI9_5 */
    /* IRQ 24 */ Default_Handler,        /* TIM1_BRK_TIM9 */
    /* IRQ 25 */ Default_Handler,        /* TIM1_UP_TIM10 */
    /* IRQ 26 */ Default_Handler,        /* TIM1_TRG_COM_TIM11 */
    /* IRQ 27 */ TIM1_CC_IRQHandler,     
    /* IRQ 28 */ Default_Handler,        /* TIM2 */
    /* IRQ 29 */ TIM3_IRQHandler,        
    /* IRQ 30 */ Default_Handler,        /* TIM4 */
    /* IRQ 31 */ Default_Handler,        /* I2C1_EV */
    /* IRQ 32 */ Default_Handler,        /* I2C1_ER */
    /* IRQ 33 */ Default_Handler,        /* I2C2_EV */
    /* IRQ 34 */ Default_Handler,        /* I2C2_ER */
    /* IRQ 35 */ SPI1_IRQHandler,        
    /* IRQ 36 */ Default_Handler,        /* SPI2 */
    /* IRQ 37 */ USART1_IRQHandler,      
    /* IRQ 38 */ USART2_IRQHandler,      
    /* IRQ 39 */ USART3_IRQHandler,      
    /* IRQ 40 */ Default_Handler,        /* EXTI15_10 */
    /* IRQ 41 */ Default_Handler,        /* RTC_Alarm */
    /* IRQ 42 */ Default_Handler,        /* OTG_FS_WKUP */
    /* IRQ 43 */ Default_Handler,        /* TIM8_BRK_TIM12 */
    /* IRQ 44 */ Default_Handler,        /* TIM8_UP_TIM13 */
    /* IRQ 45 */ Default_Handler,        /* TIM8_TRG_COM_TIM14 */
    /* IRQ 46 */ TIM8_CC_IRQHandler,     
    /* IRQ 47 */ DMA1_Stream7_IRQHandler,
    /* IRQ 48 */ Default_Handler,        /* FSMC */
    /* IRQ 49 */ Default_Handler,        /* SDIO */
    /* IRQ 50 */ Default_Handler,        /* TIM5 */
    /* IRQ 51 */ Default_Handler,        /* SPI3 */
    /* IRQ 52 */ Default_Handler,        /* UART4 */
    /* IRQ 53 */ Default_Handler,        /* UART5 */
    /* IRQ 54 */ Default_Handler,        /* TIM6_DAC */
    /* IRQ 55 */ Default_Handler,        /* TIM7 */
    /* IRQ 56 */ DMA2_Stream0_IRQHandler,
    /* IRQ 57 */ DMA2_Stream1_IRQHandler,
    /* IRQ 58 */ DMA2_Stream2_IRQHandler,
    /* IRQ 59 */ DMA2_Stream3_IRQHandler,
    /* IRQ 60 */ Default_Handler,        /* DMA2_Stream4 */
    /* IRQ 61 */ Default_Handler,        /* ETH */
    /* IRQ 62 */ Default_Handler,        /* ETH_WKUP */
    /* IRQ 63 */ Default_Handler,        /* CAN2_TX */
    /* IRQ 64 */ Default_Handler,        /* CAN2_RX0 */
    /* IRQ 65 */ Default_Handler,        /* CAN2_RX1 */
    /* IRQ 66 */ Default_Handler,        /* CAN2_SCE */
    /* IRQ 67 */ Default_Handler,        /* OTG_FS */
    /* IRQ 68 */ Default_Handler,        /* DMA2_Stream5 */
    /* IRQ 69 */ Default_Handler,        /* DMA2_Stream6 */
    /* IRQ 70 */ Default_Handler,        /* DMA2_Stream7 */
    /* IRQ 71 */ Default_Handler,        /* USART6 */
    /* IRQ 72 */ Default_Handler,        /* I2C3_EV */
    /* IRQ 73 */ Default_Handler,        /* I2C3_ER */
    /* IRQ 74 */ Default_Handler,        /* OTG_HS_EP1_OUT */
    /* IRQ 75 */ Default_Handler,        /* OTG_HS_EP1_IN */
    /* IRQ 76 */ Default_Handler,        /* OTG_HS_WKUP */
    /* IRQ 77 */ Default_Handler,        /* OTG_HS */
    /* IRQ 78 */ Default_Handler,        /* DCMI */
    /* IRQ 79 */ Default_Handler,        /* CRYP */
    /* IRQ 80 */ Default_Handler,        /* HASH_RNG */
    /* IRQ 81 */ Default_Handler,        /* FPU */
};
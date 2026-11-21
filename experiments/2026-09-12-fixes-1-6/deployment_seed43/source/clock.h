/*
 * clock.h — Clock configuration for STM32F405
 *
 * After clock_init() succeeds, these are the system frequencies.
 * Use these defines throughout the firmware for any frequency-dependent
 * calculations (baud rates, timer periods, delay loops, etc.).
 * Never hardcode frequencies — if we ever change the clock config,
 * we only update these defines.
 */

#ifndef CLOCK_H
#define CLOCK_H

/* Initialize the clock tree: HSE → PLL → 168MHz
 * Returns 0 on success, -1 on failure (stays on 16MHz HSI) */
int clock_init(void);

/* System clock frequencies in Hz.
 * Use these for all timing calculations. */
#define SYSCLK_HZ       168000000U   /* 168MHz */
#define AHB_HZ           168000000U   /* 168MHz (SYSCLK / 1) */
#define APB1_HZ          42000000U    /* 42MHz  (AHB / 4) */
#define APB2_HZ          84000000U    /* 84MHz  (AHB / 2) */
#define APB1_TIMER_HZ    84000000U    /* 84MHz  (APB1 × 2, because prescaler > 1) */
#define APB2_TIMER_HZ    168000000U   /* 168MHz (APB2 × 2, because prescaler > 1) */

/* HSE crystal frequency — from Sologood board (SYSTEM_HSE_MHZ 8) */
#define HSE_HZ           8000000U     /* 8MHz */

#endif /* CLOCK_H */

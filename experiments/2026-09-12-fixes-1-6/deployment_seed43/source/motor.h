/*
 * motor.h — PWM motor output driver
 *
 * Controls 4 ESCs via standard PWM (1000–2000 µs pulses at 400 Hz).
 *
 * Motor pin mapping (from Betaflight `resource` command):
 *   Motor 1:  PC8  →  TIM8_CH3  (AF3)
 *   Motor 2:  PC9  →  TIM8_CH4  (AF3)
 *   Motor 3:  PA8  →  TIM1_CH1  (AF1)
 *   Motor 4:  PA9  →  TIM1_CH2  (AF1)
 *
 * Both TIM1 and TIM8 are advanced-control timers on APB2 (168 MHz).
 * Prescaler = 167 → 1 MHz tick rate → 1 tick = 1 µs (perfect for PWM).
 * Auto-reload = 2499 → period = 2500 µs = 400 Hz.
 *
 * After motor_init(), all outputs are at 1000 µs (idle).
 * Call motor_set() with cmd in [0.0, 1.0]:
 *   0.0 = 1000 µs (idle / disarmed)
 *   1.0 = 2000 µs (full throttle)
 *
 * ESC arming: most ESCs need 2-3 seconds of 1000 µs signal after
 * power-on before they accept throttle commands. Call motor_init()
 * early and wait before sending non-zero commands.
 */

#ifndef MOTOR_H
#define MOTOR_H

/* Initialize TIM1 and TIM8 for PWM output on motor pins.
 * All outputs start at 1000 µs (idle). */
void motor_init(void);

/* Set a single motor's throttle.
 *   index: 0–3 (Betaflight motor 1–4)
 *   cmd:   0.0 (idle/1000µs) to 1.0 (full/2000µs), clamped internally */
void motor_set(int index, float cmd);

/* Set all motors to idle (1000 µs).
 * Call this in any error condition or before disarming. */
void motor_all_stop(void);

#endif /* MOTOR_H */

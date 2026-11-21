#ifndef FLIGHT_SAFETY_H
#define FLIGHT_SAFETY_H

#include <stdint.h>

/* A fault stays latched until MCU reset. There is deliberately no re-arm API. */
void flight_safety_latch(void);
int flight_safety_check(float roll_deg, float pitch_deg, float tilt_limit_deg);
/* Final output gate: validates the entire command before writing any motor. */
int flight_safety_write_motors(const int mapping[4], const float commands[4]);

#endif

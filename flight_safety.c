#include "flight_safety.h"
#include "motor.h"
#include <float.h>

static int fault_latched;

static int finite_float(float x)
{
    return x >= -FLT_MAX && x <= FLT_MAX;
}

void flight_safety_latch(void)
{
    fault_latched = 1;
    motor_all_stop();
}

int flight_safety_check(float roll, float pitch, float limit)
{
    if (!finite_float(roll) || !finite_float(pitch) ||
        !finite_float(limit) || limit <= 0.0f ||
        roll > limit || roll < -limit || pitch > limit || pitch < -limit) {
        fault_latched = 1;
    }
    if (fault_latched) {
        motor_all_stop();
        return 0;
    }
    return 1;
}

int flight_safety_write_motors(const int mapping[4], const float commands[4])
{
    for (int i = 0; i < 4; ++i) {
        if (!finite_float(commands[i]) || commands[i] < 0.0f || commands[i] > 1.0f ||
            mapping[i] < 0 || mapping[i] >= 4) {
            fault_latched = 1;
        }
    }
    if (fault_latched) {
        motor_all_stop();
        return 0;
    }
    for (int i = 0; i < 4; ++i) motor_set(mapping[i], commands[i]);
    return 1;
}

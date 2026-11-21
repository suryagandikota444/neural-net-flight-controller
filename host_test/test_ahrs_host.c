#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "../ahrs.c"

static void check_norm(void)
{
    assert(ahrs_is_valid());
    assert(fabsf(q0*q0 + q1*q1 + q2*q2 + q3*q3 - 1.0f) < 2e-6f);
    assert(isfinite(ahrs_get_roll()) && isfinite(ahrs_get_pitch()));
}

int main(void)
{
    ahrs_init();
    for (int i=0; i<10000; ++i) {
        ahrs_update(0,0,0,0,0,1,0.01f);
        check_norm();
    }
    assert(ahrs_get_roll()==0 && ahrs_get_pitch()==0);
    /* Zero acceleration is valid gyro-only propagation: +90 deg yaw. */
    for (int i=0; i<100; ++i) ahrs_update(0,0,1.5707963268f,0,0,0,0.01f);
    check_norm();
    assert(fabsf(ahrs_get_yaw()-90.0f)<0.25f);
    const float saved=q3;
    ahrs_update(NAN,0,0,0,0,1,0.01f);
    assert(!ahrs_is_valid() && q3==saved);
    ahrs_update(0,0,0,INFINITY,0,1,0.01f);
    assert(!ahrs_is_valid() && q3==saved);
    ahrs_update(0,0,0,0,0,1,0);
    assert(!ahrs_is_valid() && q3==saved);
    ahrs_update(0,0,0,0,0,1e-30f,0.01f);
    check_norm();
    /* Supported +20-degree roll: gravity in the rotated body frame. */
    ahrs_init();
    for (int i=0; i<1500; ++i) ahrs_update(0,0,0,0,sinf(0.34906585f),cosf(0.34906585f),0.01f);
    check_norm();
    assert(fabsf(ahrs_get_roll()-20.0f)<0.5f);
    puts("PASS: actual C AHRS rest, gyro rotation, invalid inputs, norm, static tilt");
    return 0;
}

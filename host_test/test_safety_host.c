#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "../flight_safety.c"

static float applied[4];
static int active_writes;
void motor_all_stop(void) { for(int i=0;i<4;++i) applied[i]=0; }
void motor_set(int i, float cmd) { applied[i]=cmd; ++active_writes; }

int main(void)
{
    const int mapping[4]={3,1,0,2};
    const float command[4]={0.1f,0.2f,0.3f,0.4f};
    assert(flight_safety_check(0,0,60));
    assert(flight_safety_write_motors(mapping,command));
    assert(applied[3]==0.1f && applied[0]==0.3f && active_writes==4);
    assert(!flight_safety_check(61,0,60));
    /* Even a later write in the same tick cannot override the stop. */
    assert(!flight_safety_write_motors(mapping,command));
    assert(!flight_safety_check(0,0,60));
    assert(active_writes==4);
    for(int i=0;i<4;++i) assert(applied[i]==0);
    /* Reset private state here to emulate a new MCU boot in this host test. */
    fault_latched=0;
    assert(!flight_safety_check(NAN,0,60));
    fault_latched=0;
    float bad[4]={0.1f,0.2f,NAN,0.4f};
    assert(!flight_safety_write_motors(mapping,bad));
    assert(active_writes==4); /* validate ALL commands before first write */
    puts("PASS: actual C output gate mapping, tilt latch, no re-arm, invalid commands");
    return 0;
}

/*
 * rc_input.h — ExpressLRS receiver input on USART2 (RX2 = PA3, TX2 = PA2).
 *
 * The USART2 interrupt only copies received bytes into a ring buffer.
 * The control loop calls rc_input_poll() once per tick to run the CRSF
 * decoder over whatever arrived, so all parsing happens in normal code.
 * (The debug/telemetry UART stays on USART3 — see uart.h.)
 */
#ifndef RC_INPUT_H
#define RC_INPUT_H

#include <stdint.h>
#include "crsf.h"

void     rc_input_init(void);

/* Decode all bytes received since the last call. Returns the number of
 * valid RC channel frames decoded (0 if none arrived this tick). */
int      rc_input_poll(crsf_t *c);

/* Bytes lost because the ring buffer was full or the USART overran. */
uint32_t rc_input_overruns(void);

/* Total bytes received on USART2 since rc_input_init() — any value > 0
 * proves the receiver is powered and wired to RX2 (used by TEST 16). */
uint32_t rc_input_bytes(void);

#endif /* RC_INPUT_H */

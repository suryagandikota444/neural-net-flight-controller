/*
 * nn.h — Neural network inference for quadrotor control
 *
 * The trained policy network:
 *   Input:  obs[12]   — normalized sensor observations
 *   Hidden: 32 neurons (ReLU activation)
 *   Hidden: 32 neurons (ReLU activation)
 *   Output: out[4]    — residual motor commands in [-1, 1]
 *
 * Total compute: ~1536 multiply-accumulates + 64 ReLU comparisons.
 * At 168 MHz with hardware FPU: ~10–20 µs per forward pass.
 *
 * The output is NOT the final motor command. It needs to be
 * converted using the residual action mapping:
 *   motor_cmd[i] = clamp(NN_HOVER_THROTTLE + out[i] * NN_ACTION_SCALE, 0, 1)
 *
 * Where:
 *   NN_HOVER_THROTTLE  base throttle the policy was trained around
 *                      (0.372638 since the measured thrust curve, 2026-11-21)
 *   NN_ACTION_SCALE    range of adjustment around hover (0.4)
 *
 * An output of 0 means "hover." Positive = more thrust, negative = less.
 *
 * Both values live ONLY in weights.h, written by export_weights.py next to
 * the weights they were trained with. nn.c copies them into the variables
 * below so main.c and the harness always use the numbers that match the
 * network. (Until 2026-11-22 this header hard-coded 0.3226f, so exporting
 * a policy with a new hover throttle silently left the firmware on the
 * old one.)
 */

#ifndef NN_H
#define NN_H

/* Run the neural network forward pass.
 *
 *   obs:  input observation vector (12 floats, each in [-1, 1])
 *   out:  output actions (4 floats, each roughly in [-1, 1])
 *
 *   Computation:
 *     h1 = ReLU(W1 @ obs + B1)     (32 neurons)
 *     h2 = ReLU(W2 @ h1  + B2)     (32 neurons)
 *     out =      W3 @ h2  + B3     (4 outputs, linear — no activation)
 */
void nn_forward(const float obs[12], float out[4]);

/* Residual action mapping constants — defined in nn.c from weights.h */
extern const float nn_hover_throttle;
extern const float nn_action_scale;
#define NN_HOVER_THROTTLE  nn_hover_throttle
#define NN_ACTION_SCALE    nn_action_scale

#endif /* NN_H */

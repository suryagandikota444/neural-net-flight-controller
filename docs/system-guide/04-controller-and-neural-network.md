**Treat the controller as one stateful transformation from timestamped sensor data and commands to motor outputs.** The neural network is one part of that transformation. Your primary consistency test should compare the whole transformation used in simulation and firmware, including reset, calibration, filtering, clipping, altitude logic and actuator mapping.

**1. Make the observation contract explicit.** Current observation construction is split across `drone_env.py`, `main.c` and `fw_common.py`. Record the source, units, frame, normalization, clipping and measurement age of every element:

| Index | Meaning | Training source/scaling | Current firmware difference |
| --- | --- | --- | --- |
| 0–2 | Body angular rates | IMU-like gyro, divided by about 34.907 rad/s | Real aligned/calibrated gyro; fixed timestep assumptions elsewhere |
| 3–5 | Body specific force | Acceleration converted to g, divided by 2 | Real aligned/calibrated accelerometer; physical vibration/filtering |
| 6 | Roll error | True roll minus command, divided by 45°; added jitter | Madgwick roll, fixed zero command |
| 7 | Pitch error | True pitch minus command, divided by 45°; added jitter | Madgwick pitch, fixed zero command |
| 8 | Altitude error | `(true_z-target_z)/2`, clipped to ±1 | Barometric error, additionally clamped to ±0.05 |
| 9 | Upward vertical velocity | True velocity divided by 5 m/s | Filtered barometric derivative |
| 10–11 | Horizontal velocity | True world X/Y velocity divided by 3 m/s | Always zero |

A 0.05 normalized altitude clamp corresponds to only 0.10 m of distinguishable altitude error with `ALT_SCALE=2`. Beyond that distance, many different true altitude errors look identical to the NN. This is a controller-design decision, not a noise filter.

The network sees neither direct horizontal position nor a functioning yaw-rate command channel. Giving it world-frame horizontal velocity while most attitude/rate information is body-relative also requires a coherent yaw convention. If you later support arbitrary yaw and horizontal motion, test yaw-rotated scenarios and decide how frame conversion is handled. Do not silently repurpose an observation index while reusing old weights.

**2. Hand-compute reference observations.** For an ideal level, stationary, thrust-supported hover at the target, the intended normalized vector is `[0,0,0,0,0,0.5,0,0,0,0,0,0]`. A +1 rad/s roll rate becomes approximately 0.02865 in index 0. A +10° roll error becomes approximately 0.22222 in index 6. An altitude 0.4 m below target becomes -0.2 in training but only -0.05 in current firmware. A descent at 1 m/s becomes -0.2 in index 9.

Create paired Python/C tests for these values, signs and clamps. Also test gyro/acceleration values at limits, just above limits and around zero. Include invalid inputs as a distinct fault test, not as another value that a clamp is expected to sanitize. `NaN` does not behave like an ordinary out-of-range float in C comparisons.

Then create physically consistent trajectory samples. Changing roll while leaving a supported-board gravity vector unchanged is not a valid static tilt observation. On a moving aircraft, the appropriate specific force is determined by dynamics. Arbitrary vectors are useful for numerical coverage but should not be overinterpreted as possible aircraft states.

**3. Understand deterministic inference versus training exploration.** For observation `o`, your actor computes:

```text
h1 = max(W1*o + B1, 0)
h2 = max(W2*h1 + B2, 0)
raw_action = W3*h2 + B3
action = clip(raw_action, -1, +1)
```

PPO training samples from an action distribution centered on the actor output. Deterministic inference uses its mean before applying the environment's bounds. The deployable artifact therefore needs the actor, its activation/preprocessing convention and action mapping; it does not need the PPO value network or training optimizer. If observation normalization or another wrapper is introduced later, its state becomes part of the export contract.

Inspect [nn.c](../../nn.c) while evaluating a small manually constructed network with known weights in a host test. Verify bias addition, array indexing, ReLU on negative/zero/positive inputs, and the absence of final-layer ReLU. Test architecture checks separately from behavior. A tensor can have the correct shape and the wrong semantics.

**4. Expand arithmetic parity in a deliberate hierarchy.** First compare all six exported actor arrays to the intended checkpoint. Next compare raw actor outputs across PyTorch, exported NumPy and actual C on the same vectors. Then compare clipping and motor mapping. Finally compare complete stateful sequences including filters and calibration resets.

The existing 17 golden vectors produced a maximum C-vs-NumPy difference of about `1.43e-6`. Extend the vector set with recorded realistic observations, boundary cases, random bounded vectors and near-ReLU-boundary inputs. Thousands of cheap vectors are reasonable for arithmetic regression. Record absolute and relative errors, but do not demand relative agreement near a zero output. Floating-point accumulation/FMA differences mean bitwise identity is not always the right requirement.

A starting absolute output tolerance around `1e-5` can be investigated against the target/compiler; the existing host test allows `5e-4`. Convert that tolerance to actuator relevance: under scale 0.4, action error `epsilon` means approximately `400*epsilon` microseconds before quantization. Confirm actual error bounds before tightening or loosening assertions. Test the ARM build when possible, because a native host build does not validate all target/compiler behavior.

Always regenerate golden vectors when weights or inference semantics change. The current Makefile only generates the header when it is missing. Attach a model/header hash to the reference data so a pass cannot accidentally certify a different network.

**5. Characterize network behavior without mistaking it for a PID table.** The current best actor has asymmetric output even at the ideal hover vector; the largest residual is about 0.705. Yet it stabilizes altitude well in the current simulator. A single observation is not its full closed-loop operating distribution. Investigate the asymmetry through actual equilibrium trajectories, actuator forces and repeated states rather than declaring the policy broken solely from one output.

Plot action components, total commanded thrust, roll/pitch/yaw moments, clipping and true aircraft motion for a controlled episode. For small perturbations around a realistic state, estimate finite-difference sensitivities `J_ij ≈ (a_i(o+eps*e_j)-a_i(o-eps*e_j))/(2*eps)`. Express epsilon in meaningful physical units, such as one degree or a small angular-rate change. Repeat across representative states because ReLU networks have different local linear regions.

Convert actions to actual modeled forces before interpreting restorative moments. Compare the change from the local baseline: for positive roll error, the incremental torque should generally oppose it in a simple near-hover case, but a nonzero baseline bias can make an absolute pair-sum test misleading. Numerical parity, local response plausibility and full feedback stability are three different tests.

**6. Understand the output filter as controller state.** The training filter is `s_next = 0.3*a + 0.7*s`. Firmware selects alpha 0.60 when `abs(a-s)>=0.15`, otherwise 0.08. At a fixed sample period, an EMA's equivalent time constant is `tau=-dt/log(1-alpha)`. At 10 ms, alpha 0.3 gives about 28 ms; alpha 0.08 gives about 120 ms. Ninety-percent step settling takes roughly 70 ms versus 280 ms after integer tick rounding.

A large step can initially use the fast branch and then fall into the slow tail; the measured 0.5-step response took eight ticks, not the six expected by the test. This is why naming a branch “fast” does not establish the complete response time.

Test steps of both signs, subthreshold steps, values near the switching threshold, sinusoids, bursts, noise and command reversals. Log selected alpha and filter state. Compare attenuation and phase delay. With variable timestep, decide whether filtering represents a fixed per-tick algorithm to be reproduced exactly or a physical time constant to be implemented with `alpha(dt)=1-exp(-dt/tau)`. Change the same semantics in training and deployment.

The simplest initial consistency experiment is to use one fixed filter everywhere and benchmark it. That is a proposed A/B test, not evidence that alpha 0.3 is optimal for the real aircraft. Once actuator/estimator timing is measured, select the filter using end-to-end stability and noise results. Do not hide physical motor lag inside this software filter.

**7. Decide who controls altitude.** The original training environment lets the NN control attitude and collective thrust. Current firmware additionally computes:

```text
alt_adjustment = clip(Kp*(target_alt-alt) - Kd*vertical_velocity,
                      -0.10, +0.10)
base = hover_throttle + alt_adjustment
motor_i = clip(base + 0.4*filtered_nn_i, 0, 1)
```

A positive altitude deficit increases the base; upward velocity decreases it through damping. In current code `Kp=0.15` command/m and `Kd=0.02` command/(m/s). The loop is PD, not PID: there is no integral term. A constant thrust mismatch generally requires a persistent error or a nonzero collective NN contribution. Adding integral action is not the first response to a wrong thrust map; if eventually added, include anti-windup and reset behavior.

There are two coherent next designs. Keep the NN responsible for altitude, using the measured sensor/actuator/filter path and removing unexplained extra control. Or make altitude an explicit conventional outer loop and train the NN for the desired residual attitude role. In the latter case, define how collective and differential commands are separated. For example, subtracting the four-action mean can remove collective freedom from an attitude residual, but that changes the policy problem and must be present during training; it is not a safe post-export tweak to an old policy.

I would first implement switchable, documented controller variants in simulation so both designs can be evaluated on identical conditions. Choose on evidence. The present mixture of a full-control NN, altitude clamp and extra PD loop should not be treated as already validated architecture.

Correct a misleading bound in current test comments: `0.3226+0.10=0.4226` is the maximum adjusted base, assuming zero collective NN residual. It is not the absolute maximum sustainable total command. The NN can have a nonzero mean output. The per-motor algebraic upper bound with these constants is 0.8226 before any other limits, but reachable stable equilibrium depends on the learned policy and plant. Neither value alone proves a real hover capability.

**8. Test actuator authority and saturation as part of the controller.** For every scenario, log raw actions, clipped actions, filtered actions, collective base, per-motor commands, requested force/moment and achieved force/moment after actuator limits. Clipping one motor can change roll/yaw balance even when average throttle looks adequate.

Test high/low collective demands combined with roll, pitch and yaw corrections. Determine how much differential authority remains around hover and at the ends of the operating envelope. If introducing a mixer desaturation or constrained allocation algorithm, put it in both training and firmware simulation and test the axes it sacrifices under saturation. A simple per-motor clip is a real allocation policy, even if the code does not name it as one.

**9. Test command changes and persistent state.** Feed level hover followed by bounded roll/pitch/altitude steps and returns to hover. Later add yaw-rate commands after defining the observation/reward contract; currently they have no effect on the policy's inputs. The first command-related test should show that changing a setpoint changes exactly the intended observation components, with the intended sign and timestamp.

Reset after a saturated episode and compare against a newly constructed controller. Reset while a ramp is incomplete. Test fault/re-arm transitions. The NN itself is feedforward, but the estimator, output filter, barometer, ramp and possibly motor model remember history. Unobserved filter/motor state also means the actor faces partial observability; it may infer enough from recent dynamics, but that should be established experimentally. Adding previous commands or recurrent state is a later design option requiring a new input/export contract.

**10. Make faults explicit and testable.** Check finite sensor values before estimator arithmetic and finite estimator/NN/motor values before output conversion. Handle zero gradient and zero/invalid quaternion norms deliberately. Reject invalid motor indices. Record sensor freshness and transport failures. Define fault state, permitted outputs and how recovery is authorized.

For the immediate bench controller, an invalid arithmetic state should lead to a latched safe output and a recorded reason. Flight fault handling requires a separately designed policy for the available sensors and operating conditions; blindly stopping all motors after every transient sensor issue is not a universal flight recovery strategy. The present specific bug is simpler: after the tilt branch calls `motor_all_stop`, normal motor writes occur later in the same iteration. Prevent that fallthrough and test that later computations cannot overwrite a latched stop.

Compile-test these paths with injected invalid inputs and mocked actuator writes. Assert the actual emitted command sequence, not only that a stop function was called at some point. Include startup, loss of sensor data, overrun, explicit stop, repeated arm requests and reset. Inspect floating-point-to-integer conversion: a clamp based on comparisons alone does not safely convert NaN to a PWM count.

**11. Extract a portable C controller core after defining the contract.** A proposed structure is `controller_reset`, `controller_step` and a typed input/output/debug structure. Inputs contain calibrated/timestamped sensor data, commands, elapsed time and health flags; state contains estimator/filter/altitude/ramp/fault variables; outputs contain motor commands and diagnostic stages. Hardware drivers remain outside this core.

Compile the same core for STM32 and a native shared library used by the simulator. Use documented float32 layouts and explicit interfaces; do not pass ambiguous Python objects into C. Compare the C core against the old path before replacing firmware integration. Eliminate separately maintained constants or generate them from one configuration with an observation/action version and hash.

This creates a useful progression: deterministic actor in the environment; complete C controller with simulated sensors; the C controller processing recorded physical sensor data; the target MCU running known supplied samples; eventually hardware attached to a real-time simulated plant. The existing `FirmwareController` is only a Python approximation of part of this chain. Keep it as a reference model if useful, but test it against the actual core and do not let it silently fix C defects.

You have validated the controller when the same input sequence produces bounded matching stage outputs across Python/C/target builds, every state transition behaves as specified, and the complete controller meets closed-loop criteria on the corrected plant. A larger neural network is not a remedy for an undefined observation or timing contract.

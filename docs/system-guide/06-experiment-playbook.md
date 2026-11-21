**Compare a real subsystem with its simulated counterpart under the same known input, then compare the assembled feedback loop.** If you compare only two videos of hovering, you cannot tell whether a difference comes from thrust, sensor bias, filter lag, control logic or initial conditions. This chapter gives you a repeatable experiment structure and maps your existing tests into it.

**1. Give every experiment one question.** Write its hypothesis, controlled inputs, outputs, independent reference, fitted parameters and acceptance rule before collecting data. For example: “Does the motor model predict thrust within the chosen error budget at three battery-voltage conditions?” is testable. “Does the drone look stable?” is too ambiguous for diagnosing a mismatch.

Use this template for each run or coherent group of runs:

```text
Experiment ID and date:
Question / hypothesis:
Hardware configuration and fixture:
Code revision + saved dirty changes:
Firmware / model / configuration hashes:
Independent reference and its calibration/uncertainty:
Input sequence and actual input measurement:
Sampling rates, clocks and synchronization:
Fixed parameters / intentionally varied parameters:
Repetitions and ordering:
Acceptance criteria chosen before looking at results:
Raw-data paths and processing version:
Observed results, including failures and missing data:
Fit data versus held-out validation data:
Interpretation and remaining alternative explanations:
Parameter changes justified by this result:
Next discriminating experiment:
```

Retain failed experiments. A lost packet, loose mount, sensor reset or timing overrun can explain an apparent control failure; deleting the run removes that evidence. Label invalid experiments with a reason and keep them separate from valid failures of the system under test.

**2. Maintain an aircraft parameter ledger.** Use measured values and uncertainty, not only nominal constants. `Unknown` is a useful status; zero is a physical assertion and must not stand in for missing information.

| Parameter | Current repository representation | Measurement needed | Main destination after validation |
| --- | --- | --- | --- |
| Total mass | `BASE_MASS=1.118 kg`; compiled flying subtree currently 1.338 kg | Repeated ready-to-operate weighing with configuration recorded | XML mass/inertia model and shared nominal configuration |
| Motor positions | Sites near `(±0.16, ±0.16, 0.01) m` | Coordinates relative to body reference and CoG | XML body/sites and force/moment tests |
| CoG | Root nominal zero; randomized ±2 mm currently ineffective | Tared support-reaction measurements; separate height estimate if needed | Physical inertial position, compiled-model checks |
| Inertia | Root diagonal `(0.018, 0.018, 0.035) kg m²` plus child contributions | Component estimate and, if practical, independent dynamic measurement | Total or component inertial model with explicit convention |
| Thrust map | Linear normalized command × 8.5 N per motor | Force versus actual PWM/voltage, up/down sweeps, repeats | Actuator model and command-to-force/inverse mapping |
| Reaction torque | `Q/T=0.02 m`, alternating signs | Torque/RPM measurement or identified rotational response with known inertia | Yaw reaction model and uncertainty envelope |
| Motor response | Instantaneous physical response; separate software EMA | Delay, rise/fall response and operating-point dependence | Per-motor actuator state model |
| Sensor alignment | Sign macros and mounting assumptions | Known board/body motions and static orientations | Shared frame transform and matching simulator sensor pose |
| IMU calibration | Startup bias estimates and raw scaling | Six-face acceleration and independent gyro rotation tests | Calibration coefficients and startup validity rules |
| IMU variability | Training gyro 0.09 rad/s and accel 0.22 g noise | Adequately sampled static, motion and motor-condition captures | Sensor noise/bias/filter model with units and bandwidth |
| Barometer | Device settings, 25 Hz software updates, velocity EMA alpha 0.1 | Raw pressure, independent height steps, noise/drift/latency | Sensor/estimator model and scheduling |
| Control timing | Nominal 100 Hz, fixed `dt=0.01 s` | Actual acquisition, compute, output and missed-update timing | Scheduler, timestamps and timing-disturbance model |
| Aerodynamics/contact | Limited modeled effects | Residual motion evidence and bounded operating-envelope tests | Drag/ground/contact model only as supported by evidence |

For each fitted number, add units, reference frame, operating range, source experiment ID, confidence/uncertainty and date. Model variation across motors explicitly when observed. Record which quantities covary instead of independently randomizing every constant.

**3. Upgrade observability before collecting the next serious dataset.** The current TEL format is useful for seeing broad behavior, but it lacks device acquisition timestamps and some intermediate states. Its `nn_0..3` fields are post-smoothing, so they do not directly reveal raw network output. `obs_8` is clamped, so it cannot recover unrestricted altitude error. Host arrival time includes serial transmission and buffering.

Design a versioned log schema with the following groups. You do not have to transmit every field on every tick: use synchronized records at appropriate rates, explicit field validity, and a bandwidth budget.

| Group | Fields and purpose |
| --- | --- |
| Identity | Schema version, run ID, firmware/config/model hashes and board/sensor identifiers |
| Time and health | Sequence number, device acquisition time, control start/end time, sample age, overrun count, dropped-record count and sensor status |
| Raw measurements | IMU integer registers or raw samples, barometric pressure/temperature, supply voltage/current and RPM when available |
| Calibrated measurements | Gyro in rad/s, specific force in m/s² or g with explicit units, applied biases/scales and frame transform |
| Estimated state | Quaternion, roll/pitch, altitude/reference, vertical velocity, estimator validity and filter state |
| Task/controller | Command values, all 12 observations, raw NN output, clipped output, smoothed output, chosen alpha, altitude P/D terms, base and final motor command |
| Actuator output | Requested and actually applied PWM, output timestamp, enable/arming state and saturation flags |
| Independent reference | Measured position/orientation/force and reference timestamp, calibration and quality flags |
| Simulation only | True state, actual body parameters, modeled force/moment, disturbance, contact and randomization samples |

Do not feed the simulation-only group into the controller except in a deliberately labeled ideal-state diagnostic. Distinguish requested PWM from a measured pin waveform and from actual motor force: each comparison proves a different link.

Buffer logging or use a suitable nonblocking transport so instrumentation does not change control timing. At 115200 baud with conventional 8N1 framing, available payload capacity is about 11,520 bytes/s before application overhead; the current long text record cannot fit at 100 Hz. Quantify that budget before choosing a format. Measure timing with logging enabled and disabled and record the difference.

**4. Establish a passive electronics baseline.** First implement an explicit bench entry point that cannot progress into automatic arming or powered motor tests. The present startup and harness do not provide that guarantee. With propellers removed and propulsion power isolated as appropriate to the bench setup, verify boot messages, sensor identity/configuration, timestamps, communication status and output-disable behavior. A logic analyzer or oscilloscope can check whether the intended motor pins remain in the defined disabled state without inferring this from printed messages.

Capture at least several repeats of stationary data after warmup, with the board orientation and environmental conditions recorded. Sixty seconds is a useful initial capture duration for basic mean/variance and timing checks; use longer runs for temperature drift or low-frequency bias. Read at the actual sensor/control sampling rate and choose a sufficiently high-rate acquisition path for vibration analysis. A 100 Hz control log cannot identify vibration components above its Nyquist limit without an appropriate higher-rate measurement and antialiasing strategy.

Expected output is a baseline report: sensor IDs and register readbacks, calibration status, mean/std per axis, sample-period distribution, sample age, packet losses and estimator finiteness. Do not conclude that the aircraft is correctly calibrated merely because startup subtracts an offset and prints zeros.

**5. Run sensor and estimator protocols with an independent reference.** Use the procedures in [chapter 03](03-sensors-estimation-and-timing.md), and organize the resulting runs as follows:

| Protocol | Controlled input and reference | Fit/measure | Hold-out check |
| --- | --- | --- | --- |
| Accelerometer six-face | Known ±axis orientations, stationary support | Offset and scale; richer model if justified | Intermediate orientations absent from fitting |
| Gyro bias/scale | Stationary periods and independently known rotations | Bias versus temperature; scale and axis sign | Different angles/rates and both directions |
| AHRS static/dynamic | Supported known tilts and reference rotation trajectories | Convergence, phase delay, error and drift | Motions not used to choose estimator parameters |
| Barometric height | Repeated independently measured height changes with timestamps | Pressure-to-height behavior, bias, noise and lag | Different heights, directions and repeats |
| Vertical-velocity estimate | Reference height trajectory with controlled processing | Noise/delay and response to starts/stops | A separate trajectory and operating condition |
| Loaded vibration | Appropriate guarded test setup and measured operating point | Spectrum, clipping, mounting resonance and condition dependence | Unused command/voltage points and repeat setup |

A camera observing a calibrated scale or known geometry can provide low-frequency position/angle evidence if its resolution, frame rate, perspective and timing are accounted for. It is not automatically a high-bandwidth motion reference. Differentiating noisy video position twice to infer force can make noise dominate. Prefer the direct reference appropriate to the quantity being identified, and propagate its uncertainty.

For vertical velocity, do not validate one noisy derivative against another derivative produced with the same filter and then call their agreement independent truth. Use a known controlled trajectory or a separately characterized reference pipeline; state the bandwidth over which the comparison is meaningful.

**6. Characterize propulsion as a measured actuator subsystem.** Use an appropriately guarded, rated force/torque measurement fixture with a verified stop path and operating limits. The current harness's improvised restraint instructions are not a qualified fixture procedure. Do not infer real free-flight hover from a restrained airframe's controller output alone; restraints supply unknown reaction forces and alter dynamics.

For each motor/propeller/ESC combination, collect commanded and measured PWM, thrust, voltage at the relevant electrical location, current, RPM if available and temperature. Use stepped operating points within the fixture's verified envelope, include ascending and descending sequences, and repeat points after warmup. Avoid treating the first sample after a step as steady force.

Fit a monotonic thrust curve over the measured operating region. Validate on withheld operating points and runs. Look for systematic errors versus voltage and motor identity. Record extrapolation regions rather than assuming a fitted curve remains valid outside the data. A smooth curve with small training residuals can still be physically wrong if the load cell was mis-tared or supply voltage changed during the sweep.

For dynamics, apply bounded changes around multiple operating points and record time-resolved force/RPM and actual input timing. Identify dead time separately from rise/fall time constants. Check whether one first-order response explains the held-out data; use separate acceleration/deceleration dynamics only when measurements justify them. Feed the measured command through the simulator's actuator model and compare force traces before closing the aircraft loop.

The simulation's 8.5 N value is a placeholder until this evidence exists. While measurements are pending, test a clearly labeled uncertainty range; do not spend time tuning the NN to a guessed single value and calling the model calibrated.

**7. Use three distinct replay/comparison modes.** Each answers a different question:

| Mode | What is kept the same | What it isolates |
| --- | --- | --- |
| Sensor replay | Recorded sensor samples, commands, timestamps, initial estimator/controller state | Python/C/target controller and estimator behavior |
| Open-loop plant replay | Measured actuator input sequence, initial plant state and documented disturbances | Actuator/physics prediction without controller differences |
| Closed-loop matched experiment | Same task, operating envelope, controller configuration and reference procedure | Performance of the assembled system, including feedback interactions |

For sensor replay, preserve chronological order and sample delays. Reset state once at the documented start, not at every row. Compare intermediate signals so an output mismatch can be traced to alignment, calibration, observations, inference, filtering or mapping. Include known dropout and reset events.

For open-loop plant replay, use recorded actual PWM or another properly defined input, not the simulator's independently generated control actions. Initialize from measured state with uncertainty. Start with short prediction windows so estimator error, unknown wind and chaotic divergence do not overwhelm interpretation. The same input sequence can be replayed in simulation; do not apply uncontrolled open-loop sequences to a freely moving physical aircraft merely to match this test.

For closed-loop comparisons, exact path matching over a long trajectory is usually less informative than matching response envelopes, frequencies, biases, latency, saturation and failure modes across repeated trials. If real and simulated controls differ substantially early, trace the first divergence rather than comparing only final altitude.

**8. Align time without hiding the error you need to measure.** Prefer a shared trigger or recorded synchronization event. Otherwise estimate a clock mapping from multiple known events and document uncertainty. Host and device clocks can have an offset and a small rate difference; a single start timestamp may not align a long run.

Choose any permitted delay correction using calibration data and keep it fixed on validation data. Show both uncorrected system response and a clearly labeled aligned diagnostic when useful. Independently shifting every output to minimize its error hides real relative delays and can make an unstable timing model look accurate. Do not use arbitrary time warping as evidence of controller equivalence.

Resample only with documented interpolation/filtering. Preserve gaps and validity masks; do not draw a smooth signal through a long dropout and analyze it as a normal measurement. Restrict frequency comparisons to the common reliable measurement bandwidth.

**9. Quantify real-versus-sim error by quantity and operating condition.** For a measured reference `y_real` and prediction `y_sim`, compute residual `e=y_sim-y_real`. Report bias, RMSE, maximum or high-percentile absolute error, and repeat-to-repeat variability in physical units. For cyclic orientation, use a wrapped angle or quaternion error; direct subtraction near ±180° produces a fictitious large error.

Plot residual against time, command, voltage, temperature, motor identity and motion direction. Patterns identify missing structure: error increasing with command suggests an incorrect curve; opposite-sign error on acceleration versus deceleration suggests dynamics/hysteresis; a shifted but similarly shaped response suggests delay; a fixed error suggests a bias or reference mismatch. These are hypotheses to test, not unique diagnoses.

For periodic motion or vibration, compare power spectra and relevant cross-signal timing within the measured bandwidth. Keep units consistent: spectral density and amplitude spectra have different units and normalization. A similar total standard deviation does not mean the frequency content or control impact is similar.

Define an error budget at the task level. If you want to establish 5 cm altitude accuracy but the independent reference uncertainty is 10 cm, that experiment cannot resolve your target. Sensor repeatability, absolute accuracy and latency are separate limitations. A smooth barometer signal is not automatically an accurate one.

**10. Split identification data by runs, not neighboring samples.** Fit parameters on some complete runs and validate them on separate runs, ideally with unused operating conditions within the intended envelope. Adjacent samples share noise and state; randomly assigning rows from the same step response to train/test creates leakage.

Reserve entire voltage/temperature/command combinations for stress checks when practical. Estimate confidence intervals using independent runs or appropriate blocks, and show the number of independent repetitions. If changing several model parameters produces equally good fits, report that identification ambiguity. Add a measurement that distinguishes them instead of presenting an arbitrary parameter combination as uniquely measured.

For example, insufficient vertical acceleration could result from excess mass, low thrust, incorrect command scaling or a tilted thrust axis. Weighing mass and measuring force independently removes ambiguity that trajectory fitting alone may not resolve. Similarly, angular acceleration alone confounds torque and inertia until one is independently constrained.

**11. Reuse the existing fourteen harness tests with accurate scope.** The [test harness](../../test_harness.c) is valuable diagnostic code, but its automatic execution and printed conclusions need repair. Select tests explicitly, separate powered from passive tests, use structured result records and verify stop handling. The current file automatically reaches tests 10 and 14.

| Test | What it currently exercises | How to strengthen or interpret it |
| --- | --- | --- |
| 1 — SPI register readback | Sensor communication and selected configuration | Compare expected/actual ID and all relevant fields; exercise timeout/error reporting. |
| 2 — Static IMU | Resting means/variability | Record orientation, calibration state, sample timing and independent gravity expectations; extend to repeated conditions. |
| 3 — Axis alignment | Responses while manually rotating/tilting | Specify body motions and right-hand signs; compare raw, aligned and estimated values with a known reference. |
| 4 — AHRS convergence | Estimated attitude during tilt/return | Add exact-rest C regression, reference orientation, convergence dwell and dynamic error/latency. |
| 5 — Observation vector | Sensor-to-normalized-input mapping | Hand-computed vectors, clipping limits, units and the training/firmware differences in chapter 04. |
| 6 — NN inference | Known-input computation and response expectations | Distinguish numerical parity from physical plausibility; identify model/header hashes and raw versus clipped outputs. |
| 7 — Named HIL closed loop | Real sensor/controller calculations with motor output disabled | Useful sensor-to-command integration; not a simulated aircraft feedback loop. Add actual plant coupling for HIL. |
| 8 — Live tilt to predicted motor response | Axis/motor-pair response diagnostics | Interpret incremental torque about realistic state and distinguish raw/filtered/PWM outputs. |
| 9 — IIR lag | Filter response during an input change | Add synthetic exact step sequences, measured timestamps, threshold and both-direction coverage. |
| 10 — Motor wire verification | Actual motor operation to identify mapping | Props-off, explicit selection and verified stop; confirm physical location/direction, not only electrical channel numbering. |
| 11 — IIR stress | Artificial sequences around adaptive behavior | Record state/alpha and compare exact C/Python sequences; inspect limit cycles and phase lag. |
| 12 — Timing jitter | Loop timing/overrun observations | Include sensor acquisition age and output timing, external measurement, logging on/off, and worst-case code paths. |
| 13 — Barometer-to-command variability | Barometric height/velocity feeding calculated altitude P/D adjustment on the bench | Does not measure powered propwash effects or prove the cause of actual motor pulsing. Measure both noise and lag. |
| 14 — Throttle staircase | Powered command steps, IMU diagnostics and manually observed force/hover information | Replace automatic/improvised procedure with a guarded calibrated actuator experiment; capture force/voltage/timing explicitly. |

Some printed recommendations conflict with the implementation or one another. For example, increasing an EMA alpha passes more high-frequency variation and reduces lag; decreasing alpha smooths more and adds lag. A suggestion to reduce barometer update frequency also changes derivative/filter timing. Reproduce the symptom and measure the resulting noise-delay tradeoff before adopting a test's suggested constant.

Test 13's peak-to-peak P/D contribution can identify a path that creates command variation under that particular bench condition. It cannot, by itself, assign a percentage of real-flight motor pulsing to that path or justify the hardcoded claims about how many times worse propwash will be. Use simultaneous raw/estimated/controller/actuator data and controlled A/B changes for that conclusion.

**12. Create a run manifest that travels with the results.** This is a proposed JSON shape, not an existing CLI feature. Replace placeholders with actual values and store the referenced configurations alongside it.

```json
{
  "schema_version": 1,
  "experiment_id": "hover_nominal_candidate_a",
  "code": {"commit": "<git revision>", "dirty_snapshot": "source-snapshot.patch"},
  "artifacts": {"checkpoint_sha256": "<hash>", "header_sha256": "<hash>", "firmware_sha256": null},
  "configuration": {"plant": "plant.json", "controller": "controller.json", "scenario": "scenario.json"},
  "runtime": {"python_packages": "environment.txt", "compiler": null, "backend": "mujoco+controller-c"},
  "seeds": {"scenario": 100, "plant": 200, "sensor": 300, "disturbance": 400},
  "timing": {"physics_dt_s": 0.00025, "control_period_s": 0.01, "horizon_s": 30},
  "task": {"kind": "attitude_altitude", "target_altitude_m": 1.0, "horizontal_position_target": null},
  "criteria": "acceptance.json",
  "results": {"trajectory": "trajectory.npz", "summary": "summary.json"}
}
```

Use separate explicit seeds only after implementing the corresponding independent streams. Record actual effective configuration after overrides, not merely the requested settings. A git diff does not include untracked files, so preserve those separately in the source snapshot.

**13. End an experiment with a justified change or a narrower question.** If measured mass is wrong, change the physical mass representation and re-run analytic tests. If measured force differs, change the actuator model and re-run held-out force prediction. If controller outputs differ on identical sensor replay, repair implementation consistency. If both implementations match but closed-loop behavior fails on a calibrated model, adjust the controller/task/training design.

Change one main cause per comparison and keep the previous results. When several corrections are logically inseparable, state that bundle and test its component invariants as well as the combined behavior. Record whether the old model's performance worsens after correcting physics; that can be evidence that it learned the old plant, not a reason to restore a known physical error.

The deliverable from this chapter is a traceable chain: physical measurement → fitted parameter with uncertainty → held-out prediction → simulation configuration → controller benchmark. That chain is what will make your simulation increasingly credible.

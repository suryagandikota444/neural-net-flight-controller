**Return to hardware: current state, failure modes, deployment, and measurement — September 12, 2026.**

You confirmed that the Sologood F405 / F450 setup is unchanged and that you still have the USB-to-UART adapter and a test stand. This guide describes the code currently on disk. The binary actually installed on the board has not been identified or read back. No hardware was flashed or powered by this review.

The immediate objective should be a repeatable, instrumented bench test of the complete controller. A good-looking viewer episode is useful evidence about the policy on one nominal simulated aircraft. It does not establish that the deployed controller can take off, hold position, or handle faults.

**1. What is already working, and what remains unproved.**

The strongest candidate from the latest experiments is `experiments/2026-09-12-fixes-1-6/training_seed43/best_model.zip`, selected after 753,664 additional training transitions. It uses the existing 12 → 32 → 32 → 4 actor. The fixes and exact experiment results are in [the implementation report](../../experiments/2026-09-12-fixes-1-6/REPORT.md).

On 20 unused seeds per scenario, its nominal 30-second altitude RMSE averaged 2.42 cm. However, its mean maximum nominal horizontal displacement was 3.74 m, and randomized-hover RMSE averaged 24.22 cm. It completed 140/140 episodes but met every criterion in only 94/140. Completion, altitude accuracy, and remaining within a test area are separate results. The viewer follows the drone with its camera and disables domain randomization; both make it easy to miss weaknesses revealed by numerical tests.

The selected actor passed actual native C arithmetic comparison on 1,017 observations, with maximum absolute action error 2.86e-6. That verifies inference arithmetic, not the complete estimator/controller/motor loop. The root `weights.h` was deliberately preserved; building the firmware today does not automatically select the new simulation checkpoint. Neither the new candidate nor the recent safety fixes can be assumed to be on the board.

**2. Refresh: how your project fits together.**

| Part | Existing files | What it does |
| --- | --- | --- |
| Python flight model | `fc-nn/quadrotor.xml`, `drone_env.py` | MuJoCo physics, observations, policy actions, noise and randomized parameters. |
| Training and selection | `train.py`, `run_experiment.py`, `evaluation.py` | PPO learning and fixed scenario evaluation. The experiment runner produced the latest candidate. |
| Visual flight | `viewer.py`, `main.py view` | Runs the PPO actor on the Python environment. Keyboard/gamepad commands stay on the computer. |
| Model export | `export_weights.py`, root `weights.h` | Converts the actor to C arrays compiled into the binary. There is no runtime ZIP loading on the MCU. |
| Firmware startup | root `startup.c`, `clock.c`, `main.c` | Starts the MCU, configures clocks/drivers, calibrates, and runs the control loop. |
| Sensors and estimation | `spi.c`, `imu.c`, `i2c.c`, `baro.c`, `ahrs.c` | Reads physical sensors, estimates attitude, and derives relative altitude/vertical velocity. |
| Controller and actuators | `main.c`, `nn.c`, `flight_safety.c`, `motor.c` | Builds observations, runs the actor, filters actions, adds altitude control, and writes PWM. |
| Flashing | root `Makefile` | Builds ARM binaries and downloads them over STM32 DFU. |
| Telemetry | `uart.c`, `uart.h`, `fc-nn/log_imu.py` | Sends text over a separate UART connection and saves parsed TEL records. |
| Analysis | `fc-nn/analyze_imu.py` | Produces six sets of plots and descriptive statistics from the current TEL CSV format. |
| Hardware tests | `test_harness.c` | Automatically runs 14 tests, including powered motor tests. It is not a passive bench image. |

The real path is: physical motion → IMU/barometer → alignment/calibration/AHRS → 12 observations → C actor → adaptive filter plus altitude PD → PWM → ESC/motor/prop → motion. A separate branch sends telemetry to the laptop.

There is no implemented RC command/arming receiver path, MAVLink ground station, normal USB telemetry interface, or gamepad-to-drone bridge in this firmware. The viewer's controls do not control the physical aircraft. The UART driver exposes transmit functions; wiring its RX pin does not create an arm/stop command handler.

**3. The current startup sequence needs changing before powered controller testing.**

Read [main.c](../../main.c), especially the initialization and motor-identification block. Normal mode currently:

1. Configures the clock, UART, IMU, PWM, AHRS and barometer. PWM initializes to 1000 microseconds.
2. Waits about three seconds for the ESCs.
3. Automatically commands FL, FR, RL and RR one at a time at `0.50` command for 300 ms each, with stops between. This is **1500 microseconds**, despite a comment saying approximately 30%.
4. Settles and establishes the barometer reference if a barometer was detected.
5. Runs a nominal two-second IMU settling/calibration interval, assuming a level, stationary board.
6. Sets `armed = 1` without a pilot command, then ramps the controller output over 50 ticks, nominally half a second.

Do not use elapsed seconds as a protection mechanism: initialization, barometer waits, and faults change the sequence timing. A reset also clears the new fault latch and starts this automatic sequence again.

The repaired safety gate latches invalid AHRS/tilt/output faults until reset and prevents normal-loop motor writes from overriding the stop. It does not remove startup motor activity, provide a remote stop, handle every stale sensor or deadline fault, or guarantee a stop if the CPU hangs before reaching the gate. `motor_all_stop()` sends 1000-microsecond pulses; it is not an electrical power disconnect. Verify how your ESCs interpret that signal.

The harness also needs separation: tests 1–6 check registers, static IMU, axis response, AHRS, observations and inference; test 7 calculates a controller response with physical sensors and motors off; tests 8–9 inspect tilt response/filter lag; test 10 spins motors; tests 11–13 inspect filter behavior, loop timing and barometer-induced command noise; test 14 automatically performs an all-motor throttle staircase from 1100 to 1700 microseconds. Its boot banner still says motors are off. The Makefile warning was corrected, but the harness itself remains unsuitable for unattended passive measurement.

Test 7 is not a simulator connected to the board's motor outputs. Its calculations also have separately maintained constants, including an altitude derivative gain different from `main.c`. A passing harness label is not proof that the exact flight loop passed. A genuine hardware-in-the-loop setup runs flight code on the target against a simulated plant; see the [PX4 description of HITL](https://docs.px4.io/main/en/simulation/hitl). That reference explains the technique, not an integration already present in this repo.

**4. Ways it can fail outside the viewer, and the experiment that would expose each.**

| Failure source | Specific evidence or uncertainty here | Discriminating test and likely adjustment |
| --- | --- | --- |
| Different policy artifact | Viewer selects a ZIP; firmware compiles the root header. Board contents are unverified. | Preserve checkpoint/header/binary hashes, run actor parity, and identify the running build at boot. |
| Missing horizontal information | Simulation supplies true world-frame horizontal velocity; firmware always supplies zero in observations 10 and 11. | Evaluate the candidate with those channels zeroed. Train with the actual available observations, or implement and model a real velocity estimator. Keeping 12 inputs and masking two is possible without increasing network size. |
| Different altitude controller | Simulation normally targets 1 m with NN altitude control. Firmware targets 0.3 m, clamps normalized altitude error to ±0.05, and adds PD correction capped at ±0.10 command. | Run these exact semantics in simulation; isolate target, clamp and PD changes individually, then test the combined deployed design. |
| Different action filtering | Training EMA alpha is 0.3. Firmware uses 0.08 or 0.60 depending on a 0.15 residual threshold. | Compare actual C filter sequences and closed-loop recovery, not only static actor outputs. Select one validated filter contract for both paths. |
| Attitude estimate differs from truth | MuJoCo attitude plus white jitter substitutes for a real estimator. White noise does not reproduce estimator delay, drift or vibration sensitivity. | Replay timestamped physical IMU through actual C AHRS against known fixture angles, then couple that estimator to simulated sensors. |
| Axis/sign/scale mistake | Alignment prose says identity; code negates accelerometer Y and gyro X. | Measure six stationary orientations and known positive/negative rotations. Verify right-hand conventions, sensor scale, estimator sign, and resulting corrective motor moment end to end. Do not fix signs by guessing from comments. |
| Calibration erases a real tilt | Startup subtracts the measured acceleration offset from expected level gravity without a stationarity/level acceptance gate. | Repeat calibration level, tilted and moving with outputs inhibited. Reject invalid calibration instead of absorbing orientation into bias. |
| Wrong thrust mapping | Simulation assumes thrust is linear in command with 8.5 N maximum per motor. Actual PWM, deadband, propeller loading and ESC behavior are unmeasured here. | Measure force versus PWM for each installed propulsion unit and battery condition; model a measured curve and uncertainty. |
| Motor delay and asymmetric response | Software EMA does not represent ESC/motor spin-up, spin-down, transport delay or desynchronization. | Measure bounded steps on a suitable stand; estimate onset delay and 10–90% rise/fall times at several operating points. Add separate actuator dynamics. |
| Mass, CoG, inertia or arm geometry wrong | 1.118 kg and diagonal inertia (0.018, 0.018, 0.035) are repository nominal assumptions, not recent measurements. Battery mounting changes CoG/inertia. | Weigh the complete aircraft, measure motor coordinates and CoG, estimate inertia with a documented method, and record uncertainty. |
| Motor order, rotation or prop installation wrong | Code maps NN FL/FR/RR/RL to driver indices 3/2/1/0. Actual wiring and physical prop orientation need confirmation. | Explicit props-off single-motor tests and rotation checks, followed by separately controlled loaded thrust-direction verification. |
| Unsupported ESC signaling | Firmware emits 1000–2000 microseconds at 400 Hz; ESC firmware/configuration is not interrogated. | Verify the exact ESC accepts that protocol and range; measure pulses. Do not assume DShot settings or a calibration procedure transfer to PWM. |
| Vibration and aliasing | Real IMU configuration and mounting differ from added independent simulation noise; the old flight log is too slow to resolve control-band vibration. | Acquire regularly timestamped sensor data at sufficient rate, with appropriate anti-alias filtering. Compare stationary, props-off and loaded-stand conditions separately. |
| Barometer false altitude or delay | Real pressure is affected by airflow, temperature/reference drift and filtering. Firmware reads at nominal 25 Hz and filters differentiated altitude. | Measure known vertical movements and fixed-height motor-on pressure changes against an independent height reference. Model measured delay/noise and handle freshness. |
| Control-loop stalls | Default TEL messages use blocking UART. SPI waits are unbounded; timing calculations assume 10 ms. | Measure acquisition, compute and PWM commit times, actual tick interval and worst-case stalls. Make logging nonblocking and add bounded sensor operations. |
| Frozen but plausible sensors | Finite old values can continue through the loop; barometer read errors are ignored in `main.c`. | Inject repeated/stale samples and bus failures in a controlled test. Track sample age, validity and fault reason; define responses before flight. |
| CPU hard fault or hang | Default fault handlers loop forever; no complete watchdog/output-timeout design is present. Hardware timers can retain their last PWM setting. | With props removed, verify output behavior under injected hangs/resets. Add tested watchdog/output inhibition and an independent emergency-stop mechanism suitable for the rig. |
| Battery sag, brownout or electrical interference | There is no implemented battery voltage/current monitoring or corresponding flight policy. | Measure supply and loaded voltage, resets and motor response across battery conditions; add monitoring and model thrust variation. |
| Takeoff, landing and ground effect | Normal training starts airborne at commanded altitude; the physical aircraft starts supported on the ground. | Add actual startup/ramp/contact and takeoff/landing scenarios after hover is consistent. Evaluate near-ground aerodynamics and barometer disturbance separately. |
| Horizontal drift or yaw behavior outside the task | No position target exists; yaw-rate commands do not enter the policy's observation/reward. | Define attitude/altitude stabilization as the first milestone. Treat position hold and commanded yaw as explicit additional capabilities. |
| Limited robustness or overfitting | Nominal visualization is easier than the full suite; all randomized-hover cases failed acceptance. | Sweep one parameter family at a time, retain failures, then test combined measured uncertainty and unused seeds. Report worst cases and pass counts, not just average reward. |
| Fixture, weather or structural effects | A restrained drone has different translation/rotation, airflow and contact dynamics from free flight; flex and loose hardware are not established in the model. | Document the fixture, inspect hardware, simulate the constraints when comparing traces, and widen conditions only after each narrower test passes. |

Motor/ESC thrust nonlinearity and the value of measuring the specific motor/ESC/battery combination are also documented in [ArduPilot's thrust-scaling guide](https://ardupilot.org/copter/docs/motor-thrust-scaling.html). Its parameters are not parameters for this firmware; use the measurement principle rather than copying its defaults.

These are failure categories to close with evidence, not a claim that every one is occurring on your aircraft. Several are already proven code mismatches. Others require measurements. No finite checklist proves behavior under every conceivable condition.

**5. A staged return to testing.**

**Stage A — establish a reproducible computer baseline.** Preserve the selected ZIP, current weight header, source changes and experiment report. Repeat the appropriate host/simulation checks when their code changes. Freeze a versioned definition of observations, action mapping, filter, altitude controller and sensor timing. Put the exact C controller in the simulated loop before treating a policy-only pass as a deployment result. The existing `test_firmware_sim.py` is a useful diagnostic approximation, not that complete C implementation.

Test unavailable horizontal velocity first. Then evaluate deployed filtering and altitude handling, both separately and together. For the actual end-to-end contract, use the real AHRS/observation/filter/safety code, modeled sensor sampling, and measured actuator behavior. Retrain after selecting this consistent contract. A larger actor is not the first remedy for these mismatches.

**Stage B — make a truly passive firmware mode.** This is required implementation work, not an existing `make BENCH=1` command. It must boot disarmed, skip the motor-identification sequence, prohibit all automatic arming and motor tests, and remain inhibited while computing/logging proposed commands. Add an explicit state machine, prearm checks, operator arm/disarm/stop, bounded test duration, sensor freshness, deadline/fault reasons, and a validated output stop path. Split powered harness tests into explicit selections. Correct the misleading mode comments at the same time.

On the unpowered aircraft, inspect connectors, polarity, motor screws, propellers, battery condition and mount security. Verify the stand's load capacity and independent disconnect arrangement; having a stand does not establish that the particular fixture is adequate. For initial electronics tests, remove all props and leave propulsion power disconnected. Confirm whether your board's USB supply powers the required sensors using the exact board wiring; avoid accidental backfeeding between supplies.

**Stage C — connect telemetry and validate sensors with actuation inhibited.** Record detected IMU/barometer IDs and configured ranges/filter registers. Hold level and still for accepted calibration. Record a stationary interval, known stationary tilts, controlled rotations and known height changes. Check units, signs, bias, drift, quaternion validity, sensor freshness, altitude reference and actual sampling intervals. Pass requires agreement with independent references within predeclared tolerances; a plausible plot alone is insufficient.

A rigidly supported aircraft with the flight controller demanding takeoff cannot respond to those commands. Nonzero or saturated proposed motor outputs on that fixture do not, by themselves, prove an NN defect. Compare observations and response to the fixture's actual constrained conditions.

**Stage D — validate outputs and fault behavior with props removed.** Use explicit bounded single-motor commands, not the current auto-start routine. Confirm motor identity and direction, pulse period/range, startup deadband, stop and reset behavior. Exercise the arm/stop path, a latched tilt fault, simulated invalid/stale sensor data, telemetry disconnect, command-link timeout and injected loop overruns. Check PWM on a scope/logic analyzer as well as the software log. A keyboard stop on the laptop is not the independent emergency stop for a powered stand.

**Stage E — characterize loaded propulsion on the appropriate stand.** Use a guarded, rated measurement setup, controlled commands and an independent disconnect. The existing test-14 strap/scale narrative and its 40-degree abort threshold are not an adequate test plan to follow blindly. Do not hold the aircraft or improvise a hand-held loaded test. Remove/fit props only with propulsion power disconnected.

Measure force versus PWM, repeatability, battery voltage/current, spin-up/down delays, vibration and barometer disturbance. Include multiple operating points near the measured hover region and permitted battery conditions. Record which motor/ESC/prop assembly each dataset describes. Do not treat a software motor command as measured thrust, or a props-off response as the loaded propeller response.

**Stage F — return the measured aircraft model to simulation.** Fit parameters using part of the physical data, and validate predictions on different recordings. Bound randomization with measurement uncertainty and operating variation. Include the fixture if comparing restrained experiments. Repeat exact-controller recovery, command tracking, start/stop, fault, low-voltage, sensor delay and actuator mismatch cases. Keep the original fixed seeds for comparisons and unused seeds for generalization.

**Stage G — restrained controller tests, then an explicitly reviewed flight test.** A stand allowing selected axes can establish the sign and stability of particular control responses, but cannot prove all-axis free flight. Model its changed inertia and constraints. Expand one degree of freedom or operating condition at a time, keep a short predeclared duration, log every attempt and stop on the first unexplained instability. Free flight comes after tested pilot control/stop behavior, a coherent takeoff/landing plan, full-controller acceptance and measured actuator/sensor margins. There is currently no implemented pilot takeover mode to assume will recover the aircraft.

**6. Your existing telemetry wiring and tools.**

The current selection is `UART_PERIPH = 3` in [uart.h](../../uart.h). That configures USART3 TX on PB10 and RX on PB11, at 115200 baud, normally 8N1. Use the board's pinout to identify the pads corresponding to those MCU pins; labels printed on a board are not independently verified here.

| Connection for receiving logs | Purpose |
| --- | --- |
| FC PB10 / USART3 TX → adapter RX | Flight-controller telemetry to the laptop. |
| FC GND → adapter GND | Common signal reference. |
| Adapter TX → FC PB11 / USART3 RX | Not needed for receive-only logging; no command handler is implemented. |

Use a 3.3-V-compatible TTL serial adapter, not an RS-232 interface. Do not connect its supply pin or add another power source without checking your actual board/adapter power wiring. The laptop's USB connection to this adapter is the telemetry link; the FC's own USB DFU connection is the flashing link.

The fresh simulation environment contains NumPy and Matplotlib. At this review it **does not contain pyserial or SciPy**; those are needed for the logger and analyzer. The ARM compiler and `dfu-util` are installed. The previously used `/dev/cu.SLAB_USBtoUART` device was not currently enumerated; reconnect the adapter and discover its present name.

Computer-only setup commands, from the repo root:

```bash
cd /Users/sg/Documents/Projects/FC
FC_PY=/Users/sg/.cache/fc-sim-venv-20260912/bin/python
uv pip install --python "$FC_PY" pyserial scipy
"$FC_PY" -m serial.tools.list_ports -v
```

Port enumeration is a pySerial feature; see [its tool documentation](https://pyserial.readthedocs.io/en/latest/tools.html). The old cloud-backed `fc-nn/.venv` had import problems, so use the cache environment above.

Once the passive/explicitly controlled firmware setup has been verified, capture a uniquely named session. Substitute the actual enumerated port:

```bash
FC_SESSION="experiments/hardware-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$FC_SESSION"
FC_PORT=/dev/cu.SLAB_USBtoUART
"$FC_PY" fc-nn/log_imu.py --port "$FC_PORT" --baud 115200 \
  --seconds 45 --out "$FC_SESSION/telemetry.csv"
```

Only one program should open the adapter at a time. Start capture before the controlled reset/test so you can see startup messages. The existing logger prints non-TEL startup text but does not retain it in its CSV. Save terminal output separately when you need that evidence.

The current logger starts its duration clock at the first valid TEL record. If no valid record arrives, or records cease, it can keep waiting beyond the requested duration; Ctrl+C ends it. Its console progress prints every 50 records, approximately ten seconds at the current nominal telemetry rate. It is not a fast live alarm system. A sensor-only future format would need a corresponding parser update; the current logger expects exactly 28 TEL values.

Analyze a completed current-format capture:

```bash
"$FC_PY" fc-nn/analyze_imu.py "$FC_SESSION/telemetry.csv"
```

It generates raw-IMU-labeled traces, attitude plots, observation plots, NN/motor plots, command histograms and gyro PSD PNGs next to the CSV, then opens plot windows. Despite those labels, the currently transmitted IMU values have already been aligned and bias-corrected. It can overlay several compatible CSVs supplied as positional arguments. It cannot directly consume the older six-channel `imu_static.csv` format or the simulation NPZ trajectories. Check sampling and format before drawing conclusions from a PSD or overlay.

The previous compatible flight capture and its existing figures remain in `fc-nn/flight_test.csv` and `fc-nn/flight_test_*.png`. Those are historical evidence; exact present hardware configuration and firmware provenance are not established by their filenames.

**7. What the existing log can and cannot tell you.**

| Existing fields | Interpretation and limits |
| --- | --- |
| `t_s` | Laptop monotonic receive time relative to the first parsed record, not MCU acquisition time. |
| `gx_rads`…`az_g` | Aligned, bias-corrected IMU values used by this controller; not untouched sensor register counts. |
| `roll_deg`, `pitch_deg` | AHRS estimates, not external attitude truth. |
| `obs_0`…`obs_11` | The 12 values actually supplied to the NN at that logged tick. Horizontal-velocity slots are zero. |
| `nn_0`…`nn_3` | Actions after clipping and adaptive smoothing. These are not raw actor outputs. |
| `mot_0`…`mot_3` | Requested final normalized commands in FL/FR/RR/RL order. Nominal pulse conversion is `1000 + 1000*command` microseconds. They are not measured PWM, RPM or force. |

The normalized altitude error is clamped to ±0.05. Multiplying `obs_8` by 2 and adding the 0.3-m target only reconstructs estimated altitude while that field is unsaturated, approximately between 0.2 and 0.4 m. Outside that range the magnitude is lost. The current TEL format omits actual altitude, raw pressure, the altitude PD term, reference state and battery voltage. It therefore cannot support a reliable real-versus-sim altitude RMSE on its own.

There are no MCU timestamps, sequence numbers, raw actor actions, actual selected alpha, fault reasons or drop counters. After a latched fault, the loop can skip the TEL block, so silence does not distinguish a cutoff from a broken cable. A requested motor command in a log does not prove the output signal or physical motor response.

`RAW_LOG_MODE` currently defaults to 1 and controls the TEL block. It does **not** disable motors. Its old comment promising safe raw-only mode is false for the present implementation. `DEBUG=0` does not disable this TEL stream. The comments suggesting `make CFLAGS_EXTRA=...` are also stale: the Makefile does not consume `CFLAGS_EXTRA`. Do not assume a command-line flag created a passive build; inspect and test the resulting mode.

The divider is 20 at a requested 100-Hz loop, so TEL is nominally **5 Hz**, not the stated 10 Hz or 100 Hz. At 115200 baud with 8N1, a roughly 230-byte line takes about 20 ms to transmit, twice the intended 10-ms control period. `uart_write_char()` busy-waits. Average bandwidth fitting the link does not prevent each transmission from delaying control. SysTick's polled flag does not count every missed tick, and the estimator still receives a fixed 0.01-s timestep. Measure this on target; do not infer exact achieved timing from the nominal divisor.

**8. The monitoring improvement I would implement before the next powered experiment.**

Use bounded nonblocking transmission with a sequence counter and explicit dropped-record count. Add a boot record identifying firmware revision/source hash, model hash, observation/action schema, control mode, sensor IDs/configuration and calibration state. Continue state/fault reporting when actuation is inhibited.

For each control tick, retain hardware acquisition time, actual elapsed time, sensor age/validity, estimator attitude, command setpoints, raw height/vertical-velocity estimates, all observations, raw/clipped/filtered actions, altitude correction, requested/final PWM, output state and fault reason. Also measure battery voltage/current and record them at an appropriate rate if the hardware supports it. High-rate IMU capture should be a separate defined mode with its own bandwidth budget.

All those fields as floats will not fit at 100 Hz over the current link. Use a compact selected control record, for example 64–80 bytes at 100 Hz, and decimated detailed records, or measured higher baud/local storage. At 8N1 the theoretical capacity is 11,520 bytes/s; a full 28-float payload is already 11,200 bytes/s at 100 Hz before framing/timestamps. Leave headroom, record overruns/drops and verify worst-case control timing with logging enabled. Nonblocking transmission is necessary even at a low average data rate.

A useful live view would show armed/inhibited/fault status, log age, roll/pitch versus targets, height versus target, four PWM outputs, clipping, battery voltage and timing overruns. This dashboard does not yet exist. The existing analyzer is for review after capture; it must not be mistaken for a pilot stop path.

**9. Measure real versus simulated at three different levels.**

First compare **the same observation through the same actor**. Replay recorded observations through the selected PPO actor, exported NumPy arrays, actual C and, when available, target-MCU inference. Compare raw outputs before clipping/smoothing. Existing post-filter 5-Hz records cannot reconstruct missing 100-Hz filter state exactly. A successful arithmetic comparison says nothing about whether the observations were physically correct.

Next compare **the same timestamped sensor/command sequence through the complete controller**. Reset calibration, AHRS, filter, ramp and fault states consistently. Compare each intermediate stage to locate the first disagreement. Ideally the same C controller core serves firmware and simulation, so the comparison focuses on drivers, timing and sensor models instead of duplicated implementations.

Finally compare **the same bounded physical experiment against the plant model**. Replay measured motor commands and actual timing into the simulator with the measured mass, actuator curves, initial state, battery condition and fixture constraints. Compare short-horizon motion/force predictions against independent physical measurements. Open-loop prediction separates plant error from policy reaction; long open-loop drift alone does not identify the cause. Then compare closed-loop performance distributions under matched tasks and disturbances. Stochastic real and simulated trajectories need not match sample for sample.

Do not grade AHRS against itself. Use known fixture angles or an independent calibrated visual reference for attitude; a measured rail/position reference, suitable ranging sensor or calibrated camera for height; and a scale/load cell for force. Record reference uncertainty and delay. A barometer can report that altitude is perfect while the actual aircraft moves. Synchronize clocks with a timestamped event or visible marker, document offset/drift, and align physical events rather than simply lining up sample indices.

| Quantity | Report | Why it matters |
| --- | --- | --- |
| Attitude tracking | Bias, RMS, peak error, overshoot, sustained settling time, rates | Separates steady offset, oscillation and inadequate recovery. |
| Altitude tracking | RMSE, peak loss/overshoot, percentage within band, vertical velocity | Detects sag and dangerous tails hidden by averages. |
| Horizontal motion | Maximum displacement and speed | Enforces the available test area even though position hold is not implemented. |
| Actuation | Per-motor PWM range, time saturated, command rate, force/RPM lag | Detects lack of authority, chatter, mapping problems and propulsion mismatch. |
| Estimation | Error against reference, bias drift, sensor age and clipping | Separates wrong perception from wrong control. |
| Timing/logging | Median/p95/p99/max tick interval, compute time, sample age, missed deadlines, sequence gaps | Exposes intermittent stalls and missing evidence. |
| Power/faults | Loaded voltage, current, resets, first fault reason, stop latency | Relates behavior to electrical limits and verifies fault handling. |
| Robustness | Full-duration completion, every-criterion pass count, worst case across repetitions | Keeps crashes and difficult cases visible. |

For a scalar reference error `e`, RMSE is `sqrt(mean(e*e))` on uniformly spaced samples. With irregular timing, use a documented time-weighted integration or resample appropriately; do not give stalled intervals equal weight just because they contain one record. Report completed duration and failure separately. Compare the same target, duration, environment and fixture, and keep test data separate from model-fitting data.

The current simulation thresholds are useful declared benchmarks, not validated aircraft safety limits. Select physical abort thresholds from the stand's permitted travel/forces, available authority and measured stop latency before testing. Do not inherit the current 60-degree flight cutoff or test-14 40-degree threshold as appropriate limits for a constrained rig.

**10. How deployment works once those prerequisites are met.**

The build/flash chain already exists: selected `.zip` → exported actor header → compile `nn.c` plus the full firmware → `.elf` → `.bin` → STM32 DFU at flash address `0x08000000`. The current Makefile now tracks header and flag changes, fixing the old stale-build problem.

A computer-only candidate preparation sequence, after the shell setup above, is:

```bash
FC_MODEL=experiments/2026-09-12-fixes-1-6/training_seed43/best_model.zip
cp weights.h "$FC_SESSION/weights-before.h"
git diff --binary > "$FC_SESSION/source.patch"
"$FC_PY" fc-nn/export_weights.py "$FC_MODEL" \
  --out "$FC_SESSION/candidate-weights.h"
"$FC_PY" host_test/test_candidate_parity.py "$FC_MODEL" \
  --out "$FC_SESSION/candidate-parity.json"
shasum -a 256 "$FC_MODEL" "$FC_SESSION/candidate-weights.h"
```

This stages a header and checks actor parity; it does not put that header in the build or validate the flight loop. Also preserve new/untracked source files in your source snapshot: `git diff` alone omits them. The exporter hard-codes the hover/action constants, so verify those constants against the selected controller/plant contract whenever that contract changes.

Only after the passive/explicit-control mode and intended test configuration have been implemented and verified, promote the reviewed header into an isolated build checkout or the backed-up root `weights.h`, then build there. The existing commands are `make` for normal firmware and `make test` for the harness; **both existing images can operate motors**. `make ESC_CAL=1` creates the special ESC calibration image, not a normal test image. It must not be selected accidentally, and any ESC calibration must follow the exact ESC's supported procedure.

For a deliberately prepared firmware image, `make TARGET="$FC_SESSION/firmware"` keeps the build artifact in the session folder. Record the effective flags, source snapshot, header and binary hashes, compiler version and `make size` output. A successful link does not measure MCU timing or prove the correct binary is running.

The Makefile's `flash` target builds and invokes `dfu-util -a 0 -s 0x08000000:leave -D <binary>`. `flash-test` selects the harness. Entering DFU is board-specific; confirm the board's boot-button/pad procedure and the device identity before selecting a download. The `leave` suffix requests leaving DFU, so expect the application to start as part of deployment. With props removed and propulsion power disconnected, use `dfu-util -l` to inspect the intended board in DFU. Do not connect a powered propeller setup just to discover what an unknown existing binary does.

After deliberate flashing, confirm the boot build/model identity, mode, sensor detection and inhibited outputs. Until boot identity and explicit modes are implemented, that verification is itself incomplete. Keep a known bench image and its source/header with a documented recovery procedure. Do not use the filename `working` as evidence of which firmware is presently on the board.

**11. The next work session I recommend.**

First implement passive boot, explicit powered-test selection, pilot stop/arm state and nonblocking timestamped telemetry. In parallel as separate work, align the full firmware observation/filter/altitude path with simulation, starting with the two unavailable velocity channels. Then use your existing adapter and stand to collect stationary/known-motion data with actuation inhibited, verify outputs props-off, measure loaded propulsion, and return those measurements to the simulator. Only promote a model after that complete controller passes the defined tests.

The outputs of that session should be a named firmware build, a model/header hash, a verified mode, a documented physical setup, a timestamped recording with a usable reference, plots/metrics, and one explained next adjustment. This turns returning to the drone into a reproducible experiment instead of another unrecorded trial.

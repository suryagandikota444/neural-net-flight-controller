**Flight-controller project review — September 12, 2026**

The project has a complete experimental pipeline: train a small neural controller in MuJoCo, inspect its behavior, export its actor weights, and run inference in bare-metal STM32F405 firmware. The current best checkpoint has useful altitude-hold behavior in the existing simulator. The next milestone should be a physically correct, repeatable simulation with the same controller and observations as the firmware. More training is premature until those contracts are fixed.

This review covers the source tree, build rules, all nine commits, current uncommitted work, checkpoint metadata/tensors, TensorBoard logs, hardware telemetry, and the available test tools. Firmware and test-harness builds were checked in a temporary directory. No hardware was connected or flashed. Simulation, firmware, weights, and existing tests were left unchanged. The files in this review directory are new review artifacts.

The history establishes the following progress:

| Date | Evidence and milestone |
| --- | --- |
| February 22 | Initial neural-controller/firmware work; simulation adapted to the F450, 1400 KV motors, 8045 props, nominal 1.118 kg aircraft. |
| February 28 | Commit `cc0c795`, “end to end pipeline working”: motor driver, AHRS, barometer and inference integrated. This is a historical milestone label, not proof of stable flight. |
| March 2 | Commit `ffd11a0`, “working with altitude”: altitude-control work and the older `working_w_pulse` checkpoint family. |
| March 7 | Commit `96b10f0`, component test harness. The latest checkpoint archives were also created March 7. |
| July 11 | Latest commit `675cbac`, titled simply “commit”: curriculum training, expanded hardware diagnostics, IMU bias calibration, lower altitude derivative gain, full telemetry and shared firmware-simulation utilities. Many included files contain March-era work. |
| July 11–12 working files | Uncommitted Python physics/parity/firmware-simulation tests, C host test, Makefile host-test target, and hardware test 14. These are the most recent unfinished testing work visible here. |

At review start, `Makefile` and `test_harness.c` were modified; `fc-nn/test_env_physics.py`, `test_nn_parity.py`, `test_firmware_sim.py`, `host_test/`, `test.bin`, and `test.elf` were untracked. There were also modified `.DS_Store` files. Preserve the useful testing work before reorganizing the project. There was no README, dependency manifest, CI configuration, or durable record of hardware-test passes. `screenlog.0` is empty. Despite older commits about untracking artifacts, checkpoint ZIPs, TensorBoard logs, several binaries and `.DS_Store` files are currently tracked.

The code is organized as follows:

| Area | Files | Current behavior |
| --- | --- | --- |
| Physics and task | [drone_env.py](../../../fc-nn/drone_env.py), [quadrotor.xml](../../../fc-nn/quadrotor.xml) | MuJoCo at 4 kHz; actions at 100 Hz; 10-second training episodes; randomized thrust, mass, CoG and wind; attitude/altitude commands; curriculum. |
| Learning | [train.py](../../../fc-nn/train.py) | PPO with 11 environments in the saved runs, 4096-step rollouts, 512 batch size, ReLU actor `12 → 32 → 32 → 4`; 1,604 actor parameters. |
| Inspection/export | [main.py](../../../fc-nn/main.py), [viewer.py](../../../fc-nn/viewer.py), [validate.py](../../../fc-nn/validate.py), [export_weights.py](../../../fc-nn/export_weights.py) | Command dispatcher, interactive viewer, checkpoint tests, C weight export. |
| Flight controller | [main.c](../../../main.c), [nn.c](../../../nn.c), [ahrs.c](../../../ahrs.c), [baro.c](../../../baro.c), [motor.c](../../../motor.c) | 100 Hz loop; sensor alignment/calibration; Madgwick attitude; barometric altitude/vertical velocity; NN; adaptive output filter; altitude PD supplement; PWM motor outputs. |
| Board support | `startup.c`, `linker.ld`, `clock.c`, `spi.c`, `i2c.c`, `imu.c`, `uart.c`, corresponding headers | Bare-metal startup and FPU enable, 168 MHz clock, sensor buses, IMU detection/configuration, blocking UART, 400 Hz PWM. |
| Diagnostics | [test_harness.c](../../../test_harness.c), [host_test](../../../host_test), [fw_common.py](../../../fc-nn/fw_common.py), three Python test files | Hardware tests 1–14; actual C NN parity; physics sanity; approximate firmware-controller rollouts. |
| Hardware evidence | [log_imu.py](../../../fc-nn/log_imu.py), [analyze_imu.py](../../../fc-nn/analyze_imu.py), CSVs and PNGs | Static IMU and flight-test telemetry capture/analysis. |

The deployed observation layout is gyro XYZ, accelerometer XYZ, roll/pitch error, altitude error, vertical velocity, and two horizontal-velocity slots. The firmware fixes the final two slots at zero. Actions are motor residuals around approximately 0.3226 throttle, with scale 0.4. Motor order is FL, FR, RR, RL; the firmware maps these to motor indices 3, 2, 1, 0.

The saved models have identifiable provenance:

| Checkpoint | Steps recorded in archive | Export relationship |
| --- | ---: | --- |
| `runs/quad_hover/best_model.zip` | 14,498,550 | Exactly matches all actor tensors in root `weights.h`. |
| `runs/quad_hover/final_model.zip` | 40,009,728 | Latest completed run; differs from the exported model. |
| `runs/quad_hover_working_w_pulse/best_model.zip` | 4,099,590 | Exactly matches `weights_working_w_pulse.h`. |
| `runs/quad_hover_working_w_pulse/final_model.zip` | 20,004,864 | Useful older comparison checkpoint. |

The latest log, `PPO_9`, reached curriculum progress 1.0 at 16,084,992 steps. Its final rolling crash rate was 40.5%, with 42% crashes in the last rollout. Its highest evaluation reward was 1323.49 at the current best checkpoint; final evaluation reward was 1066.08. These are different difficulty distributions: `CurriculumCallback` advances only the training environments, leaving the evaluation environment at curriculum 0. The logs therefore do not establish mastery of the final task. Earlier logs also vary substantially; model names alone are not reliable quality rankings.

Fresh controlled rollouts clarify what currently works. These used all four archived actor networks, five seeds (100–104), fixed 1 m altitude/zero attitude setpoints, fresh environments and filter state, nominal plant parameters, and the existing observation noise. Random command changes were disabled. The physics itself was deliberately left as written, including its mass discrepancy. Float32 actor tensors were read from the archives and evaluated through `fw_common.nn_forward_np`.

| Checkpoint | 30 s survival | Mean altitude RMSE | Time within ±5 cm | Combined roll/pitch RMS | Worst horizontal displacement |
| --- | --- | ---: | ---: | ---: | ---: |
| Current best | 5/5 | 2.36 cm | 99.2% | 1.26° | 4.10 m |
| Current final | 5/5 | 65.20 cm | 40.5% | 16.39° | 3.11 m |
| Older best | 5/5 | 6.39 cm | 3.3% | 0.54° | 2.21 m |
| Older final | 5/5 | 2.34 cm | 99.5% | 1.35° | 3.19 m |

The current best also survived all five 10-second, 0.5 N wind tests, with 3.82 cm mean altitude RMSE. It first entered the ±10° attitude band after 0.17 s in each 30° roll-recovery test, but altitude RMSE over those two seconds was 24.44 cm. Merely crossing the attitude band once is an inadequate recovery criterion. The older final is another worthwhile baseline; its nominal hover is comparable, but its tilt-test altitude RMSE was worse at 43.01 cm. These small controlled samples are diagnostic, not evidence of broad reliability.

The full individual results are in [policy-results.json](policy-results.json). [run_baseline.py](run_baseline.py) repeats this 60-episode comparison without changing project source or checkpoint files. It uses [checkpoint_arrays.py](checkpoint_arrays.py), a restricted reader for the existing float32 checkpoint format, to avoid importing Torch. This is a review benchmark, not a replacement for a corrected project validation suite.

The highest-priority findings are:

1. **The nominal aircraft mass and hover calculation disagree.** `quadrotor.xml:62` assigns 1.118 kg to the central body, then the four motor bodies each add 0.055 kg at lines 85, 98, 110 and 122. The dynamic aircraft subtree weighs 1.338 kg. `drone_env.py:76` computes hover from 1.118 kg, so nominal throttle is 0.322576 instead of the current model's equilibrium 0.386052. The physics test fell from 1 m to approximately 0.008 m in two seconds. Its accelerometer read 0.836 g at that throttle because the aircraft was descending. In isolated checks, either using the current model's correct hover command or reducing the parent mass by 0.220 kg restored stationary hover. Resolve total mass and inertia consistently against the intended aircraft; do not just compensate with another unexplained throttle constant.

2. **CoG randomization is currently ineffective.** `drone_env.py:308` writes `body_mass` and `body_ipos` directly. A requested CoG offset `[0.002, 0.001, 0]` produced an actual `xipos - xpos` offset of zero in the loaded model. The compiled `body_sameframe` optimization remained enabled. Disabling that optimization and refreshing derived constants made the offset take effect in an isolated check. Prefer a correctly rebuilt model or a validated runtime update path. MuJoCo documents the derived-state requirements for these edits in its [model-change documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#mjmodel-changes).

3. **Training and firmware implement different control problems.** Training uses EMA alpha 0.3; firmware uses 0.08/0.60 with a 0.15 switch threshold (`drone_env.py:364`, `main.c:812`). The host test measured 280 ms versus 70 ms to settle a small correction, and 80 ms for a larger firmware step where the test expects at most 60 ms. Training gives the network true horizontal velocity, altitude and vertical velocity; firmware supplies zeros for horizontal velocity, clamps normalized altitude error to ±0.05, uses barometer-derived velocity and adds an altitude PD loop (`main.c:733`). Training obtains attitude directly from MuJoCo plus Gaussian jitter; that does not implement the estimator's dynamics. Use the same observation construction, filtering, action mapping and altitude-controller choice in training and testing.

4. **The old validation scripts can mislead.** `validate.py:129` does not disable random setpoints, and its custom reset omits `_smooth_action` and setpoint resets. The nominal hover scenarios can start issuing random commands every two seconds and contaminate subsequent scenarios. The printed PWM conversion at line 49 still uses the old direct-action formula instead of residual mapping. Tilt recovery passes after any single sample within ±10°, and hover/wind checks ignore horizontal displacement. Seed scenarios, reset all controller/task state, check full-duration completion, and require sustained recovery and explicit altitude/attitude bounds.

5. **The firmware simulator is an approximation, and its “A baseline” is not training conditions.** `test_firmware_sim.py` calls a Python replica, not the compiled controller. All scenarios use the barometer approximation and zero horizontal-velocity slots; even A differs from training. Startup bias calibration and motor ramp are omitted. The AHRS “ground settle” repeatedly reads sensors from a stationary simulation state located in the air with zero thrust. A→B changes several factors at once; F does not retain E's thrust deficit. Stalled runs report controller-tick time instead of physical elapsed time. The script returns success even when episodes fail or hover errors are large. Its printed causal diagnoses are hypotheses, not established causes. Build a true baseline, change one factor per test, log elapsed physical time and failure reasons, and add machine-checkable criteria.

6. **Curriculum progress, evaluation and experiment provenance need repair.** The separate evaluation environment never advances from curriculum 0 (`train.py:157`, `209`). Resume constructs a new callback at progress 0 and calls `learn` with default timestep-reset behavior. Checkpoints do not save curriculum state or the physics/observation configuration that produced them. Save a configuration snapshot, seed, code revision, checkpoint hash and curriculum state; evaluate a fixed suite covering easy and hard cases. Existing filenames such as `best_model` should be interpreted within their original selection procedure.

7. **Yaw commands and position holding are unfinished.** `_cmd_yaw_rate` is written but never used in observations or reward (`drone_env.py:283`, `450`, `490`), so the viewer's yaw controls cannot command the policy as advertised. The task has velocity penalties but no horizontal-position target/feedback or lateral boundary; this explains how altitude tests pass while the drone travels metres. Decide whether the first milestone is attitude/altitude stabilization or position hold. Position hold would need a suitable observation/estimation and control design. The “diagonal” symmetry term also compares left versus right motors (`drone_env.py:530`), so its implemented purpose differs from its comment.

8. **Hardware noise evidence needs better interpretation.** `flight_test.csv` contains 211 samples over 44.90 s at roughly 5 Hz, with roll/pitch standard deviations 7.65°/9.30° and average motor command around 1374 µs. The test's first-difference estimate is 0.465 rad/s gyro and 0.862 g acceleration, versus training noise of 0.09 rad/s and 0.22 g. That is substantial variability, but a 5 Hz log can mix actual movement with aliased vibration; it does not identify a high-frequency sensor-noise model. `imu_static.csv` has 4491 samples over 45 s at approximately 100 Hz. Preserve these captures, distinguish static noise from motion and motor-induced vibration, and obtain timestamped sensor/control-rate evidence before choosing noise/filter parameters. The current analyzer also expects the newer 28-channel TEL CSV and cannot directly consume the old six-channel static capture.

There are additional firmware findings to resolve before returning to hardware. They are concrete issues in the current code, not simulation-test results:

- `main.c:704` calls `motor_all_stop()` on excessive tilt, but the following `continue` is commented out. The same loop iteration proceeds to write active motor commands again.
- `ahrs.c:341` normalizes a zero gradient without a guard. At exact level rest, one update with gyro `[0,0,0]` and accel `[0,0,1]` produced NaN roll and pitch in a host-adapted test; only the ARM square-root instruction was replaced with `sqrtf`. The Python Madgwick replica has a guard and therefore hides this defect. The Python/C implementations also differ in numeric precision and angle conversion. Include stationary and finite-value parity tests before trusting noise-free estimator simulations.
- `main.c:111` enables raw telemetry by default, independently of `DEBUG`; UART transmission is blocking. The telemetry divisor is 20, yielding 5 Hz at a 100 Hz loop despite comments saying 10 Hz. A roughly 230-byte line takes about 20 ms on a 115200-baud link, exceeding one loop period; timing still assumes a fixed 10 ms. `DEBUG=1` also does not currently enable the promised 2 Hz debug output because the source default remains zero.
- Normal firmware unconditionally runs a startup motor-identification sequence at 50% command (`main.c:562`) and then automatically arms. The test-harness build messages claim motors are disabled, but tests 10 and 14 operate motors; test 14 runs automatically after a countdown (`test_harness.c:3517`). The harness is not a passive bench-only binary.
- Axis-alignment prose says identity while macros negate accel Y and gyro X (`main.c:286`). The Python plant cannot validate actual mounting/wiring conventions. SPI/UART waits are unbounded, and there is no comprehensive sensor-health/finite-value failsafe in the control path.
- Firmware build targets depend on `.c` files, not `weights.h`, other headers or `linker.ld` (`Makefile:84`, `100`). Exporting new weights can leave an existing firmware binary considered up to date. Changes to build flags likewise do not trigger rebuilds. Golden-vector generation only checks whether its header exists. Action constants are duplicated in `nn.h`, `weights.h`, Python and the export script.

The existing firmware-style scenario ladder ran with five seeds and a 30-second target duration:

| Scenario | Survived | Mean altitude RMSE | Combined attitude RMS |
| --- | ---: | ---: | ---: |
| A, labeled training baseline | 5/5 | 0.45 m | 19.2° |
| B, firmware control path | 5/5 | 0.50 m | 15.8° |
| C, plus Madgwick | 4/5 | 0.30 m | 32.5° |
| D, measured-variability noise | 5/5 | 0.42 m | 23.7° |
| E, thrust deficit | 2/5 | 0.17 m | 22.3° |
| F, loop stalls | 5/5 | 0.40 m | 25.8° |

These values include terminated runs' shorter trajectories, and the scenarios have the limitations described above. In particular, 5/5 survival does not imply acceptable hovering. The current best policy's good direct-simulation performance and poor approximate firmware-path performance make consistency work a high-value next step.

The checks completed during this review were:

- Fresh ARM firmware build: successful, 16,488-byte binary. Test-harness link: successful, 59,376 bytes of text plus 396 bytes of initialized data. Builds have warnings about unused diagnostics, array-initializer braces and RWX load segments; neither is warning-clean.
- Actual `nn.c` on the host: all 17 golden observations matched NumPy within maximum absolute error `1.43e-6`. Existing golden vectors were independently compared to freshly evaluated `weights.h` and matched exactly. Two filter-behavior assertions failed.
- Layer 0 Python parity checks: passed using the archive-array loader in place of SB3 loading. The ideal-hover output has a large asymmetric residual (largest approximately 0.705); this is a diagnostic observation, not proof the policy cannot stabilize in feedback.
- Layer 1 physics: four failed assertions—hover equilibrium, hover accelerometer magnitude, and two noise-coverage checks. Mixer signs and gyro response passed. The noise failures have the interpretation limits above.
- Layer 2 firmware-style scenarios: completed, with results shown above.
- Controlled checkpoint comparison: all 60 episodes completed; full seed-level results are saved here.
- Ordinary Torch/SB3 CLI validation did not finish. The existing virtual environment has cloud-offloaded files marked `dataless`, and imports stalled in file reads. Those attempts were stopped. NumPy/MuJoCo, archive inspection and the direct actor tests did run. No packages were upgraded or installed.

To resume now, start with the reproducible baseline and physics tests from the repository root:

```bash
fc-nn/.venv/bin/python docs/reviews/2026-09-12/run_baseline.py
fc-nn/.venv/bin/python fc-nn/test_env_physics.py
make host-test
fc-nn/.venv/bin/python fc-nn/test_firmware_sim.py --runs 5 --duration 30
```

The baseline writes `policy-results-latest.json` beside this report and preserves the dated results. The physics and host-filter checks currently fail for the reasons above. The firmware-scenario script prints metrics but its exit status is not a quality gate. `make host-test` builds a native test executable; it does not flash hardware.

Restore the existing virtual environment's local files or recreate it outside a cloud-synced directory before resuming SB3 training/viewing. The installed core versions are Python 3.13.2, MuJoCo 3.5.0, Gymnasium 1.2.3, Stable-Baselines3 2.7.1, Torch 2.10.0, NumPy 2.4.2 and TensorBoard 2.20.0. Viewer/analysis packages include pygame 2.6.1, pynput 1.8.1, matplotlib 3.10.8, scipy 1.17.1, imageio 2.37.2, imageio-ffmpeg 0.6.0 and pyserial 3.5. Freeze a working environment instead of silently upgrading dependencies while comparing historical checkpoints.

Once those imports work, the existing visual and training entry points are:

```bash
cd fc-nn
.venv/bin/mjpython main.py view --model runs/quad_hover/best_model.zip
.venv/bin/tensorboard --logdir runs/tensorboard
```

On macOS, this passive viewer requires `mjpython`; see [MuJoCo's Python documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). Treat it as visual inspection, with the yaw-command limitation above. `main.py validate` needs the validation fixes before its verdict can be relied on. The viewer also samples recorded frames using the physics timestep even though its loop runs at policy frequency, producing roughly a 40× recording-time mismatch; its documented pause/reset/quit behavior is not fully wired into the custom loop.

I recommend continuing in this order:

1. Preserve the current best, older final, historical logs and uncommitted tests. Save their hashes/configuration and make the environment reproducible.
2. Correct total mass/inertia, derived model updates and CoG randomization. Require physics tests to verify hover, free fall, force/torque signs and actual effects of parameter changes. Separate pure physics checks from comparisons against telemetry noise.
3. Repair deterministic validation. Add stable hover, sustained tilt recovery, altitude steps, wind/gusts, long-duration drift, and failure reasons. Use identical fixed seeds for comparisons and separate held-out seeds for qualification.
4. Define one controller contract for the current sensors. For the present aircraft, begin with attitude/altitude stabilization using observations it can actually measure. Share output filtering and altitude logic between training and deployment; ideally expose the actual C controller/estimator to simulation instead of maintaining a separately hardened Python copy.
5. Add realism progressively: estimator and barometer dynamics, motor lag/deadband/nonlinear thrust, sensor bias/noise, timing variation, battery/thrust variation, then ground start/takeoff and landing. Current training starts in the air at its commanded altitude, so takeoff is untrained.
6. Retrain in a new run directory after the benchmark is trustworthy. Keep the current best as the comparison baseline and regard continued training from it as a warm start after physics changes. Repair curriculum persistence before treating `--resume` as a full experiment continuation.

A proposed first acceptance target is 30–60 seconds of hover across 100 held-out seeds, no terminations, altitude RMSE below 5 cm, and steady-state combined attitude RMS below 2°. Require recovery within two seconds followed by a sustained stable interval, with an explicit allowed altitude loss. Apply those criteria first to a nominal plant, then a specified randomized envelope. These are proposed engineering targets, not claims that the current controller meets them. Position-hold acceptance should be a separate milestone with a defined horizontal target and suitable feedback.

After those changes, an example new training run is:

```bash
cd fc-nn
.venv/bin/python main.py train \
  --resume runs/quad_hover/best_model.zip \
  --timesteps 1000000 --n_envs 4 \
  --save_dir runs/sim_v2_2026_09_12
```

This command has not been run. Starting from scratch should also be compared if the corrected observation/controller contract changes substantially. When export eventually becomes appropriate, the destination from inside `fc-nn` must be explicit: `main.py export <model.zip> --out ../weights.h`. The current default writes `fc-nn/weights.h`, which is not the header used by root `nn.c`; fix build dependencies before relying on subsequent firmware builds.

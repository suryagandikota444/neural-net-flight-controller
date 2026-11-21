**Your next milestone should be a reproducible, physically consistent simulation of the controller you actually intend to run.** Work in the order below. Each step has an exit condition so you can tell whether you have finished it, and each change should preserve the historical baseline for comparison.

**1. Preserve the current evidence and make the tools reproducible.** The September 12 review found a usable current best model, meaningful hardware logs, several new uncommitted tests and incomplete simulation/firmware consistency. Preserve all of them. The newest commit was `675cbac` on July 11, 2026; checkpoint archive metadata places the current best and final training artifacts on March 7. Commit date, training timestamp and experiment quality are separate facts.

From the repository root, these read-only commands show your starting point:

```bash
git status --short
git log -12 --date=iso --format='%h %ad %s'
git diff --stat
shasum -a 256 weights.h weights_working_w_pulse.h fc-nn/runs/quad_hover/best_model.zip
```

Save a source snapshot including untracked files before making implementation changes. A patch alone does not capture untracked tests. Preserve dated checkpoints/logs/results; use new output directories for subsequent experiments. Do not use `make clean` as an automatic first step if you want to retain the current local binaries as evidence.

The existing virtual environment contains cloud-offloaded files, and ordinary Torch/SB3 imports stalled during review. Restore those local files or recreate the environment outside the cloud-synced project directory, using recorded versions. First verify Python, NumPy, MuJoCo and a small environment step; then verify Torch/SB3 imports and checkpoint loading. Freeze a working dependency set. Do not combine a dependency upgrade with a physics correction if you want to attribute a change in results.

Exit condition: you can identify the exact source/model/configuration being tested, reproduce the historical diagnostic results within expected numerical variation, and import the normal training stack reliably. You can continue source reading and most C/physics work while resolving the Torch environment.

**2. Make software-only and passive bench execution explicit.** The immediate source changes are concrete correctness issues, not parameter tuning:

| Location | Proposed change | Required evidence |
| --- | --- | --- |
| [main.c](../../main.c) startup and motor-output path | Explicit bench/armed/fault states; remove automatic motor sequences from ordinary passive startup; make stop/fault bypass all later active output writes | A forced fault produces only the defined disabled outputs for the rest of the tick and subsequent latched state. |
| [ahrs.c](../../ahrs.c) normalization | Guard zero/invalid gradient and quaternion norm; define finite-state recovery/fault behavior without replacing the estimator algorithm silently | Actual C passes exact rest, zero acceleration, known rotations and invalid-input tests. |
| [main.c](../../main.c), [uart.c](../../uart.c), drivers | Explicit telemetry configuration, bounded failure handling, nonblocking/buffered logging where appropriate, measured timestamps and sample age | Worst-case timing and invalid/stale-sensor tests; no unsafe float-to-motor conversion. |
| [test_harness.c](../../test_harness.c) dispatcher | Select one test explicitly; separate passive and powered cases; accurate messages and structured results | Default bench path cannot reach tests 10/14 or normal automatic arming. |
| [Makefile](../../Makefile) | Track header/linker/flag dependencies; bind golden vectors to the weights/configuration | Changing a header or build configuration reliably produces the intended new executable and matching reference data. |

For the simple current all-sources-at-once build, adding headers to prerequisites is insufficient if the recipe still uses `$^`: it would also pass those headers and the linker script as compilation inputs. Explicitly compile `$(SRCS)` or `$(SRCS_TEST)` while tracking dependencies, or move to object files with generated dependency files and configuration-specific build directories. Ensure changed flags invalidate the appropriate outputs.

Write fault tests against externally visible behavior: all applied outputs, fault latch and recovery transition. A test that merely asserts a stop function was called would miss the present bug where later code writes motors again. Bench-disable behavior and an eventual in-flight fault strategy need separately defined state transitions; a full flight response depends on the failure and available fallback capabilities.

Exit condition: native tests expose and then verify the repaired faults; passive target operation is deliberate and bounded; an exported header cannot be ignored by the build. These repairs can proceed alongside simulation work and should precede new powered experiments.

**3. Correct the nominal plant before fitting anything else.** Start in [quadrotor.xml](../../fc-nn/quadrotor.xml) and [drone_env.py](../../fc-nn/drone_env.py). Choose one mass/inertia accounting convention. If 1.118 kg is the measured total aircraft mass, it must include the four motors rather than adding their masses again. If it is only an assumed historical total, label it until reweighed.

Do not blindly subtract 0.220 kg while leaving an inconsistent inertia model. Either model one rigid aircraft with explicit whole-aircraft mass/CoG/inertia and noncontributing visual geometry, or define components whose summed spatial inertia matches the aircraft. Verify the compiled flying subtree, excluding the separate mocap target.

Repair CoG randomization so the actual compiled inertial position changes. MuJoCo's same-frame optimization currently defeats the field edits. Prefer constructing a validated model from configuration; if mutating a compiled model, account for the relevant flags and derived constants and assert the resulting physical state. The test must observe inertial position and torque response, not only read back the array you assigned.

Build the following policy-free regression sequence: nominal mass/CoG/inertia; free fall; thrust-supported hover; single-axis torque signs; angular acceleration for known torque/inertia; effective CoG shift; randomized extrema; and timestep convergence with the same 100 Hz controller period. Keep sensor-noise comparisons in a separate test family from Newtonian physics.

Exit condition: you can predict the result of each test on paper and the simulator matches it within a documented numerical tolerance. The existing NN may perform worse on the corrected model; preserve that result as evidence of the correction's effect.

**4. Replace ad hoc validation with a canonical scenario runner.** Adjust [validate.py](../../fc-nn/validate.py), [test_env_physics.py](../../fc-nn/test_env_physics.py), [test_firmware_sim.py](../../fc-nn/test_firmware_sim.py) and the shared environment interface. Implement explicit nominal/randomized configuration, complete resets, independent randomness, fixed commands/events, correct physical-time accounting and machine-readable results.

Make a failure cause an unsuccessful test result when the test's declared acceptance rules fail. Keep separate commands or modes for exploratory diagnostics that merely print results. Remove verdicts based on a single in-band attitude sample and misleading PWM conversions. Report completed duration, sustained recovery, trajectory quality, saturation and all failure reasons.

Use [chapter 05](05-simulation-validation-and-training.md) to establish development and holdout suites. Run the current best, latest final and older reference policies through identical conditions. The dated results already show that the current best is a better direct-hover baseline than the latest final; preserve this comparison instead of assuming the largest timestep count wins.

Exit condition: two invocations of the same case agree; a deliberately weak controller fails for an explicit reason; a synthetic known-good trajectory passes the metric calculation; and a crashed short trajectory cannot outrank a complete hover through a low conditional RMSE.

**5. Choose the deployed observation and controller contract.** Read [chapter 04](04-controller-and-neural-network.md), then write the decision into configuration and tests. The present training environment uses true horizontal velocity, true altitude/vertical velocity, noisy true attitude and fixed output alpha 0.3. Firmware uses zero horizontal-velocity inputs, estimated attitude, barometric state, clamped altitude error, a PD collective adjustment and adaptive filtering. They are materially different controllers.

For the first milestone, use the state your hardware can actually estimate for attitude/altitude stabilization. Decide whether the NN controls collective altitude or an explicit outer loop does. Compare coherent variants in simulation; include each variant during its own training/evaluation. Do not retrofit mean subtraction, an altitude clamp or an additional loop to an existing actor and assume it remains valid.

Create a shared specification for units/frames, observations, output limits, timing, filtering, reset and fault state. Extract portable controller arithmetic into C functions with explicit state and input/output structures, leaving peripheral drivers in the hardware layer. Run that same C implementation against MuJoCo sensor inputs and recorded sensor traces.

Add meaningful paired tests: hand-calculated observations, actor arrays/raw outputs, filter steps/threshold transitions, altitude terms, motor limits, complete reset sequences and invalid-state outputs. Existing Python replicas are useful diagnostic references but should not be the only evidence for behavior of the C you deploy.

Exit condition: for identical timestamped inputs and initial state, Python/reference and deployed C signals agree at every documented boundary, with justified numeric tolerances. Closed-loop tests identify their exact controller variant and never silently substitute truth unavailable on hardware.

**6. Add realism in order of measured impact.** Implement measured mass and actuator mapping first, then actuator dynamics, sensor/estimator behavior and timing, while using one-factor comparisons to understand regressions. In practice some sensor/timing measurements can be collected before a thrust fixture is available; they do not need to wait for a perfectly calibrated propulsion model.

Track unknown quantities in the ledger and test bounded uncertainty envelopes. The aim is sufficient fidelity for the operating envelope and decisions you care about, not unlimited visual detail. If motor lag strongly changes stability and visual propeller shape does not change the modeled forces, motor lag deserves priority.

Use independent measurements to fit parameters, held-out measurements to validate them, and closed-loop simulation to judge controller robustness. Do not tune a plant parameter solely because it makes the NN's hover look better. That makes a favorable training environment rather than a measured model.

Exit condition: each major model element either predicts held-out data over a stated range or is explicitly uncertain and covered by a documented robustness test. Combined realistic cases meet the agreed simulation criteria through the actual C controller.

**7. Repair training bookkeeping, then train a candidate deliberately.** In [train.py](../../fc-nn/train.py), persist curriculum/timestep/configuration state, seed training and evaluation explicitly, evaluate fixed difficulty suites, and save enough metadata to reproduce selection. Keep a resume experiment distinct from a new warm start under corrected physics or changed observations.

First run a few complete rollouts to check logging, finite values, evaluation, saving and restart. Then choose a limited experimental question, such as whether the shared sensor/controller path learns adequate stabilization under a measured actuator model. Hold the other settings fixed and save to a new directory.

The existing CLI supports this example warm start after the prerequisite repairs; this is an instruction for a future experiment and has not been executed as part of writing this guide:

```bash
cd fc-nn
.venv/bin/python main.py train \
  --resume runs/quad_hover/best_model.zip \
  --timesteps 1000000 --n_envs 4 \
  --save_dir runs/sim_v2_contract_trial
```

Four workers change the rollout batch from the historical eleven-worker setup, so record that difference. The current CLI has no seed/configuration/curriculum-resume options that solve the bookkeeping problems automatically; implement and document those before treating this command as a fully reproducible experiment. If the observation/action contract changes substantially, compare a fresh policy as well as any compatible warm start.

Exit condition: a candidate meets fixed acceptance criteria on development and unused qualification scenarios, has reproducible artifacts and provenance, and passes the actual C inference/controller suite. Improving average reward alone does not complete this step.

**8. Export and inspect deliberately.** From inside `fc-nn`, the root firmware header destination must be explicit:

```bash
.venv/bin/python main.py export runs/sim_v2_contract_trial/best_model.zip --out ../weights.h
```

This command replaces the deployed header and should be used only for the selected candidate after preserving the previous header/hash. The default destination is a different `fc-nn/weights.h`. Verify all actor arrays, observation/action constants and generated golden vectors against the selected checkpoint, then build the target. A successful link verifies neither flight behavior nor deployed-device contents.

For an isolated normal firmware build using the current Makefile, this writes into a new temporary directory without replacing the existing root binary:

```bash
task_build_dir=$(mktemp -d /tmp/fc-build.XXXXXX)
make -B TARGET="$task_build_dir/firmware" all
```

The commands above do not flash hardware. Record compiler/version/flags, ELF and binary hashes, memory use and warnings. After an eventual authorized device update, verify the running firmware identity as part of the experiment record; a filename on your computer is not proof of what the board runs.

Exit condition: the chosen checkpoint, header, golden vectors, compiled controller and target build are traceably the same candidate. Target inference and timing measurements agree with the deployment contract.

**9. Use existing entry points with their actual limitations.** This table is a working command map, not a claim that every current script is a trustworthy acceptance test.

| Purpose | Command and working directory | Current interpretation |
| --- | --- | --- |
| Historical policy benchmark | Root: `fc-nn/.venv/bin/python docs/reviews/2026-09-12/run_baseline.py` | Repeats the dated direct-policy diagnostic; adapt through a new canonical runner after environment changes. |
| Physics diagnostics | Root: `fc-nn/.venv/bin/python fc-nn/test_env_physics.py` | Currently fails mass-related and noise-coverage checks; separate their causes. |
| Actual C NN/filter check | Root: `make host-test` | NN arithmetic passed during review; two filter expectations failed. Builds a local host executable. |
| Python parity/reference generation | Inside `fc-nn`: `.venv/bin/python test_nn_parity.py` | Needs the normal model-loading stack and writes reference data; ensure the intended model/header pair. |
| Approximate firmware scenarios | Root: `fc-nn/.venv/bin/python fc-nn/test_firmware_sim.py --runs 5 --duration 30` | Python approximation with identified baseline/metric limitations; current exit success is not quality acceptance. |
| Visual inspection on macOS | Inside `fc-nn`: `.venv/bin/mjpython main.py view --model runs/quad_hover/best_model.zip` | Inspect motion and signals; yaw commands and recording/event timing need repairs. |
| Training curves | Inside `fc-nn`: `.venv/bin/tensorboard --logdir runs/tensorboard` | Compare reward/failure/difficulty metrics; historical evaluation remained at easy curriculum. |
| Existing checkpoint validator | Inside `fc-nn`: `.venv/bin/python main.py validate runs/quad_hover/best_model.zip` | Repair reset/random-command/PWM/pass logic before relying on its verdict. |
| Serial capture | Inside `fc-nn`: `.venv/bin/python log_imu.py --port <actual-port> --seconds 60 --out <new-file.csv>` | Use only with the intended passive/controlled firmware state; old TEL format lacks acquisition timestamps. |
| Telemetry analysis | Inside `fc-nn`: `.venv/bin/python analyze_imu.py <TEL-format-file.csv>` | Current analyzer expects the newer 28-value TEL schema, not the old six-channel static file. |

Replace angle-bracket placeholders with actual values; they are not literal shell arguments. Viewer startup on macOS uses `mjpython` because of the passive-viewer requirement documented by [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). Use visualization to inspect behavior and saved metrics to judge it.

**10. Diagnose symptoms with a discriminating test before changing gains.** Keep this table beside your experiment log.

| Symptom | First checks that separate causes | Adjustment only after the result |
| --- | --- | --- |
| Falls at nominal zero-residual hover | Compiled mass; applied force; actual command mapping; acceleration with policy disabled | Correct mass/force units or calibrated hover mapping. |
| Good direct NN simulation, poor firmware-style behavior | Same recorded inputs through observations, filters, altitude path and actual C | Align the first divergent controller/estimator boundary. |
| Oscillating motor commands on a stationary board | Raw NN vs EMA vs baro P/D; sensor spectrum; actual timing | Correct the identified noise/lag path; measure the new phase response. |
| Slow tilt recovery | Actual torque authority, actuator lag, software filter tail, estimator delay and saturation | Adjust the limiting element or retrain against the measured path. |
| Persistent altitude offset | Measured thrust curve, mass, baro reference, saturation and collective NN output | Repair the bias/model error; reconsider altitude-loop design if necessary. |
| Hover altitude looks good but aircraft travels meters | Task/observations, true lateral motion and imposed force balance | Specify drift bounds for stabilization, or design position feedback as a separate task. |
| Random NaN or extreme output | Actual C exact-rest/invalid-input tests, finite checks and sample validity | Fix numerical/fault handling before interpreting controller performance. |
| Results change when logging/viewer is enabled | Loop timing, sensor age, shared RNG consumption and extra observation reads | Decouple instrumentation from timing/randomness. |
| CoG randomization has no measurable effect | Compiled inertial position, same-frame flags and known-force torque response | Repair model construction/derived state, then repeat envelope tests. |
| Latest checkpoint worse than an earlier one | Fixed independent benchmark, evaluation difficulty, curriculum and seed variance | Fix selection/training procedure; retain the stronger candidate. |
| New weights seem to have no effect | Checkpoint/header/binary hashes and build prerequisites | Correct artifact path/dependencies and rebuild deliberately. |
| Real and sim differ despite similar noise standard deviation | Bandwidth, color, bias, mounting vibration, latency and actuator dynamics | Add the measured missing structure, not just more white noise. |

**11. Organize your return into manageable work sessions.** Do not try to read thousands of C lines and tune a policy in one sitting. Each session should produce a small artifact that the next session uses.

| Session/work package | Concrete output | Completion test |
| --- | --- | --- |
| Source orientation | Annotated data-flow map and one hand-traced control tick | Explain every observation and every transformation to PWM without guessing units. |
| Historical baseline | Preserved source/model hashes and reproduced diagnostics | Identify the current best, its strengths and its known limitations. |
| Physics correction | Nominal plant configuration and analytic tests | Mass/hover/torque/CoG/timestep checks pass for explained reasons. |
| Test-runner repair | Fixed scenarios, complete reset and failure-aware metrics | Known failing cases fail; identical seeds reproduce. |
| Controller agreement | Shared contract and actual C sequence tests | Locate no unexplained divergence between training/deployment paths. |
| Passive sensor/timing evidence | Versioned logs, reference measurements and calibration report | Sample age, axes, numerical health, noise and delay are quantified. |
| Actuator identification | Measured maps/dynamics and held-out fit report | Force/response prediction error is bounded over the chosen envelope. |
| Integrated realism | Nominal and combined-uncertainty benchmark | Actual C controller meets agreed simulation targets or produces a localized failure. |
| Deliberate training | Candidate runs and fixed-suite comparisons | Improvement repeats and survives unused qualification cases. |
| Deployment preparation | Export/build/replay/target-timing report | Candidate identity and behavior are consistent end to end. |

Your first practical session can stay entirely on the computer: read the index and observation table, reproduce the historical baseline, inspect the total-mass discrepancy, and write down the exact first stabilization task. Then correct and test the plant and evaluator before committing another long training run. Physical measurements can progressively narrow uncertainty while this work continues.

**12. Know what “simulation ready” means.** You should be able to answer all of these with a file, test result or measurement:

- Which physical aircraft configuration does the model represent, and what is still assumed?
- Does the compiled model have the intended mass, inertia, CoG and motor forces/moments?
- Do reset, timing, sensor sampling and disturbances behave reproducibly?
- Does the deployed C controller consume the same information and compute the same outputs as the tested version?
- Do sensor, actuator and timing models predict independent measurements within stated limits?
- Does the policy pass the full declared task duration and recovery criteria across the stated envelope and unused seeds?
- Can you reproduce the selected model's training context, export and target build?
- Are takeoff, landing, position hold and fault handling explicitly implemented/tested or clearly outside the current milestone?

When one answer is missing, the next experiment should address that missing evidence. This gives you a concrete route to improving simulation without confusing more training, more realistic graphics or more filters with demonstrated progress.

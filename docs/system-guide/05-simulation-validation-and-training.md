**Make simulation a measuring instrument before using it to select or improve a policy.** A successful episode only establishes something useful when the plant, initial conditions, controller, duration and acceptance rules are known. This chapter defines that instrument and then explains how to train against it.

**1. Separate interface correctness, physical correctness and control performance.** These need different tests. An environment can satisfy Gymnasium's API while modeling the wrong mass; a physically correct model can expose a poor controller; a controller can succeed by exploiting a modeling error.

| Test family | Question | Example in this repository |
| --- | --- | --- |
| API and numerical | Are shapes, types, bounds, reset and termination valid? | Every observation has shape `(12,)`, is finite and matches the declared space. |
| Physics | Does the plant obey known forces, moments and units? | A measured-weight hover force holds altitude without a policy. |
| Controller arithmetic | Do implementations compute the same result? | Exported NumPy and actual C agree on raw inference and filter sequences. |
| Estimation | Does estimated state follow independently known motion? | C AHRS follows a specified rotation and stays finite at exact rest. |
| Closed loop | Does the whole system satisfy the specified task? | Sustained attitude/altitude stabilization across a defined parameter envelope. |
| Real-versus-sim | Does the model predict held-out physical measurements? | A thrust curve predicts measured force at unused command/voltage combinations. |

Run the SB3 environment checker once the normal Python environment is working. Its checks complement your own assertions; they cannot identify incorrect physics. The project's environment class is `QuadrotorEnv`:

```bash
cd fc-nn
.venv/bin/python -c 'from stable_baselines3.common.env_checker import check_env; from drone_env import QuadrotorEnv; check_env(QuadrotorEnv())'
```

The checker is described in [SB3's custom environment documentation](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/custom_env.html).

**2. Build a policy-free reference plant.** Use one factory/configuration path for the environment, physics tests, firmware simulation and benchmark. It should return a known nominal aircraft and apply randomization only when explicitly requested. Tests need public ways to disable noise, wind, random commands and curriculum without rewriting private fields in several scripts.

Before running an NN, verify total flying mass, center of mass, inertia, gravity, free fall, thrust-supported hover, motor moment signs, yaw reaction torque and timestep convergence. Then repeat selected tests at the extremes of the configured mass, thrust and CoG ranges. Confirm that each randomized parameter actually changes the compiled model and the resulting dynamics.

The existing dated benchmark is a **historical diagnostic**. It overrides private environment fields to freeze commands and randomization. If you change the mass representation or environment internals, audit those overrides: assigning `BASE_MASS` to the root body could recreate a mass-accounting problem. Preserve the dated script for reproduction of the old state and create a canonical evaluator using the new shared configuration interface.

Use a simple conventional controller as a diagnostic baseline after validating the plant. For example, an explicitly documented attitude/rate controller plus altitude control can reveal whether the physics and sensor interface are controllable at all. It is not automatically a flight-ready fallback, and its gains also need units, saturation handling and tests. Its value is interpretability: you can inspect requested torque and damping while debugging a learned policy.

**3. Specify reset as a complete transaction.** A reset must initialize physics, task, sensors and controller state. Test every item below; a reset that only changes position is incomplete.

| State group | What must be initialized or deliberately retained |
| --- | --- |
| Physics | Position, quaternion, linear/angular velocity, actuator state, external forces and contact-related state through the appropriate MuJoCo reset path |
| Aircraft parameters | Nominal or sampled mass, inertia, CoG, motor coefficients and any derived constants |
| Task | Command values, command schedule, target reference frame, episode clock and curriculum/scenario configuration |
| Sensors | Biases, colored-noise states, sample schedules, delay queues and calibration status |
| Estimation | AHRS quaternion, barometer reference, previous height/time and vertical-velocity filter |
| Controller | Output EMA, previous action for reward, arming/ramp state, altitude-loop state and latched fault |
| Evaluation | Accumulators, termination reason, time in tolerance, saturation counters and event timestamps |

Compare a fresh instance with a previously failed/saturated instance after the same seeded reset. They should produce the same trajectory for the same inputs within the chosen numerical tolerance. Test same seed/same configuration, different seeds, repeated resets, reset after an exception, and vectorized workers.

Separate random-number streams for domain parameters, sensor noise, commands and disturbances. In the current implementation, extra `_get_obs()` calls consume randomness. A diagnostic viewer or added log read should not silently alter later command schedules or wind samples. Sample sensor noise as part of advancing sensor time; reading a stored observation should not resample the world.

**4. Establish one timeline.** Write down this sequence explicitly: acquire the observation at time `t`, apply the command defined for time `t`, compute the action, evolve actuators/physics to `t+dt`, acquire the next observation, compute reward and termination for the resulting transition. Other orderings can be valid, but all code and metrics must use the chosen one.

Currently command changes around the post-step observation/reward path and the viewer's command update can create inconsistent event timing. Add a test with a single scheduled command step at a known tick. Verify its first appearance in the observation, controller output, reward and event log. Do the same for a gust and a sensor dropout.

Use physical elapsed time when injecting stalls. Holding an old command for 50 ms advances the aircraft by 50 ms even if it counts as one controller iteration. Record both controller ticks and simulated time. Require delayed samples to carry their acquisition timestamp; changing `dt` alone does not reproduce stale data.

Distinguish task failure (`terminated`) from an administrative time limit (`truncated`). A crash or invalid state is a failure; reaching the normal horizon is usually truncation. This distinction affects value bootstrapping during learning, as explained by [Gymnasium's time-limit guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Report it correctly even if your evaluator also creates a separate pass/fail result.

**5. Test the reward as an engineering objective.** Log every reward component separately. Current positive reward combines an exponential attitude/altitude term with an alive bonus; penalties cover horizontal/vertical speed, angular rates, raw-action changes and a motor-pair symmetry expression. The reported sum hides how those incentives compete.

Construct short, hand-understandable trajectories: perfect hover; correct altitude with steady tilt; level at the wrong altitude; oscillation around target; smooth recovery; rapid but saturated recovery; large horizontal drift; and early termination. Calculate reward components independently for a few samples. Confirm that outcomes you prefer actually receive the ordering you intended over a complete episode.

Distinguish three quantities when reading “raw action”: the actor's unbounded mean, the stochastic action sampled during training, and the action passed to the EMA after environment clipping. The reward's action-change penalty uses that pre-EMA, bounded action. It is not necessarily the unbounded actor output logged by an inference diagnostic. Also, PPO does not backpropagate through this NumPy reward calculation or MuJoCo. Rewards contribute to return/advantage estimates; PPO differentiates a policy/value loss using collected transitions. The source comment about directly punishing the NN's gradient is therefore an informal description of an incentive, not the differentiation path. You can trace the actual update in [SB3's PPO implementation](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ppo/ppo.html).

Inspect these particular choices before adjusting coefficients:

- The current so-called diagonal penalty compares `(a0+a3)` with `(a1+a2)`, which is left versus right in your motor order. It can discourage a useful roll correction. Rename it to its actual meaning, then test whether retaining it improves the stated task.
- Penalizing raw action changes differs from penalizing changes in filtered motor command, force, current or aircraft jerk. Choose the quantity for a stated purpose. A penalty before the filter may regularize network output but does not directly measure actuator stress.
- A commanded sustained tilt produces horizontal acceleration. Penalizing horizontal velocity while demanding tilt can create a compromise rather than perfect command tracking. For a tilt-step test, evaluate attitude response and a separately specified motion envelope.
- An alive bonus can reward remaining in an undesirable state when termination and error penalties permit it. Plot behavior, time outside bounds and failure causes alongside return.
- Reward magnitude changes with episode length and control frequency. If you change frequency, decide which terms represent an integral over time and which represent event costs; scale them accordingly rather than preserving numbers automatically.

Do not optimize return first and then choose acceptance thresholds that fit the winning policy. Define physical outcomes first. Reward is a training signal, and those outcomes are the evaluation criteria.

**6. Create a fixed scenario suite.** Start with ideal sensors and the corrected nominal plant. Add measured realism progressively. Give each case an ID, configuration file, seed set, duration, event schedule and acceptance rule.

| Scenario | Controlled changes | Main questions and metrics |
| --- | --- | --- |
| Level hover | Initial level attitude, airborne at the specified target | Bias, altitude RMSE, attitude RMS, oscillation, saturation and horizontal displacement |
| Initial tilt | Positive/negative roll and pitch; later combined axes | Recovery time with dwell, peak angle, altitude loss, saturation and yaw coupling |
| Rate disturbance | Known initial angular velocity or torque pulse | Damping, peak excursion and recovery without conflating it with a command change |
| Altitude step | Bounded upward/downward target steps | Rise time, overshoot, settling, thrust margin and barometer delay sensitivity |
| Attitude command | Roll/pitch steps and returns; both signs | Tracking, command latency and allowed translation |
| External force | Lateral force steps, pulses and changing directions | Attitude/altitude stability and drift; position-hold criteria only for a position task |
| Parameter variation | Mass, CoG, inertia, per-motor force/torque changes | Actual robustness envelope and asymmetric actuator authority |
| Sensor variation | Bias, noise, sample rate, lag and mounting error | Estimator error, command noise and accumulated drift |
| Timing variation | Jitter, stale sample, delayed output and bounded missed updates | Physical-time stability and fault behavior |
| Actuator variation | Nonlinear map, lag, deadband, voltage and unequal motors | Recovery/hover feasibility and sustained saturation |
| Long duration | Longer stable command and changing operating conditions | Slow drift, accumulated estimator error and rare failures |
| Startup/takeoff/landing | Ground contact, calibration, arming/ramp and reference changes | Correct state transitions, liftoff/landing behavior and bounded failure handling |

Ground start is a separate development task. Current episodes start in the air and terminate below `z=0.05`; placing the aircraft on the ground is therefore not a valid takeoff test without changing the task/state machine. Define allowed ground contact during startup, a liftoff timeout, contact/tilt limits and landing criteria. Test those states before including them in training.

For tilt, first use a small grid such as ±5°, ±15° and ±30° about each axis, then combined cases within the intended envelope. For disturbances, use documented force/torque units. Do not label 0.5 N as a wind speed without an aerodynamic conversion model.

**7. Use metrics that cannot hide failure.** Store per-episode results before aggregating. At minimum include full-duration completion, termination reason, actual elapsed time, altitude bias/RMSE/maximum error, roll/pitch error RMS and peaks, maximum horizontal displacement, saturation fraction, maximum consecutive saturation duration and finite-state violations.

For irregular sample times, calculate time-weighted error:

```text
RMSE = sqrt(sum(error_k^2 * dt_k) / sum(dt_k))
time_in_band = sum(dt_k for samples satisfying the band) / sum(dt_k)
```

For a fixed timestep this reduces to ordinary sample RMS. Specify whether attitude RMS is per-axis or `sqrt(mean(roll_error^2 + pitch_error^2))`; those are different quantities. Keep radians internally and label plots in degrees explicitly.

Define recovery as the first time after a disturbance when the relevant errors enter their bands **and remain there for a specified dwell interval**. A single crossing of ±10° is not settling. Require altitude and angular-rate conditions as appropriate so a fast pass through level is not success. If recovery never occurs, report failure/censored recovery, not zero seconds or a conveniently clipped value.

Report failures before conditional error summaries. A controller that crashes after two seconds can have a smaller trajectory RMSE than one that survives thirty seconds. Show completed-run metrics separately and retain pre-failure metrics for diagnosis. Do not silently replace post-crash samples with zeros or discard failed episodes. An optional combined score needs an explicit failure penalty and should accompany, not replace, the underlying outcomes.

Plot representative successes, the worst completed run, and every failure around its event. The minimum useful synchronized plot has command and truth; estimated state; observation values; raw/filtered/final actions; and termination or saturation flags. Save numeric data with the plot.

**8. Separate development seeds from qualification seeds.** Five seeds are useful for an inexpensive regression or triage. They are weak evidence about rare failure. Use a larger fixed development suite while iterating, then a previously unused qualification set after choosing the model and thresholds. Keep a final holdout set unavailable to routine model selection.

Use identical scenario seeds for paired model comparisons, while ensuring controller-dependent random sampling does not change the disturbance schedule. Report the distribution and worst observed case, not just the mean. For confidence intervals, resample independent episodes or experimental runs; treating thousands of adjacent control ticks as independent dramatically overstates certainty.

As an illustration, zero failures in 100 independent, identically distributed episodes has a one-sided 95% binomial upper bound of `1-0.05^(1/100)`, approximately 2.95%, on failure probability under that same distribution. It does not establish zero risk, cover unmodeled real-world conditions, or replace hardware validation. Use that calculation to understand evidence strength rather than to claim certification.

An initial proposed milestone is 30–60 seconds of attitude/altitude stabilization across 100 held-out seeds with no terminations, altitude RMSE below 5 cm and steady-state combined attitude RMS below 2°. Define the excluded transient interval in advance and still report transient peaks. For a bounded tilt disturbance, propose recovery within two seconds plus a one-second dwell and a separately chosen altitude-loss bound. Revise these targets based on task needs and achievable independent sensor accuracy; they are not measured specifications of the present hardware.

**9. Build an interpretable realism ladder.** Begin with the exact training-style observation/control path, with every difference named. Then introduce one factor at a time: measured mass/thrust; output filter; missing horizontal-velocity feedback; actual altitude observation and altitude controller; sensor estimator; barometer timing; motor dynamics; noise/bias; and timing disturbances. The appropriate order depends on which factors are already corrected, but comparisons must state their common baseline.

After isolated tests, combine factors deliberately and test interactions. A factorial subset can reveal that motor lag is tolerable alone but unstable with estimator delay. Domain-randomization envelopes should preserve physically meaningful correlations: voltage and thrust capability, mass and inertia, sample rate and filter response.

The existing A–F script does not currently establish this chain. Its A already uses barometric estimation and zero horizontal velocity; some adjacent scenarios change multiple things; and F does not retain E's thrust deficit. Rework its configuration and actual C integration before interpreting letters as causal proof. Include both truth-based diagnostic state and sensor-based controller state, clearly separated so truth cannot leak into the deployed controller.

**10. Understand what your PPO settings mean before changing them.** The current setup uses 4,096 steps per environment. Eleven workers collect 45,056 transitions per rollout, corresponding to 40.96 seconds per worker with resets inside that stream. PPO then reuses that batch for multiple optimization epochs. More workers change data collection and update batch size; they are not merely a speed switch.

The current network is small: two 32-unit hidden layers. Learning rate is `3e-4`, batch size 512, epochs 10, discount 0.995, GAE lambda 0.95, policy clip 0.2, entropy coefficient 0.005, value coefficient 0.5 and max gradient norm 0.5. Read the policy loss, value loss, entropy, approximate KL, clip fraction, explained variance, reward components and failure rate together. No single one proves convergence.

At 100 Hz, discount 0.995 gives about `1/(1-gamma)=200` ticks, or two seconds, as a rough discounted reward horizon; rewards ten seconds away carry a factor of about 0.00665. This is an interpretation, not a mandatory setting change. Consider whether delayed outcomes such as slow drift or takeoff failure are represented adequately by the task and critic before tuning discount.

First make the experiment reproducible and preserve current hyperparameters. If learning is unstable, investigate broken observations/rewards, unrealistic state distributions, saturation and excessive curriculum difficulty before widening the network or searching many learning rates. Compare at least several training seeds for a serious architecture or reward decision. A lucky single training run is not proof of superiority.

**11. Repair curriculum and checkpoint selection.** The current training callback advances difficulty only in training environments; the separate evaluation environment remains at curriculum zero. Keep fixed easy, intermediate and hard evaluation suites so progress is visible at each level. You can also report an evaluation at current training difficulty, but that changing distribution cannot replace a fixed benchmark.

A checkpoint manifest should include total timesteps, optimizer/model state, curriculum state and schedule, all relevant seeds, environment/controller configuration, package versions, source revision and dirty patch, actor/header hashes and evaluation results. Decide what an exact resume means. Restarting callback progress and timestep accounting is a warm-start experiment, not a continuation with preserved curriculum.

Run a short training smoke test of a few complete rollouts to verify finite observations/losses, callback execution, checkpoint writing, evaluation and resume. Save to a new directory. Then run a planned budget and compare against the current best, an appropriate older model and the conventional diagnostic baseline. Keep training/validation/holdout distinct, use deterministic inference for benchmark comparisons, and report multiple training seeds where practical. These practices align with [SB3's reinforcement-learning evaluation guidance](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/rl_tips.html).

Select a model that passes the fixed physical criteria, then compare error, disturbance recovery and actuator effort among passing models. The present `best_model.zip` outperforms the latest final checkpoint on the direct hover benchmark despite having fewer training timesteps. More training and a later filename do not guarantee better task behavior.

**12. Finish with deployment equivalence.** Re-run the chosen fixed suite through the actual portable C controller, using the exact exported header, observation path and configuration intended for the target. Compare it with the training implementation at each intermediate signal. Then test recorded sensor replay on the C implementation and target-side inference/timing separately. Reserve “hardware-in-the-loop” for a setup where target execution is actually coupled to a simulated plant or sensor stream; computing predicted outputs while manually tilting a board does not close the aircraft's feedback loop.

The deliverable from this chapter is a versioned scenario suite, a per-episode result dataset, explicit pass/fail rules, and a checkpoint manifest. Once those exist, each training run answers a clear question instead of merely generating a new model file.

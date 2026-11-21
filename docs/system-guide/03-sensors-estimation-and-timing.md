**Your controller responds to measurements of the aircraft, not to the aircraft's true state.** A simulated policy that sees instantaneous true attitude and velocity is solving an easier and different problem from firmware reading noisy, delayed sensors. Validate the measurement chain in layers: raw transport, units/frame calibration, timestamping, filtering, estimation, and finally the observation vector.

Complete these sensor tests with an explicitly selected bench-only firmware. The current normal firmware and automatic full test harness are not suitable defaults for passive measurements. Use independent references when claiming accuracy: known orientations, measured angles/heights, a motion reference or calibrated instruments. The estimator's own output is not ground truth.

**1. Write a coordinate contract before changing signs.** Use body X forward, body Y left, body Z up if you retain the simulation's current convention. Define world axes separately. Define the quaternion as `[w,x,y,z]` mapping body-frame vectors to world coordinates, and verify that convention in every helper. Define gyro units as rad/s, accelerometer as either m/s² or g with an explicit conversion, and altitude/velocity as positive upward.

Under this convention, positive roll raises the left side, positive pitch lowers the nose, and positive yaw turns the nose left. These names follow a right-hand rotation about each positive axis. Many flight-controller conventions use a different handedness/sign vocabulary; do not transfer a “nose-up” comment or Betaflight motor number without converting it.

Measure a chip-to-body rotation through known static orientations and deliberate single-axis movements. A physically meaningful fixed rotation normally uses one coherent transform for measurements expressed in the same sensor frame. The current alignment macros negate accelerometer Y and gyro X, while comments describe identity. That discrepancy demands a documented hardware experiment. Do not guess replacement signs from a screenshot of a wobbling aircraft.

Draw a test table with commanded physical motion, expected gyro sign, expected gravity-vector change, measured raw sign, transformed sign and estimator angle sign. For small static roll `phi`, expected gravity-related accelerometer readings are approximately `[0, sin(phi), cos(phi)]` g. For small static pitch `theta`, they are approximately `[-sin(theta),0,cos(theta)]` g. Apply those examples only after defining the body-to-world rotation as above.

**2. Understand what the accelerometer measures.** A useful model is specific force:

```text
f_body = R_body_to_worldᵀ * (a_world - gravity_world)
gravity_world = [0,0,-g]
```

Thus a supported stationary level board reads approximately `[0,0,+g]`; an ideal freely falling board reads approximately zero; a thrust-supported level hover also reads approximately +g. An aircraft initialized motionless in midair with no thrust is about to free-fall, not physically supported at rest. Repeatedly reading that unadvanced simulation state is not a realistic “AHRS ground-settle” procedure.

An off-center IMU can also see rotational acceleration. At fixed body offset `r`, the additional acceleration includes `alpha × r + omega × (omega × r)`. Measure the sensor's actual position relative to CoG if this contribution can be appreciable. MuJoCo's accelerometer is attached to a site, so use the site placement deliberately; the official [sensor reference](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#sensor-accelerometer) defines its local frame.

**3. Prove sensor identity, configuration and transport.** Use read-only identification and readback tests first. Record the detected IMU/barometer model and raw register values. For the installed ICM42688P branch, inspect reset/bank selection, `GYRO_CONFIG0`, `ACCEL_CONFIG0`, `GYRO_ACCEL_CONFIG0`, power mode, data-ready configuration and burst layout. The [TDK ICM-42688-P datasheet](https://invensense.tdk.com/wp-content/uploads/2020/04/ds-000347_icm-42688-p-datasheet.pdf) is the reference for those fields and raw scale factors. A macro called “50 Hz” does not prove the complete sensor path has that bandwidth or delay.

Capture SPI transactions with a logic analyzer and verify clock mode/rate, address read bit, chip select, returned bytes and signed assembly. Use frozen or mocked byte sequences in a host test to check endianness and negative values without hardware. For I²C, test device absent, wrong address, repeated starts, read lengths, ACK/NACK and bus recovery. Confirm that failures propagate to a health state rather than stale-but-plausible observations.

Distinguish an output data rate of 1 kHz from a 100 Hz control loop. Reading a register at 100 Hz does not mean the sensor sampled at the same instants, nor that all high-frequency content was removed before downsampling. Use data-ready signals/timestamps where practical to identify sample age and duplicates.

**4. Calibrate static accelerometer bias and scale with six faces.** With motors inactive and the board mounted stably, collect data in +X, -X, +Y, -Y, +Z and -Z gravity orientations. Allow thermal/measurement settling, then collect 10–30 seconds per face as an initial protocol; repeat the full sequence. Do not treat a tilted hand-held board as an exact axis reference.

For one axis with positive/negative gravity raw means `r_plus` and `r_minus`:

```text
bias_counts = (r_plus + r_minus)/2
counts_per_g = (r_plus - r_minus)/2
accel_g = (raw_counts - bias_counts)/counts_per_g
```

This simple model estimates independent biases/scales. It does not identify all cross-axis misalignment. Evaluate corrected vector magnitude and off-axis residuals on additional orientations that were not used for fitting. If cross-axis errors matter, fit a full calibration transform with suitable reference orientations and report its limitations.

The current firmware's startup average subtracts offsets relative to `[0,0,1]` g. That assumes a level, stationary board. If the aircraft starts tilted, horizontal gravity components can be mistaken for bias, causing the later estimator to believe the tilted aircraft is level. Separate factory/bench calibration from startup gyro-bias estimation, or explicitly constrain and validate the startup pose assumption.

Keep calibration before normalization/clipping. Normalizing by 2 g for the NN is not the same as configuring the sensor to ±2 g. The ICM branch uses a wider hardware range; it can report values that later saturate the network's ±1 observation limit. Log both values so information loss is visible.

**5. Calibrate gyro bias, sign and scale.** Record stationary gyro output for 60–120 seconds after warm-up, with temperature if available. Estimate each axis's mean, standard deviation, slow trend and sensitivity to board position. Repeat after power cycles and at relevant temperatures. Bias should be removed from stationary readings; actual motion should not be averaged into the bias estimate.

For scale and sign, use a controlled single-axis angular motion with a known independent angle or rate. After bias correction, integrate `omega*dt` over a unidirectional movement and compare with the external reference. Perform positive and negative directions separately; a complete out-and-back can hide a scale error by net cancellation. Keep rotations about one well-defined axis or use full quaternion integration rather than summing Euler angles through arbitrary 3D movements.

A constant bias of 0.01 rad/s corresponds to roughly 0.573°/s of uncorrected angular drift. This simple calculation helps connect apparently small gyro offsets to visible behavior. The accelerometer can constrain roll/pitch drift under suitable motion conditions, but an IMU-only estimator does not provide an absolute yaw reference.

**6. Measure timing before interpreting noise.** Log a monotonic hardware timestamp or sequence-linked timer value at sensor acquisition, controller start/end, and motor-command commit. Record which sensor samples were fresh. PC serial-receive time should remain a transport diagnostic, not the sole clock for estimator replay.

For the MCU DWT cycle counter at an assumed 168 MHz, unsigned 32-bit wrap occurs about every 25.57 seconds. Unsigned subtraction handles intervals shorter than one wrap; long logs need an extended timestamp or explicit wrap tracking. Verify the actual clock assumptions. Use a GPIO timing pin and scope/logic analyzer to independently check measured execution intervals.

Report median, p95, p99, maximum, missed deadlines, duplicate samples and sensor age. Separate control-loop period, computation duration, measurement interval and end-to-end delay. A 100 Hz average can coexist with repeated destabilizing 30 ms gaps. SysTick's polled count flag does not preserve a count of every missed tick.

Trace delay from sensor sampling/filtering through bus read, estimator, neural inference, software filter, PWM frame update, ESC interpretation and thrust rise. PWM repeats at 400 Hz while the controller updates at 100 Hz. Depending on phase and preload behavior, writing a compare register need not change the pulse immediately. Measure that delay rather than assigning every effect to the neural network.

**7. Budget telemetry explicitly.** At 115200 baud with 8N1 framing, the ideal byte capacity is `115200/10 = 11520 bytes/s`. A 230-byte line needs about 20 ms to transmit even if only sent five times per second. Average throughput fitting the link does not prevent a blocking burst from exceeding the 10 ms control period.

For new logging, choose a bounded record format and check `record_bytes * records_per_second`, allowing headroom for framing and other traffic. A 128-byte record at 100 Hz already exceeds that byte budget. Use smaller binary records, selected channels, bounded buffering/DMA, higher verified link throughput or suitable local storage. At higher-rate sensor capture, do not force a large human-readable CSV through the same control-loop UART path.

Record dropped-buffer counters and sequence numbers. A nonblocking logger that silently drops the interesting transient is still a bad experiment. Log only what the selected experiment needs, but retain enough information to reconstruct the exact stage being compared.

**8. Characterize noise after separating real motion and bias.** For stationary measurements, remove the intended mean/bias and inspect time series, per-axis standard deviations, percentile tails, clipped values, cross-axis covariance and autocorrelation. Plot spectra only after checking actual sampling regularity, timestamp gaps and duplicate values. Use a defined analysis window and sampling rate for every comparison.

The first-difference estimate `std(x[k]-x[k-1])/sqrt(2)` estimates white independent measurement noise only when underlying signal changes are negligible and the noise assumptions hold. Correlated filtered noise or actual motion invalidates that interpretation. The current flight log is roughly 5 Hz, so it cannot resolve sensor/control-band vibration above 2.5 Hz without alias ambiguity. Its 0.465 rad/s and 0.862 g first-difference estimates are variability indicators, not a calibrated high-frequency noise specification.

Use separate datasets for electronics stationary, aircraft stationary, motors rotating without propellers, and properly fixtured loaded propulsion. These conditions isolate different sources; props-off motor vibration does not reproduce loaded propeller vibration. Powered tests require the validated setup described in the physics chapter. Record voltage, RPM/throttle and fixture configuration because both vibration and measurement response can depend on them.

Compare spectra at the same units, sample rate, filter configuration and operating point. Use detrending/windowing consistently. Spectral peaks can indicate periodic excitation, but identifying a peak with a particular rotor/blade frequency requires RPM and a sufficient sampling bandwidth. A load cell or camera also has its own bandwidth and delay; include those in uncertainty rather than treating the reference as perfect.

**9. Put a measured sensor model in simulation.** Begin with noiseless ideal sensors to prove correctness. Then add each measured effect in a documented order: mounting transform, deterministic calibration residual, slow bias, noise/filter dynamics, sampling delay/hold, quantization and saturation. In physical hardware, some filtering occurs before software reads or downsamples the signal; reproducing only observation-space white jitter is insufficient if aliasing and delay dominate.

A simple correlated residual model is `b_next = rho*b + sqrt(1-rho²)*sigma*normal()`, with `rho=exp(-dt/tau)`. Its stationary standard deviation is `sigma`; it represents one possible colored component, not every source of bias. A random walk instead uses increments proportional to `sqrt(dt)` and has different long-term behavior. Fit or bound these behaviors from data and record which interpretation your parameter uses.

Random draws should use dedicated seeded generators for plant parameters, sensor noise, disturbances and command schedules. Otherwise adding an observation debug call changes the future wind/command/noise sequence, making comparisons unfair. Verify whether the installed MuJoCo version actually applies XML noise attributes in the way you expect; compare noiseless repeated sensor reads and empirical variance, and avoid adding the same noise twice.

**10. Test Madgwick estimation with known trajectories before using noisy flight data.** The essential estimator tests are exact identity rest, known static roll/pitch orientations, constant single-axis rotation, bounded rotation steps, return to level, long-duration yaw drift, zero accelerometer magnitude, corrupted sensor input, delayed samples and variable `dt`. Use the accelerometer signal implied by the trajectory, not simply `[0,0,1]` while rotating the simulated body arbitrarily.

Check finite output, quaternion norm, angle error, settling time and overshoot. Compare quaternions with sign-invariant angular error:

```text
angle_error = 2*acos(clamp(abs(dot(q_est_normalized, q_true_normalized)),0,1))
```

The absolute dot product is needed because `q` and `-q` describe the same orientation. When comparing Euler angles, wrap periodic differences rather than subtracting values near ±180° naively.

First repair the C zero-gradient normalization and add finite-value checks. Then run the actual C estimator on the same samples as the Python version and compare over time. Replace only the platform-specific square-root implementation for a native host build, document that substitution, and eventually check the ARM build too. Do not regard the Python copy's extra guard as proof the firmware already has it.

**11. Understand the estimator's physical limit during acceleration.** A static tilted board provides gravity-direction information. A freely moving multirotor accelerometer also responds to thrust and translational dynamics; it cannot always distinguish tilt from acceleration. For example, in ideal thrust-driven motion its specific force can remain nearly aligned with body Z while the aircraft is tilted. Gyro integration, model assumptions and additional measurements determine performance in that case.

Increasing `AHRS_BETA` can correct certain static errors faster but can also pull attitude toward an acceleration-contaminated apparent gravity direction. Lowering it can reduce that contamination while allowing more drift. Sweep beta over defined reference trajectories, evaluate both static and dynamic error, and select from measured tradeoffs. An acceleration-magnitude gate can help in some cases but is not complete: substantial horizontal acceleration can coexist with a magnitude near 1 g. Do not identify “add more Gaussian attitude noise” as a substitute for estimator dynamics.

**12. Measure barometer behavior as a sensor-plus-estimator chain.** For the detected device, validate raw pressure/temperature compensation against documented test vectors or a trusted reference implementation. Use the relevant [BMP280 datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp280-ds001.pdf) or [DPS310 datasheet](https://www.infineon.com/assets/row/public/documents/24/49/infineon-dps310-datasheet-en.pdf?fileId=5546d462576f34750157750826c42242) to decode oversampling, readiness, filter and update-rate settings. Device branches have different behavior; a generic “BMP-like IIR” is not calibration.

On a stationary bench, log pressure, temperature, derived altitude, sample freshness and vertical velocity. Repeat through warm-up. Then move the unpowered/bench-mode sensor through independently measured height steps, such as several known increments over a metre-scale range, in both directions. Hold each level long enough to observe offset, noise and settling. Keep an independent height reference; room pressure drift and airflow can otherwise look like height error. Repeat selected tests with the sensor in its actual mounting/enclosure.

The current local approximation is `height = (pressure_reference - pressure)/12.015` metres. It is a near-ground pressure-gradient approximation, not a universal atmosphere model. Compare slope and offset over the intended range and conditions. Distinguish relative short-term height precision from absolute altitude accuracy. A simulation can demonstrate 5 cm true-height control using ideal measurements even when the actual barometric chain cannot reliably resolve that performance.

**13. Test vertical-velocity estimation separately.** `baro.c` differentiates altitude at the decimated update interval and smooths the derivative with alpha 0.1. With 40 ms updates, that EMA alone has a time constant `-0.04/log(0.9) ≈ 0.38 s` and takes about 0.88 s to reach 90% of a step. Pressure-sensor/internal-filter delays add to it. That is much slower than a 10 ms control tick.

Apply synthetic altitude ramps, steps, sinusoids and gaps to the actual derivative/filter logic. A constant-velocity ramp should converge to the known velocity; a step should produce a finite transient whose size and decay are explainable; repeated stale samples should not be mistaken for fresh independent measurements. Differentiate using elapsed time between successful fresh samples. Do not retune `ALT_KD` until the sign, scale and delay of its velocity input are known.

Construct the simulator's barometer from the observed sampling/filter/noise behavior, then run that same altitude/velocity estimator in the simulated controller. Avoid fitting pressure noise and adding an arbitrary extra altitude IIR if the measured data already includes the device filter; that can double the filtering.

You are ready to connect the controller when a static board, a known angle trajectory and a known height trajectory produce bounded, timestamp-correct sensor/estimator outputs in both C and simulation, and you can explain the remaining uncertainty by measurement rather than guesswork.

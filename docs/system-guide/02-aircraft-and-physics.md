**The aircraft model should earn your trust before the neural policy is allowed to compensate for it.** A controller can learn to stabilize an incorrectly modeled vehicle. Good simulated flight therefore does not prove that the model represents your aircraft.

Keep all physical measurements in SI units in the model: kilograms, metres, seconds, newtons, newton-metres and radians. Retain the original instrument units in the raw measurement file as well. Record the exact battery, propeller, motor, ESC, payload and mounting configuration for every measurement; moving a battery can change CoG and inertia even when total mass is unchanged.

**1. Measure the complete flight mass and resolve the present accounting error.** Weigh the unpowered flight configuration at least five times, removing and replacing it on the scale between readings. Check zero and a suitable known reference mass. Include battery, wiring, fasteners, landing supports and payload. Record mean, repeatability, scale resolution, configuration and date. If a component changes, create a new configuration entry instead of silently updating a historical measurement.

The current XML assigns 1.118 kg to `drone`, plus four fixed child motors of 0.055 kg each. Thus its flying subtree is 1.338 kg. The separate visual target has inferred mass too, but it is not attached to the aircraft. Do not estimate aircraft mass by summing every body in the scene. Inspect the drone subtree or explicitly sum its descendants. The review confirmed equilibrium at command 0.386052 for that current 1.338 kg model; command 0.322576 is appropriate only for 1.118 kg under the assumed 8.5 N-per-motor linear map.

For the corrected design, either represent the entire rigid aircraft with one measured inertial body and visual geoms/thrust sites that contribute no additional mass, or model component masses and inertias whose aggregate equals the measurements. The first representation is often simpler for this rigid-airframe task. Preserve total inertia as well as total mass. Subtracting 0.220 kg from the central mass alone was an isolated diagnostic, not a completed inertia calibration.

**2. Measure geometry relative to a declared body origin.** Choose a fixed point, preferably the IMU reference location or another well-defined airframe datum. Measure each motor shaft center as `(x_i,y_i,z_i)` relative to it. Photograph the axes and motor labels. Measure motor-plane height, IMU offset, landing-contact geometry and battery position.

The XML uses motor centers at approximately `(±0.16,±0.16,0.01)` m. The horizontal radial distance is therefore about 0.2263 m, not 0.16 m. A quoted 450 mm wheelbase, radial arm distance and X/Y torque lever arms are different quantities. Measure the geometry actually built. Small arm errors change torque authority even when total lift remains correct.

**3. Measure center of gravity without running motors.** One practical method uses several calibrated support force measurements at known planar positions. After accounting for support/fixture tare, compute:

```text
x_CoG = sum(F_i * x_i) / sum(F_i)
y_CoG = sum(F_i * y_i) / sum(F_i)
```

Use stable supports and repeat after repositioning the aircraft. Support friction or accidental contact can corrupt reactions. A balancing fixture can provide a cross-check. Vertical CoG requires an additional orientation or a component/CAD mass model; a single horizontal support test does not identify it. Record uncertainty and repeat with the battery in its allowed mounting extremes.

Set motor lever arms relative to the resulting CoG when computing torques. In MuJoCo, use body and inertial frames consistently instead of moving the visual model until it looks balanced. The present `body_ipos` randomization does not take effect because compiled same-frame information remains stale. Validate runtime changes by examining effective world inertial positions and the resulting angular acceleration. MuJoCo documents the requirements for updating inertial properties in its [model-editing discussion](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#mjmodel-changes). Rebuilding from an explicit specification is a useful correctness baseline before optimizing reset speed.

**4. Establish an inertia estimate with uncertainty.** Mass tells you how hard the aircraft is to translate; inertia tells you how hard it is to rotate. The central XML specifies diagonal inertia `[0.018,0.018,0.035]` kg·m², but child masses and their offsets add further inertia. Do not treat those three numbers as the final whole-aircraft inertia without checking aggregation.

A component model provides an initial estimate. For a point mass at displacement `r` from the chosen CoG, its inertia contribution is `m*((r·r)*I - r*rᵀ)`. Rotate each component's own inertia into the common frame and add the parallel-axis contribution. Battery and motor masses far from an axis often dominate that axis's inertia. CAD is useful if component masses are measured rather than taken from decorative geometry density.

If you have a suitable mechanical measurement fixture, small-angle compound-pendulum tests provide an independent estimate. For oscillation about a horizontal pivot, with aircraft mass `m`, CoG distance `d` from that pivot, and small-angle period `T`:

```text
I_pivot = m*g*d*T²/(4*pi²)
I_CoG   = I_pivot - m*d²
```

Derivation: small rotation gives restoring torque approximately `-m*g*d*theta`; combine it with `I_pivot*theta_ddot` to obtain the harmonic-oscillator period. Account for fixture inertia and its gravitational restoring contribution if the fixture moves. Do not simply use the bare-aircraft formula for a heavy moving cradle. Measure many cycles and repeat at several small amplitudes to detect friction and large-angle bias. To estimate another axis, change the fixture orientation deliberately. Record limitations instead of reporting false precision.

In simulation, verify inertia using a known torque with motors off and an initially stationary aircraft. At the initial instant, `angular_acceleration ≈ inverse(I)*torque`. At nonzero angular velocity the `omega × (I*omega)` term matters. A test at rest isolates inertia and torque sign without that complication. Angular accelerations should respond inversely to a controlled inertia change.

**5. Measure command-to-thrust behavior before treating PWM as force.** The current model uses `T_i = motor_cmd_i * max_thrust_i`. Firmware turns the same normalized command into a pulse width. Those equations do not establish that a physical ESC/motor/propeller produces linear thrust versus pulse width.

Use a purpose-built or engineering-validated thrust fixture rated above expected loads, with propeller guarding, calibrated load sensing, remote operation and independent power removal. Verify no motor activity on power-up in the selected measurement firmware. Test only within the motor, propeller, ESC and fixture ratings. Do not run the present automatic full harness as the measurement controller, and do not infer fixture adequacy from test-14 comments.

For each motor/propeller/ESC combination, measure actual PWM pulse width, steady thrust, supply voltage at the ESC under load, current, RPM if available, and temperatures relevant to drift. Sweep a bounded set of operating points upward and downward, allowing steady response at each point. Repeat runs, including dense measurements around expected hover. Document propeller model and spin direction. Repeat at representative supply conditions rather than assuming a fixed voltage-to-thrust exponent.

A load-cell reading expressed as equivalent mass converts to force with `F = mass_kg * g`. First establish the fixture's force sign and tare procedure with known loads. A change in a platform-scale reading is not automatically total thrust if restraints or cables carry unknown reaction forces. This is why measurement-fixture mechanics matter.

Fit a monotonic curve `T_i = f_i(pulse_us, voltage)` or a sufficiently supported simpler model. Use separate holdout measurements to evaluate prediction error. A lookup table with interpolation is often easier to audit than a high-order polynomial that behaves badly near endpoints. Record residuals in newtons, repeatability, deadband, hysteresis and the supported range. Do not extrapolate a low-power fit to full throttle.

The no-wind level hover condition is `sum(T_i) = m*g`; equal per-motor thrust is only appropriate when geometry and CoG support it. With tilt, the vertical component of total body-Z thrust is reduced. The same throttle does not maintain altitude at every roll/pitch angle.

**6. Choose and document actuator-command semantics.** Two coherent designs are possible. In a PWM-command design, the controller outputs normalized pulse command and simulation applies the measured nonlinear thrust map. In a thrust-command design, the controller outputs desired thrust and both the real actuator layer and simulation use an explicit inverse calibration/mapping, including saturation and uncertainty. Either can work. The current code implicitly uses PWM semantics on hardware and linear thrust semantics in simulation; that is the mismatch to resolve.

Changing these semantics changes the learned control problem. Preserve the current model as a historical baseline and retrain or carefully warm-start after validating the new mapping. Keep both “requested command” and “achieved/estimated thrust” in diagnostic logs so clipping or deadband is visible.

**7. Measure actuator dynamics separately from the controller's filter.** On the same suitable test stand, command bounded steps around relevant steady operating points. Measure command arrival time and thrust/RPM response. First find pure delay, then rise/settling behavior. A first approximation is:

```text
T_next = T_current + (1 - exp(-dt/tau)) * (T_target - T_current)
```

For a delay-free first-order response, one time constant reaches approximately 63.2% of a step; 10–90% rise time is approximately `2.197*tau`. Fit the delay and time constant jointly when there is a visible initial delay. Check rising versus falling steps, operating-point dependence and repeatability. If the measured response does not resemble first order, do not force that model just to obtain one number.

Store actual thrust as plant state and reset it explicitly. Apply motor dynamics at a documented rate. A software EMA on neural output and a motor's physical lag are distinct states; both contribute phase delay. Fitting one enormous “motor time constant” to hide estimator or UART lag produces the wrong model when those other components change.

**8. Measure or bound yaw reaction torque.** In the current simulator, `Q_i = sign_i * KM_KF_RATIO_i * T_i`, where `KM_KF_RATIO` has units of metres because torque divided by thrust is N·m/N. It is not a dimensionless efficiency. The motor's KV rating does not identify this ratio.

A thrust/torque stand that measures both quantities can estimate `Q/T` across the operating range. Rotor speed measurements permit alternate fits such as `T=k_f*Omega²` and `Q=k_m*Omega²`, using a clearly stated rad/s convention. If torque cannot be measured, keep a justified uncertainty interval and perform sensitivity tests; do not call a guessed constant calibrated. The current values can remain a labeled hypothesis until data exists.

For body X forward, Y left, Z up, positive yaw is counterclockwise viewed from above. A CCW rotor produces an opposite reaction torque on the airframe. The current layout is:

| Motor | Body location | Roll torque from positive thrust | Pitch torque | Yaw reaction sign |
| --- | --- | --- | --- | --- |
| FL, action 0 | `(+a,+a)` | positive | negative | negative, CCW rotor |
| FR, action 1 | `(+a,-a)` | negative | negative | positive, CW rotor |
| RR, action 2 | `(-a,-a)` | negative | positive | negative, CCW rotor |
| RL, action 3 | `(-a,+a)` | positive | positive | positive, CW rotor |

Derive this table from `tau = r × F`, then test it. Increasing the front pair creates negative pitch in this coordinate convention: the nose rises. The code's casual “positive pitch = nose up” descriptions are not a substitute for this derivation. Verify the physical motor order and spin directions independently before connecting them to policy outputs.

**9. Run plant-only simulation tests with the policy completely bypassed.** Disable sensor noise, randomization and command changes; start from a known attitude/velocity; use direct motor/force commands. Keep collision far enough away that it cannot mask the result. The existing `raw_step` helper in `test_env_physics.py` is a starting point, but use a common plant implementation eventually so its force logic cannot diverge from training.

| Test | Expected observation | What a failure points to |
| --- | --- | --- |
| Zero-thrust free fall for 0.2 s from sufficient height | Vertical displacement approximately `-0.5*g*t² = -0.1962 m`; accelerometer near zero specific force | Gravity, sensor interpretation, hidden forces, contacts or timestep |
| Correct level hover thrust | Near-zero vertical acceleration and drift; accelerometer near +1 g on Z | Total mass, thrust scale or applied-force frame |
| Equal thrust increment | Initial vertical acceleration change `4*delta_T/m` | Force scaling/mass |
| Front pair increment | Initial negative pitch acceleration | Motor site positions or coordinate signs |
| Left pair increment | Initial positive roll acceleration | Motor ordering/geometry |
| Diagonal spin-pair increment | Expected yaw sign | Reaction-torque mapping |
| Pure known external torque at rest | `I*omega_dot ≈ torque` | Inertia or torque application |
| Known lateral force | Initial acceleration `F/m` in that world direction | Body/world transform or hidden drag |
| CoG shift with equal thrust | Predictable moment from changed lever arms | Inertial-frame update |
| Repeated reset with identical seed/config | Identical state and trajectory on the pinned environment | Missing reset state or uncontrolled RNG |

Use analytical comparisons over short intervals so numerical differentiation and nonlinear motion do not dominate. Choose tolerances based on timestep and integrator convergence: for a clean short free-fall test, a millimetre-scale position discrepancy should be investigated, not accepted by an arbitrary 20 cm tolerance.

**10. Test numerical convergence without changing control frequency.** Compare physics timesteps of 0.5, 0.25 and 0.125 ms while holding the policy period at 10 ms by changing frameskip consistently. Drive the same force/command sequence and compare short-horizon trajectories, energies where appropriate, and benchmark metrics. Long closed-loop trajectories can diverge from tiny differences, so use initial local errors and distributions as well as pointwise long-time comparison. If metrics shift materially when halving timestep, the original integration configuration is not yet validated.

The current yaw torque is rotated into world coordinates once per policy tick and held over all substeps, while site thrust directions follow the rotating body during integration. At low angular speed that approximation may be small; at high speed test its effect. A body-fixed reaction torque should remain body-fixed throughout physics integration. Apply it through the actuator's torque gear or update the transform at the required physics rate, then verify that yaw is not accidentally applied twice.

**11. Add aerodynamics only after basic mechanics passes.** The current wind is a constant external force, not a calibrated wind-speed model. A 0.5 N test says how the controller handles that force; it does not establish a wind-speed rating. An aerodynamic model would relate relative air velocity to drag, perhaps initially through linear or quadratic fitted terms. Use measured force/velocity data or clearly bounded hypotheses. Avoid adding drag just to hide horizontal drift caused by an unimplemented position-control objective.

Ground effect, propeller interaction, frame flexibility and battery dynamics can matter, but should enter after you have evidence that simpler effects do not explain the discrepancy. Keep separate nominal and uncertainty parameters. A perfect-looking mesh with unmeasured inertia and thrust is less informative than a simple rigid body with validated forces and timing.

For the first software pass, require consistent nominal parameters, explicitly labeled uncertainties and plant-only tests whose outputs agree with mechanics. Continue controller/testing work while physical measurements are pending. To call the plant calibrated, additionally require a measured mass/geometry ledger, an explained inertia estimate, a supported thrust map and bounded actuator lag/torque uncertainty. The mechanical tests should pass regardless of which neural checkpoint is loaded—or whether any network exists.

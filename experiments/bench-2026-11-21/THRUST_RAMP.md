# Thrust ramp on a kitchen scale — 2026-11-21

Harness TEST 16 (`make flash-test TESTS=16 T16_MAX=1500`). Drone standing
free on a stand on a kitchen scale, props on, all four motors at the same
PWM, +25 µs every 4 s. Scale tared with the drone on it, so
thrust = −reading. Raw data: [thrust_ramp_runs.csv](thrust_ramp_runs.csv).
Serial output: `screenlog.0`.

## Data (grams of thrust, all four motors)

| PWM (µs) | Run 1 | Run 2 | az_std run 2 (g) |
|---|---|---|---|
| 1000 | 2 | 0 | 0.001 |
| 1025 | 15 | 16 | 0.018 |
| 1050 | 45 | 47 | 0.062 |
| 1075 | 90 | 90 | 0.103 |
| 1100 | 144 | 135 | 0.093 |
| 1125 | 200 | 195 | 0.118 |
| 1150 | 272 | 258 | 0.150 |
| 1175 | 325 | 320 | 0.182 |
| 1200 | — | 395 | 0.247 |
| 1225 | | 480 | 0.360 |
| 1250 | | 570 | 0.596 |
| 1275 | | 682 | 0.591 |
| 1300 | | — (stopped) | 0.653 |

- Run 1 stopped at 1200 µs on a **false** liftoff trip (vibration aliasing
  into the old 0.12 g / 0.2 s accel check; fixed: 0.35 g over 0.5 s).
- Run 1 readings line up with run 2 **without** the one-step shift that was
  suspected at the time; the two runs agree within ~5 %.
- Run 2 was stopped with the ARM switch at 1300 µs because the drone was
  getting unstable on the stand.

## Thrust curve

Best fit (run 2, 1025–1275 µs, residuals ≤ 8 g):

    thrust_g = 5843 · (u + 0.015)^1.75      u = (PWM − 1000) / 1000

Not linear: exponent ≈ 1.75. The simulator's thrust model is linear.
Slope near the top of the data ≈ 4 g/µs and rising.

## Hover PWM (needs the drone's weight with battery, W — not yet measured)

| W | Power fit | Quadratic fit | Linear from last 3 points |
|---|---|---|---|
| 800 g | 1305 | 1302 | 1305 |
| 900 g | 1327 | 1322 | 1330 |
| 1000 g | 1349 | 1342 | 1355 |
| 1100 g | 1369 | 1361 | 1379 |
| 1200 g | 1389 | 1378 | 1404 |
| 1300 g | 1408 | 1396 | 1429 |
| 1400 g | 1426 | 1412 | 1454 |

Firmware today: base 1323 µs + at most 100 µs from the altitude loop =
**1423 µs ceiling**. Healthy margin needs hover ≤ ~1370 µs.

## Vibration — the bigger finding

`az_std` jumps from 0.25 g at 1200 µs to **0.6 g at 1250–1300 µs**, about
**3× the 0.22 g noise the NN was trained with**, and still well below hover
power. The attitude estimate reported up to 3.6° of tilt with the drone
sitting on the scale. Some of this may be the drone rocking on the stand,
but it needs checking before flight: prop balance, motor/prop screws,
FC soft-mounting, and possibly an accel low-pass filter.

## Open items

1. Weigh the drone with its battery → read the hover PWM off the table.
2. Set the firmware base throttle and the simulator's thrust model
   (exponent ≈ 1.75, hover at the measured PWM) before the next retrain.
3. Track down the vibration jump above 1225 µs.

## Result (weight measured: W = 1110 g with battery)

**Hover ≈ 1372 µs** (power fit; quadratic 1363, linear 1381 → ±10 µs),
battery charge at the time not recorded — expect it to rise toward the end
of a pack.

Consequences for the current firmware/training (base 1323 µs, linear sim):

- **Base throttle 49 µs too low.** The altitude P-loop (0.15 per m) would
  need a 0.33 m error to supply it — i.e. it settles at ~0 m against a
  0.3 m target: the drone sits on the ground.
- **Control sensitivity ~1.45× the simulator's.** Around hover the real
  motors give 4.5·W of thrust per unit throttle vs 3.1·W in the linear sim,
  so every NN correction is ~45 % stronger than trained → overshoot /
  oscillation risk.
- Extrapolated thrust at hover + 0.4 (NN max) ≈ 3.8 kg → T/W ≈ 3.5.

Fix before the next retrain: hover throttle 0.372 in firmware and sim;
sim thrust model T(u) = 5843·(u + 0.015)^1.75 g (scaled to the sim's mass).

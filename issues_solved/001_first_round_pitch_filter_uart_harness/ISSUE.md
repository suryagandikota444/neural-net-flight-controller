# 001 — Pitch axis inverted, slow action filter, blocking UART, untrustworthy test harness

**Status:** solved (2026-11-21) · verified on the bench (harness tests 3, 7, 8)
**Area:** firmware (`fc_config.h`, `main.c`, `uart.c/.h`), test harness (`test_harness.c`), host tests, Python firmware replica (`fc-nn/fw_common.py`)

The drone had never flown on the neural-network firmware. A bench review (props off, USB power)
found four independent problems that each could stop it flying; the fourth one (the harness)
had been hiding the first.

---

## Problem A — pitch was inverted (the drone pushed itself further over)

### What was wrong
The IMU on the Sologood F405 is mounted rotated 180° about Z (Betaflight: `CW180`). Converting
chip axes to body axes for a 180° rotation means negating X and Y for **both** sensors:
`(x, y, z) → (−x, −y, +z)`.

The firmware used two different, inconsistent macros:

| | old | correct |
|---|---|---|
| accel | `( x, −y, z)` | `(−x, −y, z)` |
| gyro  | `(−x,  y, z)` | `(−x, −y, z)` |

The old pair is a **mirror image** (determinant −1), not a rotation. The AHRS was
self-consistent with it, so roll and yaw looked fine, but pitch came out with the opposite sign
from training. The network was trained with "nose down → pitch > 0 → speed up the front
motors". On the drone, nose down reported pitch < 0, so the network sped up the **rear**
motors and pushed the nose further down. That's positive feedback: it flips.

### Evidence (bench, before)
- Harness test 3, nose down: `ax ≈ +0.49 g`, AHRS pitch **negative** (training expects ax < 0, pitch > 0).
- Harness test 8: when the nose was tipped down, the network's pitch correction went the wrong
  way by about 180–290 µs (rear pair up instead of front pair).

### Fix
- New shared header **`fc_config.h`** with one set of alignment macros, applying the same
  rotation to both sensors:
  `ALIGN_AX/AY/AZ = (−x, −y, +z)`, `ALIGN_GX/GY/GZ = (−x, −y, +z)`.
  Only `ALIGN_AX` and `ALIGN_GY` changed sign; roll and yaw are unchanged.
- `main.c` and `test_harness.c` both use it (they previously kept separate copies).
- Host test F (`host_test/test_nn_host.c`) checks the alignment is a proper rotation:
  accel and gyro signs identical, determinant +1, Z up.

### Result (bench, after)
- Test 3: nose down `ax = −0.427 g, pitch +25.2°` ✅ · right down `ay = +0.383 g, roll +22.3°` ✅ · CW `gz = −0.41 rad/s` ✅
- Test 8: nose down → **front +228 µs**, nose up → **rear +175 µs**, right down → **right +404 µs** ✅

---

## Problem D — the action filter was ~4× slower than the network was trained with

### What was wrong
Training smooths the network output with a fixed EMA:
`smooth = 0.3·action + 0.7·smooth` (τ ≈ 28 ms at 100 Hz, 90 % of a step in 7 ticks).

The firmware used an **adaptive** filter: α = 0.08 when the change was < 0.15, else α = 0.60.
Small hover corrections always fall in the 0.08 branch, so τ ≈ 120 ms. Every correction
landed about 4× later than the policy expected, adding a large unmodelled lag to the
attitude loop.

### Fix
- `fc_config.h`: `ACTION_FILTER_ALPHA 0.3f` and `action_filter_step()`, identical to training.
- `main.c` step G uses it. The test harness (tests 7, 9, 11), host tests B–D and the Python
  firmware replica (`fw_common.py`, `FirmwareController(use_adaptive_iir=False)`) all use the
  same filter, and the host tests assert it matches training (90 % in 7 ticks, small steps
  settle in 70 ms).

---

## Problem E — telemetry printing blocked the 100 Hz control loop

### What was wrong
`uart_write_char()` waited for each byte to leave the UART. A ~260-byte telemetry line at
115 200 baud takes ~23 ms to send, which is **more than two whole 10 ms control ticks**,
during which no IMU read, no AHRS update and no motor update happened.

### Fix
- `uart.c/.h`: interrupt-driven **1 KB transmit ring buffer** (USART3 TXE interrupt, IRQ 39).
  Writes copy into the buffer and return; the interrupt drains it.
  - Single producer, ISR consumer, PRIMASK-guarded.
  - It only waits if the buffer is full, and polls a byte out itself if interrupts are disabled,
    so it can't deadlock.
- New `uart_tx_free()` and `uart_tx_flush()`.
- `main.c` TEL block: a line is only started if the whole line fits (`uart_tx_free()`), so the
  loop **never** waits. Skipped lines are counted and reported as `TELDROP,<n>`.

---

## Problem F — the test harness gave wrong answers (it graded the pitch bug as PASS)

### What was wrong
- **Backwards physics in the expectations.**
  - Test 3 expected the inverted pitch signs.
  - Test 8 expected "nose down → rear motors speed up", which is physically backwards, so the
    inverted firmware **passed**.
  - Test 10's expected sides were wrong.
- **Constants drifted from the firmware.** The harness kept its own copies of the alignment,
  gyro scale and altitude gains. It used `ALT_KD = 0.08` while `main.c` used `0.02`, so the
  "hardware-in-the-loop" test 7 wasn't testing the real firmware.
- **Problem C, the attitude estimator freezing during countdowns.** During the
  "3…2…1" prompts the harness stopped calling `ahrs_update()`. When you tilted during the
  countdown, the Madgwick filter (β = 0.05) had to catch up from the accelerometer alone,
  which it can only do at ~2β = 0.1 rad/s ≈ **5.7°/s**. Readings crept up slowly, and leftover
  roll leaked into the next tilt (15–34° of roll during pitch tilts).
- **Test 7 graded absolute PWM** (the network's built-in imbalance swamped everything),
  labelled the yaw pair wrong and never reported pitch balance.
- **Test 5** expected the wrong altitude observation (`obs[8]`).
- **Smaller issues:**
  - test 6 didn't check every golden vector
  - register dumps printed `0x0x`
  - the banner didn't match the tests actually run
  - SysTick was only started inside test 2, so running other tests alone broke
  - nothing stopped test 6 using golden vectors from a different `weights.h`

### Fix (`test_harness.c` rewrite + `Makefile`)
- **Test selection:** `make flash-test TESTS=passive|all|3,7,8`. Motor tests (10, 14, later 16)
  are opt-in only. The banner prints exactly what will run.
- All constants come from `fc_config.h`, the same header `main.c` uses.
- **Correct physics everywhere.** Correcting a tilt means raising the **low** side.
  - Test 3 auto-grades the signs.
  - Test 8 compares each tilt against a **level baseline** and checks that the correct motor
    pair rises ≥ 20 µs more than the opposite pair.
  - Test 7 reports roll / pitch / yaw balance separately.
- **Countdowns keep the attitude estimate running** (`countdown_tracking()`), which fixes the
  creep. Angles now read full value from the first sample.
- Test 5 expects `obs[8] = clamp(−0.3/2) = −0.05`.
- Test 6 checks every golden vector with a relative tolerance.
- **Golden vectors:** the Makefile refuses to build the harness if
  `host_test/golden_vectors.h` came from a different `weights.h` (SHA-256 check,
  `make check-golden`, `make golden`).
- SysTick and DWT are started in `main()`, so any subset of tests runs on its own.

---

## What this round did NOT fix (handled later)
- **The network's own 367 µs motor imbalance at level** (test 7 still FAILed). It came from the
  policy relying on horizontal velocity and an altitude value the real drone can't provide.
  This needed retraining against the "firmware contract" environment, done afterwards together
  with the measured thrust curve (`experiments/bench-2026-11-21/THRUST_RAMP.md`). After that,
  test 7 spread was 18 µs.
- **Hover throttle:** the 1323 µs base was a guess. The kitchen-scale ramp measured
  ≈ 1372 µs. Also, `nn.h` hard-coded `NN_HOVER_THROTTLE 0.3226f`, so exported weights never
  changed the firmware's base until `nn.h` was made to read it from `weights.h`.

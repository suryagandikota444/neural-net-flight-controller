# Issues solved

One folder per issue: `NNN_short_name/ISSUE.md`, plus `evidence/` when there is data
(flight-recorder pulls, bench logs, plots, the script that produced the numbers).

| # | Issue | Status |
|---|---|---|
| [001](001_first_round_pitch_filter_uart_harness/ISSUE.md) | Pitch axis inverted, slow action filter, blocking UART, untrustworthy test harness | solved 2026-11-21 |
| [002](002_attitude_oscillation_2p7hz_actuator_delay/ISSUE.md) | 2.7 Hz roll/pitch oscillation — ~40 ms motor delay not modelled in training | largely fixed 2026-11-22 |
| [003](003_yaw_spin_horizontal_drift_extra_thrust/ISSUE.md) | Yaw spin ~15°/s CW, horizontal drift (no position sensing), hover needs ~28 % more thrust than the scale test | open |

Each ISSUE.md: **Symptom → Evidence → Cause → Fix → Result**.

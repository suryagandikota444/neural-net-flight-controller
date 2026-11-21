**Seed-43 firmware candidate, prepared September 12, 2026.**

`firmware.bin` contains the actor selected at 753,664 additional transitions from `../training_seed43/best_model.zip`. The exact checkpoint is preserved as `policy.zip`; `weights.h`, `source/`, `export_weights.py`, `manifest.json`, `firmware.elf` and `build.log` identify the export and build. The root repository's `weights.h` was not changed. No hardware was flashed during preparation.

The ARM build succeeded. `binary-weights-check.json` checks that all six actor arrays in the selected checkpoint, exported header and raw ARM binary match exactly. The previous numerical inference result is in `../candidate-c-parity.json`. These checks do not establish closed-loop flight readiness.

**This is a bench-test candidate, not a flight-validated image.** Normal startup still automatically spins each motor at 50% command and then arms. Simulation/firmware observation, altitude and filter mismatches remain; telemetry is blocking. The selected policy also fails parts of the full simulation suite. Remove all props and disconnect propulsion power before flashing or inspecting this image. The UART adapter receives telemetry; flashing uses the FC's own USB DFU connection.

From the repo root, with the intended board placed into DFU using its board-specific boot procedure:

```bash
dfu-util -l
dfu-util -a 0 -s 0x08000000:leave \
  -D experiments/2026-09-12-fixes-1-6/deployment_seed43/firmware.bin
```

Confirm the listed device is the intended STM32 board. The download command leaves DFU and may immediately start the firmware. It flashes this already-built binary; `make flash` at the repo root would build from the still-preserved root weights instead.

Read [the hardware guide](../../../docs/system-guide/08-return-to-hardware.md) before progressing to powered tests. A passive mode and explicit arm/stop controls remain required implementation work.

# Makefile for STM32F405 Bare-Metal Flight Controller
#
# Usage:
#   make          — build firmware.bin  (normal flight firmware)
#   make test     — build test.bin      (harness includes powered motor tests)
#   make flash    — build and flash firmware via DFU (dfu-util)
#   make flash-test — build and flash test harness via DFU (dfu-util)
#   make clean    — remove build artifacts
#   make size     — show memory usage
#
# Prerequisites:
#   arm-none-eabi-gcc (ARM cross-compiler toolchain)
#   openocd or st-flash (for flashing)

# === TOOLCHAIN ===
# 'arm-none-eabi-' means: ARM architecture, no operating system, uses newlib ABI
# This is the standard bare-metal ARM cross-compiler prefix.
PREFIX = arm-none-eabi-
CC     = $(PREFIX)gcc
AS     = $(PREFIX)as
LD     = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
SIZE    = $(PREFIX)size

# === TARGET ===
TARGET = firmware
TEST_TARGET ?= test
PYTHON ?= fc-nn/.venv/bin/python
HOST_CC ?= cc

# === SOURCE FILES ===
# Normal flight firmware — full system
SRCS = startup.c clock.c spi.c imu.c uart.c main.c memset.c motor.c ahrs.c nn.c i2c.c baro.c flight_safety.c

# Test harness includes powered tests 10 and 14; it is not passive bench firmware.
SRCS_TEST = startup.c clock.c spi.c imu.c uart.c test_harness.c memset.c ahrs.c nn.c i2c.c baro.c motor.c

# === COMPILER FLAGS ===
CFLAGS  = -mcpu=cortex-m4       # Target the Cortex-M4 core
CFLAGS += -mthumb               # Use Thumb-2 instruction set (all Cortex-M code is Thumb)
CFLAGS += -mfpu=fpv4-sp-d16     # Use the hardware FPU (single-precision, 16 double-word registers)
CFLAGS += -mfloat-abi=hard      # Pass float args in FPU registers (faster than soft-float)
CFLAGS += -O1                   # Mild optimization — enough to be reasonable, not so much it's hard to debug
CFLAGS += -Wall -Wextra         # Enable warnings — catch bugs early
CFLAGS += -std=c99              # C99 standard — we use stdint.h, etc.
CFLAGS += -ffreestanding        # We're not running under an OS — no assumptions about standard library
CFLAGS += -fno-common           # Each variable gets its own section (helps catch duplicate definitions)
CFLAGS += -g                    # Include debug symbols (for GDB). Doesn't increase flash size (stripped by objcopy)

# === LINKER FLAGS ===
LDFLAGS  = -T linker.ld         # Use our linker script
LDFLAGS += -nostartfiles        # Don't link the default C runtime startup (crt0) — we have our own
LDFLAGS += -nostdlib            # Don't link the standard library (no printf, malloc, etc.)
                                # We'll add our own minimal implementations as needed later.
LDFLAGS += -Wl,--gc-sections   # Remove unused sections (keeps binary small)
                                # This is why KEEP() in the linker script is important for the vector table
LDFLAGS += $(CFLAGS)            # Pass arch flags to linker too (for correct library selection)

# Compiler flags for section garbage collection
CFLAGS += -ffunction-sections   # Put each function in its own section
CFLAGS += -fdata-sections       # Put each global in its own section
                                # Combined with --gc-sections, unused code/data is stripped from the binary

# === ESC CALIBRATION MODE ===
# Build with: make ESC_CAL=1    (ESC throttle range calibration — REMOVE PROPS)
# Normal build: make             (flight mode, ESC_CAL_MODE=0)
ESC_CAL ?= 0
ifneq ($(ESC_CAL),0)
CFLAGS += -DESC_CAL_MODE=1
endif

# === DEBUG TELEMETRY ===
# Build with: make DEBUG=1      (enable UART debug output at 2 Hz)
# Normal build: make             (no debug output — zero UART blocking)
DEBUG ?= 0
ifeq ($(DEBUG),0)
CFLAGS += -DDEBUG_INTERVAL=0
endif

# === BUILD RULES ===

# Default target: produce the flight firmware binary
all: $(TARGET).bin

# Link object files into an ELF (contains debug info, section headers, etc.)
HEADERS = $(wildcard *.h)

# Content signatures also handle same-second edits on macOS's older make.
# Only changed source/header/linker/command content forces a fresh link.
FLIGHT_SIGNATURE := $(shell { printf '%s\n' '$(CC) $(CFLAGS) $(LDFLAGS)'; cksum $(SRCS) $(HEADERS) linker.ld Makefile; } | cksum)
TEST_SIGNATURE := $(shell { printf '%s\n' '$(CC) $(CFLAGS) $(LDFLAGS)'; cksum $(SRCS_TEST) $(HEADERS) linker.ld Makefile; } | cksum)
ifneq ($(FLIGHT_SIGNATURE),$(shell cat $(TARGET).build-config 2>/dev/null))
.PHONY: $(TARGET).elf
endif
ifneq ($(TEST_SIGNATURE),$(shell cat $(TEST_TARGET).build-config 2>/dev/null))
.PHONY: $(TEST_TARGET).elf
endif

$(TARGET).build-config: SIGNATURE = $(FLIGHT_SIGNATURE)
$(TEST_TARGET).build-config: SIGNATURE = $(TEST_SIGNATURE)
$(TARGET).build-config $(TEST_TARGET).build-config: FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' '$(SIGNATURE)' > $@.tmp
	@cmp -s $@.tmp $@ || cp $@.tmp $@
	@rm -f $@.tmp

$(TARGET).elf: $(SRCS) $(HEADERS) linker.ld Makefile $(TARGET).build-config
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRCS) -o $@

# Convert ELF to raw binary (just the bytes that go into flash, nothing else)
# -O binary: output format is raw binary
# -S: strip all symbol and relocation information
$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary -S $< $@
	@echo "Build complete: $(TARGET).bin"
	@$(SIZE) $<

# === TEST HARNESS ===
# Builds test_harness.c instead of main.c, including motor.c.
test: $(TEST_TARGET).bin

$(TEST_TARGET).elf: $(SRCS_TEST) $(HEADERS) linker.ld Makefile $(TEST_TARGET).build-config
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRCS_TEST) -o $@

$(TEST_TARGET).bin: $(TEST_TARGET).elf
	$(OBJCOPY) -O binary -S $< $@
	@echo "Build complete: $@ (harness includes powered motor tests)"
	@$(SIZE) $<

# Flash test harness via DFU
flash-test: $(TEST_TARGET).bin
	dfu-util -a 0 -s 0x08000000:leave -D $<

# === UTILITIES ===

# Show memory usage breakdown
size: $(TARGET).elf
	$(SIZE) $<

# Flash via DFU (using dfu-util)
# 0x08000000 is the start of flash — our binary goes right at the beginning
flash: $(TARGET).bin
	dfu-util -a 0 -s 0x08000000:leave -D $<

# === HOST-SIDE TESTS (run on the Mac, no hardware needed) ===
# Layer 0: compiles the REAL nn.c + weights.h natively and checks it against
# golden vectors generated from the trained model by test_nn_parity.py.
host-test: host_test/golden_vectors.h
	$(HOST_CC) -O1 -Wall -Wno-missing-braces -std=c99 host_test/test_nn_host.c -o host_test/test_nn -lm
	./host_test/test_nn

host_test/golden_vectors.h: weights.h fc-nn/generate_golden.py fc-nn/test_nn_parity.py fc-nn/fw_common.py FORCE
	$(PYTHON) fc-nn/generate_golden.py

host-regression:
	$(HOST_CC) -O1 -Wall -Wextra -std=c99 host_test/test_ahrs_host.c -o host_test/test_ahrs -lm
	./host_test/test_ahrs
	$(HOST_CC) -O1 -Wall -Wextra -std=c99 host_test/test_safety_host.c -o host_test/test_safety -lm
	./host_test/test_safety

# Clean build artifacts
clean:
	rm -f $(TARGET).elf $(TARGET).bin $(TARGET).build-config $(TEST_TARGET).elf $(TEST_TARGET).bin $(TEST_TARGET).build-config host_test/test_nn host_test/test_ahrs host_test/test_safety

.PHONY: all clean flash flash-openocd flash-test size test host-test host-regression FORCE
FORCE:

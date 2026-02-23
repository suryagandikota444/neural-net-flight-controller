/*
 * main.c — STM32F405 Sologood FC firmware
 *
 * Revision 4: IMU streaming.
 *
 * Flow:
 *   1. Configure clock to 168 MHz
 *   2. Initialize UART2 (115200 baud, PA2/PA3)
 *   3. Initialize SPI1
 *   4. Detect IMU (MPU6000 / ICM-42688-P / BMI270)
 *   5. Initialize and configure the detected IMU
 *   6. Stream raw sensor data over UART forever
 *
 * Output format (one line per sample, ~100 Hz):
 *
 *   AX:+1234 AY:-567 AZ:+2048 | GX:+12 GY:-34 GZ:+5 | T:+350
 *
 *   AX/AY/AZ — accelerometer raw counts.
 *              ICM-42688-P at ±16 g  → 2048 LSB/g
 *              e.g. AZ≈+2048 when flat (1 g pointing down)
 *
 *   GX/GY/GZ — gyroscope raw counts.  This is rotation RATE (°/s),
 *              NOT yaw/pitch/roll angle.  At ±2000 dps → 16.4 LSB/(°/s).
 *              Angles need integration — that's the attitude step (8).
 *
 *   T        — temperature raw counts.
 *              ICM-42688-P formula: T_°C = raw / 132.48 + 25
 *
 * LED:
 *   Solid on      = streaming normally
 *   1 Hz blink    = heartbeat (every 100 samples)
 *   Rapid flash   = IMU not detected
 *   Very fast     = clock init failed
 */

#include <stdint.h>
#include "clock.h"
#include "spi.h"
#include "imu.h"
#include "uart.h"

/* ============================================================
 * GPIO — LED on PC15
 * ============================================================ */
#define RCC_AHB1ENR          (*(volatile uint32_t *)0x40023830)
#define RCC_AHB1ENR_GPIOCEN  (1U << 2)
#define GPIOC_MODER          (*(volatile uint32_t *)0x40020800)
#define GPIOC_BSRR           (*(volatile uint32_t *)0x40020818)
#define LED_PIN              15

static void led_on(void)  { GPIOC_BSRR = (1U << (LED_PIN + 16)); }
static void led_off(void) { GPIOC_BSRR = (1U << LED_PIN); }

/* ============================================================
 * Delay (spin loop, ~5 cycles/iter at 168 MHz → 33600 iter/ms)
 * ============================================================ */
static void delay_ms(uint32_t ms)
{
    volatile uint32_t n = ms * 33600U;
    while (n--) { __asm volatile ("nop"); }
}

/* ============================================================
 * LED error pattern — rapid flash
 * ============================================================ */
static void blink_error(void)
{
    for (int i = 0; i < 10; i++) {
        led_on();  delay_ms(50);
        led_off(); delay_ms(50);
    }
    delay_ms(500);
}

/* ============================================================
 * Print one raw IMU sample on a single UART line:
 *   AX:+nnn AY:+nnn AZ:+nnn | GX:+nnn GY:+nnn GZ:+nnn | T:+nnn
 * ============================================================ */
static void print_imu(const imu_raw_t *d)
{
    uart2_write_string("AX:");   uart2_write_int32(d->ax);
    uart2_write_string(" AY:");  uart2_write_int32(d->ay);
    uart2_write_string(" AZ:");  uart2_write_int32(d->az);
    uart2_write_string(" | GX:"); uart2_write_int32(d->gx);
    uart2_write_string(" GY:");   uart2_write_int32(d->gy);
    uart2_write_string(" GZ:");   uart2_write_int32(d->gz);
    uart2_write_string(" | T:");  uart2_write_int32(d->temp);
    uart2_write_string("\r\n");
}

/* ============================================================
 * main
 * ============================================================ */
int main(void)
{
    /* ── 1. Clock ───────────────────────────────────────────── */
    int clock_ok = clock_init();

    /* ── LED setup ──────────────────────────────────────────── */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
    (void)RCC_AHB1ENR;
    GPIOC_MODER &= ~(3U << (LED_PIN * 2));
    GPIOC_MODER |=  (1U << (LED_PIN * 2));
    led_off();

    if (clock_ok != 0) {
        while (1) { blink_error(); }
    }

    /* ── 2. UART ────────────────────────────────────────────── */
    uart2_init(115200);
    uart2_write_string("\r\n=== FC BOOT ===\r\n");
    uart2_write_string("Clock: 168MHz OK\r\n");

    /* ── 3. SPI ─────────────────────────────────────────────── */
    delay_ms(100);   /* IMU needs >=100 ms after power-on before SPI */
    spi1_init();
    uart2_write_string("SPI1: OK\r\n");
    delay_ms(10);

    /* ── 4. Detect IMU ──────────────────────────────────────── */
    uint8_t whoami = 0;
    imu_type_t imu = imu_detect(&whoami);

    uart2_write_string("IMU WHO_AM_I: ");
    uart2_write_hex8(whoami);
    uart2_write_string("\r\n");

    switch (imu) {
        case IMU_MPU6000:
            uart2_write_string("IMU: MPU6000\r\n");
            break;
        case IMU_ICM42688P:
            uart2_write_string("IMU: ICM-42688-P\r\n");
            break;
        case IMU_BMI270:
            uart2_write_string("IMU: BMI270\r\n");
            break;
        default:
            uart2_write_string("IMU: NOT DETECTED\r\n");
            uart2_write_string("  0x00 = MISO stuck low (unpowered/disconnected)\r\n");
            uart2_write_string("  0xFF = MISO floating high\r\n");
            while (1) { blink_error(); }
    }

    /* ── 5. Configure IMU ───────────────────────────────────── */
    imu_init(imu);
    uart2_write_string("IMU: configured\r\n");

    /* Column header to orient the terminal output */
    uart2_write_string(
        "--- streaming raw sensor data ---\r\n"
        "accel: ICM=2048 LSB/g (+-16g)  MPU=4096 LSB/g (+-8g)\r\n"
        "gyro : 16.4 LSB/dps (+-2000dps) — rate, NOT angle\r\n"
        "temp : ICM formula: T_C = raw/132.48 + 25\r\n");

    /* ── 6. Stream ──────────────────────────────────────────── */
    led_on();

    imu_raw_t sample;
    uint32_t  n = 0;

    while (1) {
        imu_read_raw(imu, &sample);
        print_imu(&sample);

        /* ~100 Hz pace.  At 115200 baud one line (~52 chars) takes
         * ~4.5 ms to transmit; 6 ms sleep gives ≈100 Hz total.
         * Drop this delay when DMA TX is in place. */
        delay_ms(50);

        /* Heartbeat blink every 100 samples (~1 Hz) */
        if (++n >= 100) {
            n = 0;
            led_off(); delay_ms(2); led_on();
        }
    }

    return 0;
}

/*
 * i2c.c — I2C1 bare-metal driver for STM32F405
 *
 * ── How I2C Works ─────────────────────────────────────────────
 *
 * I2C is a 2-wire bus: SCL (clock) and SDA (data). Both lines
 * are open-drain with pull-up resistors (typically 4.7k–10k).
 * The master (our MCU) generates the clock and initiates transfers.
 *
 * A typical register read:
 *   1. START condition (SDA goes low while SCL is high)
 *   2. Send slave address + WRITE bit (address << 1 | 0)
 *   3. Slave ACKs
 *   4. Send register address
 *   5. Slave ACKs
 *   6. Repeated START
 *   7. Send slave address + READ bit (address << 1 | 1)
 *   8. Slave ACKs
 *   9. Read data byte(s), master ACKs each except the last
 *   10. NACK the last byte
 *   11. STOP condition (SDA goes high while SCL is high)
 *
 * The STM32F4 I2C peripheral handles the low-level protocol
 * (START/STOP generation, ACK/NACK, clock stretching). We just
 * need to write to control registers and wait for status flags.
 *
 * ── Pin Configuration ─────────────────────────────────────────
 *
 * PB8 = SCL (AF4), PB9 = SDA (AF4)
 * Both configured as open-drain with internal pull-ups enabled.
 * Most FC boards also have external pull-ups on the I2C lines.
 *
 * ── Timing (Fast Mode 400 kHz) ────────────────────────────────
 *
 * APB1 clock = 42 MHz
 * CCR = APB1 / (3 × 400 kHz) = 35  (fast mode duty cycle 2:1)
 * TRISE = floor(300 ns × 42 MHz) + 1 = 13
 */

#include <stdint.h>
#include "i2c.h"
#include "clock.h"

/* ============================================================
 * REGISTER DEFINITIONS
 * ============================================================ */

/* RCC */
#define RCC_AHB1ENR      (*(volatile uint32_t *)0x40023830)
#define RCC_APB1ENR      (*(volatile uint32_t *)0x40023840)
#define RCC_APB1RSTR     (*(volatile uint32_t *)0x40023820)
#define RCC_AHB1ENR_GPIOBEN  (1U << 1)
#define RCC_APB1ENR_I2C1EN   (1U << 21)
#define RCC_APB1RSTR_I2C1RST (1U << 21)

/* GPIOB */
#define GPIOB_MODER      (*(volatile uint32_t *)0x40020400)
#define GPIOB_OTYPER     (*(volatile uint32_t *)0x40020404)
#define GPIOB_OSPEEDR    (*(volatile uint32_t *)0x40020408)
#define GPIOB_PUPDR      (*(volatile uint32_t *)0x4002040C)
#define GPIOB_AFRH       (*(volatile uint32_t *)0x40020424)

/* I2C1 (base 0x40005400) */
#define I2C1_CR1         (*(volatile uint32_t *)0x40005400)
#define I2C1_CR2         (*(volatile uint32_t *)0x40005404)
#define I2C1_OAR1        (*(volatile uint32_t *)0x40005408)
#define I2C1_DR          (*(volatile uint32_t *)0x40005410)
#define I2C1_SR1         (*(volatile uint32_t *)0x40005414)
#define I2C1_SR2         (*(volatile uint32_t *)0x40005418)
#define I2C1_CCR         (*(volatile uint32_t *)0x4000541C)
#define I2C1_TRISE       (*(volatile uint32_t *)0x40005420)

/* CR1 bits */
#define I2C_CR1_PE       (1U << 0)    /* Peripheral enable */
#define I2C_CR1_START    (1U << 8)    /* Generate START */
#define I2C_CR1_STOP     (1U << 9)    /* Generate STOP */
#define I2C_CR1_ACK      (1U << 10)   /* ACK enable */
#define I2C_CR1_SWRST    (1U << 15)   /* Software reset */

/* SR1 bits */
#define I2C_SR1_SB       (1U << 0)    /* Start Bit generated */
#define I2C_SR1_ADDR     (1U << 1)    /* Address sent / matched */
#define I2C_SR1_BTF      (1U << 2)    /* Byte Transfer Finished */
#define I2C_SR1_RXNE     (1U << 6)    /* Receive data register not empty */
#define I2C_SR1_TXE      (1U << 7)    /* Transmit data register empty */
#define I2C_SR1_AF       (1U << 10)   /* Acknowledge failure */

/* CCR bits */
#define I2C_CCR_FS       (1U << 15)   /* Fast mode select */


/* ============================================================
 * TIMEOUT — prevents infinite loops on bus errors
 * ============================================================ */
#define I2C_TIMEOUT  50000U

static int wait_flag(volatile uint32_t *reg, uint32_t flag)
{
    uint32_t t = I2C_TIMEOUT;
    while (!(*reg & flag)) {
        if (--t == 0) return -1;
    }
    return 0;
}


/* ============================================================
 * Bus error recovery — toggle SCL to release stuck SDA
 * ============================================================
 *
 * If the MCU resets mid-transfer, the slave may hold SDA low
 * (stuck in the middle of sending a byte). The fix: bit-bang
 * up to 9 SCL clocks as GPIO to let the slave finish its byte,
 * then generate a STOP condition.
 */
static void i2c_bus_recovery(void)
{
    /* Configure PB8 (SCL) as GPIO output, PB9 (SDA) as input */
    uint32_t moder = GPIOB_MODER;
    moder &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    moder |=  (1U << (8 * 2));   /* PB8 = output, PB9 = input */
    GPIOB_MODER = moder;

    #define GPIOB_IDR   (*(volatile uint32_t *)0x40020410)
    #define GPIOB_BSRR  (*(volatile uint32_t *)0x40020418)

    /* Toggle SCL up to 9 times, checking if SDA is released */
    for (int i = 0; i < 9; i++) {
        GPIOB_BSRR = (1U << (8 + 16));  /* SCL low */
        for (volatile int d = 0; d < 100; d++);
        GPIOB_BSRR = (1U << 8);         /* SCL high */
        for (volatile int d = 0; d < 100; d++);
        if (GPIOB_IDR & (1U << 9)) break;  /* SDA released */
    }

    /* Generate STOP: SDA low → SCL high → SDA high */
    /* Set PB9 as output */
    GPIOB_MODER &= ~(3U << (9 * 2));
    GPIOB_MODER |=  (1U << (9 * 2));

    GPIOB_BSRR = (1U << (9 + 16));  /* SDA low */
    for (volatile int d = 0; d < 100; d++);
    GPIOB_BSRR = (1U << 8);         /* SCL high */
    for (volatile int d = 0; d < 100; d++);
    GPIOB_BSRR = (1U << 9);         /* SDA high = STOP */
    for (volatile int d = 0; d < 100; d++);
}


/* ============================================================
 * i2c1_init — Configure I2C1 at 400 kHz on PB8/PB9
 * ============================================================ */
void i2c1_init(void)
{
    /* Enable clocks for GPIOB and I2C1 */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
    RCC_APB1ENR |= RCC_APB1ENR_I2C1EN;
    (void)RCC_APB1ENR;

    /* Peripheral reset (clears any stale state from DFU reflash) */
    RCC_APB1RSTR |= RCC_APB1RSTR_I2C1RST;
    (void)RCC_APB1RSTR;
    RCC_APB1RSTR &= ~RCC_APB1RSTR_I2C1RST;
    (void)RCC_APB1RSTR;

    /* Bus recovery in case SDA is stuck from a prior reset */
    /* First configure pins as GPIO with open-drain + pull-up for recovery */
    GPIOB_OTYPER |= (1U << 8) | (1U << 9);   /* Open-drain */
    GPIOB_PUPDR  &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOB_PUPDR  |=  ((1U << (8 * 2)) | (1U << (9 * 2)));  /* Pull-up */
    GPIOB_OSPEEDR |= ((2U << (8 * 2)) | (2U << (9 * 2)));  /* High speed */

    i2c_bus_recovery();

    /* ── Configure PB8 and PB9 as AF4 (I2C1) ─────────────────
     *
     * GPIOB_AFRH controls AF for pins 8–15.
     * Each pin has 4 AF bits:
     *   PB8: AFRH bits [3:0]  = 4 (AF4 = I2C1)
     *   PB9: AFRH bits [7:4]  = 4 (AF4 = I2C1)
     */
    GPIOB_AFRH &= ~((0xFU << 0) | (0xFU << 4));
    GPIOB_AFRH |=  ((4U   << 0) | (4U   << 4));

    /* Set PB8/PB9 to Alternate Function mode */
    GPIOB_MODER &= ~((3U << (8 * 2)) | (3U << (9 * 2)));
    GPIOB_MODER |=  ((2U << (8 * 2)) | (2U << (9 * 2)));

    /* ── Configure I2C1 peripheral ────────────────────────────── */

    /* Disable I2C before configuration */
    I2C1_CR1 = 0;

    /* Software reset to clear any stuck state */
    I2C1_CR1 = I2C_CR1_SWRST;
    I2C1_CR1 = 0;

    /* Set APB1 frequency in MHz (42 MHz) */
    I2C1_CR2 = APB1_HZ / 1000000U;   /* = 42 */

    /* Fast mode (400 kHz), duty cycle 2:1
     * CCR = APB1 / (3 × 400000) = 42000000 / 1200000 = 35 */
    I2C1_CCR = I2C_CCR_FS | 35U;

    /* Maximum SCL rise time in fast mode: 300 ns
     * TRISE = floor(300 ns × 42 MHz / 1000) + 1 = floor(12.6) + 1 = 13 */
    I2C1_TRISE = 13U;

    /* Enable peripheral */
    I2C1_CR1 = I2C_CR1_PE;
}


/* ============================================================
 * i2c1_write_reg — Write one byte to a device register
 * ============================================================ */
int i2c1_write_reg(uint8_t dev_addr, uint8_t reg, uint8_t val)
{
    /* Clear any pending AF flag */
    I2C1_SR1 &= ~I2C_SR1_AF;

    /* Generate START */
    I2C1_CR1 |= I2C_CR1_START;
    if (wait_flag(&I2C1_SR1, I2C_SR1_SB) < 0) goto fail;

    /* Send address + WRITE */
    I2C1_DR = (uint32_t)(dev_addr << 1);
    if (wait_flag(&I2C1_SR1, I2C_SR1_ADDR) < 0) goto fail;
    (void)I2C1_SR1; (void)I2C1_SR2;  /* Clear ADDR by reading SR1+SR2 */

    /* Send register address */
    I2C1_DR = reg;
    if (wait_flag(&I2C1_SR1, I2C_SR1_TXE) < 0) goto fail;

    /* Send data byte */
    I2C1_DR = val;
    if (wait_flag(&I2C1_SR1, I2C_SR1_BTF) < 0) goto fail;

    /* Generate STOP */
    I2C1_CR1 |= I2C_CR1_STOP;
    return 0;

fail:
    I2C1_CR1 |= I2C_CR1_STOP;
    return -1;
}


/* ============================================================
 * i2c1_read_reg — Read one byte from a device register
 * ============================================================ */
int i2c1_read_reg(uint8_t dev_addr, uint8_t reg, uint8_t *val)
{
    return i2c1_read_burst(dev_addr, reg, val, 1);
}


/* ============================================================
 * i2c1_read_burst — Read multiple bytes from consecutive regs
 * ============================================================
 *
 * The BMP280 auto-increments its register pointer on burst reads,
 * so reading N bytes from register R gives us R, R+1, ..., R+N-1.
 */
int i2c1_read_burst(uint8_t dev_addr, uint8_t reg, uint8_t *buf, int len)
{
    if (len <= 0) return -1;

    /* Clear any pending AF flag */
    I2C1_SR1 &= ~I2C_SR1_AF;

    /* ── Phase 1: Write register address ──────────────────────── */

    I2C1_CR1 |= I2C_CR1_START;
    if (wait_flag(&I2C1_SR1, I2C_SR1_SB) < 0) goto fail;

    I2C1_DR = (uint32_t)(dev_addr << 1);   /* Address + WRITE */
    if (wait_flag(&I2C1_SR1, I2C_SR1_ADDR) < 0) goto fail;
    (void)I2C1_SR1; (void)I2C1_SR2;

    I2C1_DR = reg;
    if (wait_flag(&I2C1_SR1, I2C_SR1_BTF) < 0) goto fail;

    /* ── Phase 2: Repeated START + read ───────────────────────── */

    I2C1_CR1 |= I2C_CR1_START;
    if (wait_flag(&I2C1_SR1, I2C_SR1_SB) < 0) goto fail;

    I2C1_DR = (uint32_t)(dev_addr << 1) | 1U;  /* Address + READ */
    if (len > 1) {
        I2C1_CR1 |= I2C_CR1_ACK;   /* ACK incoming bytes */
    } else {
        I2C1_CR1 &= ~I2C_CR1_ACK;  /* NACK the single byte */
    }
    if (wait_flag(&I2C1_SR1, I2C_SR1_ADDR) < 0) goto fail;
    (void)I2C1_SR1; (void)I2C1_SR2;

    /* Read data bytes */
    for (int i = 0; i < len; i++) {
        if (i == len - 1) {
            /* Last byte: disable ACK, generate STOP */
            I2C1_CR1 &= ~I2C_CR1_ACK;
            I2C1_CR1 |= I2C_CR1_STOP;
        }
        if (wait_flag(&I2C1_SR1, I2C_SR1_RXNE) < 0) goto fail;
        buf[i] = (uint8_t)I2C1_DR;
    }

    return 0;

fail:
    I2C1_CR1 |= I2C_CR1_STOP;
    return -1;
}


/* ============================================================
 * i2c1_probe — Check if a device ACKs at a given address
 * ============================================================
 *
 * Sends START + address byte and checks for ACK or NACK.
 * Returns 0 if device is present, -1 if not (or timeout).
 *
 * Useful for bus scanning to find which devices are connected.
 */
int i2c1_probe(uint8_t dev_addr)
{
    /* Clear any pending AF flag */
    I2C1_SR1 &= ~I2C_SR1_AF;

    /* Generate START */
    I2C1_CR1 |= I2C_CR1_START;
    if (wait_flag(&I2C1_SR1, I2C_SR1_SB) < 0) {
        I2C1_CR1 |= I2C_CR1_STOP;
        return -1;
    }

    /* Send address + WRITE */
    I2C1_DR = (uint32_t)(dev_addr << 1);

    /* Wait for either ADDR (ACK) or AF (NACK) */
    uint32_t t = I2C_TIMEOUT;
    while (t--) {
        uint32_t sr1 = I2C1_SR1;
        if (sr1 & I2C_SR1_ADDR) {
            /* Device ACKed — present */
            (void)I2C1_SR1; (void)I2C1_SR2;  /* Clear ADDR */
            I2C1_CR1 |= I2C_CR1_STOP;
            return 0;
        }
        if (sr1 & I2C_SR1_AF) {
            /* NACK — not present */
            I2C1_SR1 &= ~I2C_SR1_AF;
            I2C1_CR1 |= I2C_CR1_STOP;
            return -1;
        }
    }

    /* Timeout */
    I2C1_CR1 |= I2C_CR1_STOP;
    return -1;
}

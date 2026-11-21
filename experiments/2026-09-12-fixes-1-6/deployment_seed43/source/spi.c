/*
 * spi.c — SPI1 driver for STM32F405 (bare-metal, blocking)
 *
 * SPI1 is connected to the IMU on the Sologood F405:
 *   SCK  = PA5  (AF5)
 *   MISO = PA6  (AF5)
 *   MOSI = PA7  (AF5)
 *   CS   = PA4  (manual GPIO, active low)
 *
 * SPI1 is on the APB2 bus (84MHz after clock_init).
 *
 * This is a blocking (polling) driver — the CPU waits for each byte.
 * Later we'll add DMA-based transfers for the gyro fast path,
 * but polling is simpler to debug and perfectly fine for initialization
 * and configuration register reads.
 *
 * SPI Mode 3 (CPOL=1, CPHA=1) — required by MPU6000, ICM42688P, and BMI270.
 * Mode 3 means: clock idles HIGH, data is sampled on the RISING edge
 * (second edge). Both the master (us) and slave (gyro) agree on when
 * to read the data line.
 */

#include <stdint.h>
#include "spi.h"

/* ============================================================
 * REGISTER DEFINITIONS
 * ============================================================ */

/* --- RCC: Clock enables --- */
#define RCC_AHB1ENR     (*(volatile uint32_t *)0x40023830)
#define RCC_APB2ENR     (*(volatile uint32_t *)0x40023844)
#define RCC_AHB1ENR_GPIOAEN  (1U << 0)   /* GPIOA clock enable */
#define RCC_APB2ENR_SPI1EN   (1U << 12)  /* SPI1 clock enable */

/* --- GPIOA Registers (base 0x40020000) ---
 * We need to configure 4 pins on port A:
 *   PA4 = CS    (GPIO output, manual control)
 *   PA5 = SCK   (AF5 = SPI1)
 *   PA6 = MISO  (AF5 = SPI1)
 *   PA7 = MOSI  (AF5 = SPI1)
 */
#define GPIOA_MODER     (*(volatile uint32_t *)0x40020000)
#define GPIOA_OSPEEDR   (*(volatile uint32_t *)0x40020008)
#define GPIOA_PUPDR     (*(volatile uint32_t *)0x4002000C)
#define GPIOA_ODR       (*(volatile uint32_t *)0x40020014)
#define GPIOA_BSRR      (*(volatile uint32_t *)0x40020018)
#define GPIOA_AFR0      (*(volatile uint32_t *)0x40020020)  /* AF for pins 0-7 */

/* --- SPI1 Registers (base 0x40013000) ---
 *
 * SPI1_CR1 — Control Register 1 (offset 0x00)
 *   Bit 0:  CPHA    — Clock phase: 0=first edge, 1=second edge
 *   Bit 1:  CPOL    — Clock polarity: 0=idle low, 1=idle high
 *   Bit 2:  MSTR    — Master mode: 1=master
 *   Bits [5:3]: BR  — Baud rate prescaler:
 *                      000=÷2, 001=÷4, 010=÷8, 011=÷16,
 *                      100=÷32, 101=÷64, 110=÷128, 111=÷256
 *   Bit 6:  SPE     — SPI enable
 *   Bit 7:  LSBFIRST — 0=MSB first (standard), 1=LSB first
 *   Bit 8:  SSI     — Internal slave select (must be 1 in SW NSS mode)
 *   Bit 9:  SSM     — Software slave management: 1=CS controlled by software
 *   Bit 11: DFF     — Data frame format: 0=8-bit, 1=16-bit
 *
 * SPI1_CR2 — Control Register 2 (offset 0x04)
 *   Bit 0:  RXDMAEN — RX DMA enable
 *   Bit 1:  TXDMAEN — TX DMA enable
 *   (We won't use these yet — DMA comes later)
 *
 * SPI1_SR — Status Register (offset 0x08)
 *   Bit 0:  RXNE    — Receive buffer not empty (data ready to read)
 *   Bit 1:  TXE     — Transmit buffer empty (ready to send)
 *   Bit 7:  BSY     — SPI busy (transfer in progress)
 *
 * SPI1_DR — Data Register (offset 0x0C)
 *   Write to send a byte, read to receive a byte.
 *   Reading DR clears RXNE. Writing DR clears TXE.
 *   IMPORTANT: You must read DR after every write, even if you
 *   don't care about the received byte. Otherwise RXNE stays set
 *   and the overrun flag (OVR) gets set, which can cause issues.
 */
#define SPI1_CR1    (*(volatile uint32_t *)0x40013000)
#define SPI1_CR2    (*(volatile uint32_t *)0x40013004)
#define SPI1_SR     (*(volatile uint32_t *)0x40013008)
#define SPI1_DR     (*(volatile uint32_t *)0x4001300C)

/* 3 bit definitions */
#define SPI_CR1_CPHA     (1U << 0)
#define SPI_CR1_CPOL     (1U << 1)
#define SPI_CR1_MSTR     (1U << 2)
#define SPI_CR1_BR_DIV16 (3U << 3)    /* 84MHz / 16 = 5.25MHz (safe for init) */
#define SPI_CR1_BR_DIV4  (1U << 3)    /* 84MHz / 4 = 21MHz (fast, for gyro reads) */
#define SPI_CR1_SPE      (1U << 6)
#define SPI_CR1_SSI      (1U << 8)
#define SPI_CR1_SSM      (1U << 9)

/* SPI1_SR bit definitions */
#define SPI_SR_RXNE      (1U << 0)
#define SPI_SR_TXE       (1U << 1)
#define SPI_SR_BSY       (1U << 7)


/* ============================================================
 * CHIP SELECT CONTROL
 * ============================================================
 *
 * CS (PA4) is controlled manually via GPIO, not by the SPI
 * peripheral's hardware NSS. This is standard practice because:
 *   1. Hardware NSS deasserts between every byte, but IMU
 *      transactions are multi-byte (address + data).
 *   2. Manual control lets us hold CS low for the entire transaction.
 *
 * We use BSRR (Bit Set/Reset Register) instead of read-modify-write
 * on ODR because BSRR is atomic — it sets or clears a single bit
 * without affecting others, and without needing to read first.
 * This matters in interrupt contexts where ODR read-modify-write
 * could be interrupted between the read and write.
 *
 * BSRR bits [15:0] SET the corresponding pin (write 1 to set high)
 * BSRR bits [31:16] RESET the corresponding pin (write 1 to set low)
 */
static void cs_low(void)
{
    /* Set bit 20 (= 4 + 16) in BSRR → reset pin 4 → PA4 goes LOW */
    GPIOA_BSRR = (1U << (4 + 16));
}

static void cs_high(void)
{
    /* Set bit 4 in BSRR → set pin 4 → PA4 goes HIGH */
    GPIOA_BSRR = (1U << 4);
}


/* ============================================================
 * SPI INITIALIZATION
 * ============================================================ */
void spi1_init(void)
{
    /* ----- Enable peripheral clocks -----
     * GPIOA for the SPI pins and CS.
     * SPI1 for the SPI peripheral itself.
     * Both clocks must be on before touching any registers. */
    RCC_AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC_APB2ENR |= RCC_APB2ENR_SPI1EN;

    /* Small delay for clock propagation */
    (void)RCC_AHB1ENR;
    (void)RCC_APB2ENR;

    /* ----- Configure PA4 as GPIO output (chip select) -----
     *
     * PA4 is NOT an SPI alternate function — it's a plain GPIO
     * that we toggle manually. Set to output, push-pull, high speed.
     * Start with CS HIGH (deselected, since it's active low). */

    /* MODER: pin 4 bits [9:8] = 01 (output) */
    GPIOA_MODER &= ~(3U << (4 * 2));
    GPIOA_MODER |=  (1U << (4 * 2));

    /* High speed for CS (fast edges reduce glitch window) */
    GPIOA_OSPEEDR |= (3U << (4 * 2));

    /* Start with CS high (deselected) */
    cs_high();

    /* ----- Configure PA5, PA6, PA7 as SPI1 alternate function -----
     *
     * All three need:
     *   MODER = 10 (alternate function)
     *   OSPEEDR = 11 (very high speed — SPI clock can be up to 21MHz)
     *   AFR = 5 (AF5 = SPI1 for these pins, from the datasheet AF table)
     *
     * PA5 = SCK  (output from master)
     * PA6 = MISO (input to master — data from gyro)
     * PA7 = MOSI (output from master — data to gyro)
     */
    for (int pin = 5; pin <= 7; pin++) {
        /* Set alternate function mode (10) */
        GPIOA_MODER &= ~(3U << (pin * 2));
        GPIOA_MODER |=  (2U << (pin * 2));

        /* Very high speed */
        GPIOA_OSPEEDR |= (3U << (pin * 2));

        /* AF5 for SPI1 — pins 5,6,7 are in AFR[0] (pins 0-7) */
        GPIOA_AFR0 &= ~(0xFU << (pin * 4));
        GPIOA_AFR0 |=  (5U << (pin * 4));
    }

    /* ----- Configure SPI1 peripheral -----
     *
     * We configure SPI1 as:
     *   - Master mode (we drive the clock)
     *   - Mode 3: CPOL=1 (clock idle high), CPHA=1 (sample on rising/second edge)
     *   - 8-bit data frame (DFF=0, which is default)
     *   - MSB first (LSBFIRST=0, which is default)
     *   - Software slave management (SSM=1, SSI=1) — we control CS via GPIO
     *   - Baud rate: ÷16 = 84MHz/16 = 5.25MHz for initialization
     *
     * We start at a slow SPI speed for reliability during init.
     * The MPU6000 supports up to 1MHz during register config and
     * up to 20MHz for sensor data reads. 5.25MHz works for WHO_AM_I
     * but technically exceeds the MPU6000's 1MHz config limit.
     * In practice, 5.25MHz works for most chips — but if you get
     * wrong WHO_AM_I values, try BR_DIV64 (1.3MHz) or BR_DIV128.
     *
     * The ICM42688P and BMI270 don't have this slow-mode restriction.
     *
     * IMPORTANT: SPI must be disabled (SPE=0) when configuring CR1.
     * We set everything up, then enable SPE at the end.
     */
    SPI1_CR1 = 0;  /* Start clean — disable SPI and clear all config */

    SPI1_CR1 = SPI_CR1_MSTR      /* Master mode */
             | SPI_CR1_CPOL      /* Clock idle HIGH */
             | SPI_CR1_CPHA      /* Sample on second (rising) edge */
             | SPI_CR1_SSM       /* Software slave management */
             | SPI_CR1_SSI       /* Internal slave select HIGH (required when SSM=1,
                                    otherwise the SPI peripheral thinks it's been
                                    selected as a slave and switches to slave mode!) */
             | SPI_CR1_BR_DIV16; /* 84MHz / 16 = 5.25MHz */

    SPI1_CR2 = 0;  /* No DMA, no interrupts — pure polling for now */

    /* Enable SPI */
    SPI1_CR1 |= SPI_CR1_SPE;
}


/* ============================================================
 * SPI TRANSFER — Send and receive one byte
 * ============================================================
 *
 * This is the fundamental SPI operation. Because SPI is full-duplex,
 * sending a byte and receiving a byte are the SAME operation:
 *
 *   1. Wait for TXE (transmit buffer empty) — room to send
 *   2. Write the byte to DR — starts clocking out on MOSI
 *   3. Wait for RXNE (receive buffer not empty) — byte clocked in on MISO
 *   4. Read DR — this is the byte the slave sent simultaneously
 *
 * To READ from the gyro, you send a dummy byte (0x00 or 0xFF)
 * and the received byte is your data.
 *
 * To WRITE to the gyro, you send the real byte and ignore
 * the received byte.
 */
static uint8_t spi1_transfer_byte(uint8_t tx_byte)
{
    /* Wait until transmit buffer is empty.
     * TXE=1 means the TX buffer can accept new data.
     * On a fresh transfer this should already be set. */
    while (!(SPI1_SR & SPI_SR_TXE));

    /* Write byte to data register.
     * This immediately starts the SPI clock and shifts out the byte.
     * We write to DR as uint8_t to ensure an 8-bit transfer.
     * (Writing as uint32_t would also work since DFF=0 means 8-bit,
     * but being explicit is clearer.) */
    *(volatile uint8_t *)&SPI1_DR = tx_byte;

    /* Wait for receive buffer to have data.
     * RXNE=1 means a complete byte has been shifted in on MISO.
     * This happens at the same time as the TX byte finishes shifting out,
     * since SPI is synchronous — same clock drives both directions. */
    while (!(SPI1_SR & SPI_SR_RXNE));

    /* Read and return the received byte.
     * IMPORTANT: You MUST read DR after every transfer, even if you
     * don't need the data. If you skip this, RXNE stays set, and
     * the next received byte triggers an overrun error (OVR flag).
     * Overrun means the new byte is discarded — lost data. */
    return *(volatile uint8_t *)&SPI1_DR;
}


/* ============================================================
 * IMU REGISTER READ
 * ============================================================
 *
 * Protocol for reading one register from the IMU:
 *
 *   CS LOW
 *   Send: (0x80 | register_address)    ← bit 7 HIGH = read operation
 *   Send: 0x00 (dummy)                 ← clock out the response
 *   CS HIGH
 *
 * The byte received during the dummy send is the register value.
 *
 * Why 0x80? The IMU uses bit 7 of the first byte to distinguish
 * read (1) from write (0). Bits 6:0 are the register address.
 * This is a common convention for SPI sensor chips.
 */
uint8_t spi1_read_register(uint8_t reg)
{
    uint8_t value;

    cs_low();

    /* Send read command: set bit 7, keep register address in bits 6:0 */
    spi1_transfer_byte(0x80 | reg);

    /* Clock out the response with a dummy byte */
    value = spi1_transfer_byte(0x00);

    cs_high();

    return value;
}


/* ============================================================
 * IMU REGISTER WRITE
 * ============================================================
 *
 * Protocol for writing one register:
 *
 *   CS LOW
 *   Send: (0x00 | register_address)    ← bit 7 LOW = write operation
 *   Send: value                        ← the data to write
 *   CS HIGH
 *
 * We discard received bytes (they're meaningless during writes).
 */
void spi1_write_register(uint8_t reg, uint8_t value)
{
    cs_low();

    /* Send write command: bit 7 = 0, bits 6:0 = register address */
    spi1_transfer_byte(reg & 0x7F);

    /* Send the value to write */
    spi1_transfer_byte(value);

    cs_high();
}


/* ============================================================
 * BURST READ — Read multiple consecutive registers
 * ============================================================
 *
 * Most IMUs support auto-incrementing the register address:
 * after reading one register, the next byte reads the next
 * register. This is much faster than individual reads because
 * we only send the address once and keep clocking out data.
 *
 * Gyro data is 6 bytes (X_H, X_L, Y_H, Y_L, Z_H, Z_L),
 * and accel data is another 6 bytes. Reading all 12 in a
 * single burst takes ~6μs at 21MHz instead of ~24μs for
 * 12 individual reads.
 *
 * For the control loop at 8kHz (125μs period), this difference
 * matters — every microsecond counts.
 */
void spi1_read_registers(uint8_t start_reg, uint8_t *buffer, uint8_t count)
{
    cs_low();

    /* Send read command with starting register address */
    spi1_transfer_byte(0x80 | start_reg);

    /* Clock out 'count' bytes */
    for (uint8_t i = 0; i < count; i++) {
        buffer[i] = spi1_transfer_byte(0x00);
    }

    cs_high();
}
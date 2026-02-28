/*
 * nn.c — Neural network forward pass for quadrotor control
 *
 * ── How This Works ────────────────────────────────────────────
 *
 * A neural network is just a series of matrix multiplications
 * with a simple nonlinearity (ReLU) between each layer. There's
 * nothing magic about it — it's literally:
 *
 *   for each output neuron j:
 *     sum = bias[j]
 *     for each input i:
 *       sum += weight[j][i] * input[i]
 *     output[j] = max(0, sum)     ← ReLU: if negative, clamp to 0
 *
 * That's it. The "intelligence" is entirely in the WEIGHT VALUES,
 * which were learned during training in Python (PPO reinforcement
 * learning). The 1604 float numbers in weights.h encode everything
 * the network knows about how to fly a quadrotor.
 *
 * Our network architecture:
 *
 *   Layer 1:  12 inputs  → 32 neurons  (384 weights + 32 biases = 416 params)
 *   Layer 2:  32 inputs  → 32 neurons  (1024 weights + 32 biases = 1056 params)
 *   Layer 3:  32 inputs  → 4 outputs   (128 weights + 4 biases = 132 params)
 *                                       Total: 1604 parameters
 *
 * Why ReLU?
 *   ReLU(x) = max(0, x). It's the simplest useful nonlinearity.
 *   Without it, stacking linear layers would collapse into a single
 *   linear transformation (matrix A × matrix B = matrix AB). ReLU
 *   makes the network nonlinear, allowing it to learn complex
 *   input-output relationships. And it's just a comparison — one
 *   CPU cycle on the FPU.
 *
 * Why no activation on the last layer?
 *   The output represents motor commands around hover, which can
 *   be positive or negative. ReLU would clip negative values,
 *   preventing the network from reducing thrust below hover.
 *   So the last layer is linear (no activation function).
 *
 *
 * ── Performance ────────────────────────────────────────────────
 *
 * Layer 1: 32 × 12 = 384 multiply-accumulate operations
 * Layer 2: 32 × 32 = 1024 multiply-accumulate operations
 * Layer 3: 4 × 32 = 128 multiply-accumulate operations
 * Total:   1536 MACs + 64 comparisons (ReLU)
 *
 * The Cortex-M4 FPU does single-precision multiply in 1 cycle
 * and add in 1 cycle (or fused multiply-add in 1 cycle with
 * VFMA). So 1536 MACs ≈ 1536–3072 cycles ≈ 10–20 µs at 168MHz.
 *
 * At 100 Hz (10 ms per iteration), the NN inference takes less
 * than 0.2% of the available compute budget. Tiny.
 *
 *
 * ── Weight Storage ─────────────────────────────────────────────
 *
 * Weights are declared as `const float` in weights.h, which puts
 * them in the .rodata section → stored in flash (1 MB available).
 * Total weight data: 1604 × 4 bytes = 6416 bytes ≈ 6.3 KB.
 *
 * Flash reads have 5 wait states at 168 MHz, but the ART
 * accelerator (instruction cache + prefetch) makes sequential
 * reads nearly zero-wait for the access pattern we have
 * (sequential row-major scan through weight matrices).
 */

#include "nn.h"
#include "weights.h"


/* ============================================================
 * nn_forward — Run the neural network
 * ============================================================
 *
 * This function is deliberately simple and unoptimized for
 * clarity. Potential optimizations for later:
 *   - Use VFMA (fused multiply-add) via __builtin_fmaf
 *   - Unroll inner loops
 *   - Copy weights to CCM RAM (zero-wait-state) at startup
 *   - Use CMSIS-NN library for optimized kernels
 *
 * For now, the naive implementation is fast enough: ~15 µs
 * at 168 MHz, well within the 10 ms control loop budget.
 */
void nn_forward(const float obs[12], float out[4])
{
    float h1[32];   /* Hidden layer 1 output (on stack, in RAM) */
    float h2[32];   /* Hidden layer 2 output (on stack, in RAM) */

    /* ── Layer 1: obs[12] → h1[32] (ReLU) ─────────────────────
     *
     * For each of the 32 neurons in layer 1:
     *   1. Start with the bias value
     *   2. Multiply each input by its weight, accumulate
     *   3. Apply ReLU: if sum < 0, set to 0
     *
     * W1 is shaped [32][12]: row j contains the 12 weights
     * connecting all inputs to neuron j. This is the standard
     * PyTorch weight layout (output_dim × input_dim).
     */
    for (int j = 0; j < 32; j++) {
        float sum = B1[j];
        for (int k = 0; k < 12; k++) {
            sum += W1[j][k] * obs[k];
        }
        h1[j] = (sum > 0.0f) ? sum : 0.0f;   /* ReLU */
    }

    /* ── Layer 2: h1[32] → h2[32] (ReLU) ──────────────────────
     *
     * Same as layer 1, but with 32 inputs instead of 12.
     * This is the "thinking" layer — it combines the features
     * extracted by layer 1 into higher-level representations.
     */
    for (int j = 0; j < 32; j++) {
        float sum = B2[j];
        for (int k = 0; k < 32; k++) {
            sum += W2[j][k] * h1[k];
        }
        h2[j] = (sum > 0.0f) ? sum : 0.0f;   /* ReLU */
    }

    /* ── Layer 3: h2[32] → out[4] (linear) ────────────────────
     *
     * The output layer has NO activation function.
     * Each output directly corresponds to one motor:
     *   out[0] = Front-Left  (CCW)
     *   out[1] = Front-Right (CW)
     *   out[2] = Rear-Right  (CCW)
     *   out[3] = Rear-Left   (CW)
     *
     * These are RESIDUAL commands — centered at 0 (hover).
     * The caller converts them to actual motor commands:
     *   motor_cmd = clamp(HOVER_THR + out * ACT_SCALE, 0, 1)
     */
    for (int i = 0; i < 4; i++) {
        float sum = B3[i];
        for (int k = 0; k < 32; k++) {
            sum += W3[i][k] * h2[k];
        }
        out[i] = sum;   /* Linear — no activation */
    }
}

/*
 * test_nn_host.c — LAYER 0 (C side): compile the ACTUAL flight code on the
 * host and verify it against golden vectors from Python.
 *
 * This compiles the very same nn.c + weights.h that go into the firmware
 * (they are pure C99 with no ARM dependencies), so it proves the C forward
 * pass — loops, indexing, ReLU, weight layout — is bit-for-bit sane.
 *
 * It also replicates main.c's adaptive IIR smoother and residual mixer and
 * characterizes them, because filter lag and mixer clamping are part of the
 * control law whether you like it or not.
 *
 * Build & run:   make host-test
 *   (requires golden_vectors.h — generate with:
 *      cd fc-nn && .venv/bin/python test_nn_parity.py)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../nn.c"              /* the real flight code, weights.h included */
#include "golden_vectors.h"

/* ── Constants mirrored from main.c ─────────────────────────── */
#define ACTION_ALPHA_SLOW   0.08f
#define ACTION_ALPHA_FAST   0.60f
#define ACTION_SNAP_THRESH  0.15f
#define ALT_ADJ_MAX         0.10f

static int failures = 0;

#define CHECK(cond, ...) do {                          \
    if (cond) { printf("  \033[32m✓ PASS\033[0m  ");    \
    } else    { printf("  \033[31m✗ FAIL\033[0m  "); failures++; } \
    printf(__VA_ARGS__); printf("\n");                  \
} while (0)


/* ── main.c's adaptive IIR, verbatim logic ──────────────────── */
static void iir_step(float smoothed[4], const float raw[4])
{
    for (int k = 0; k < 4; k++) {
        float diff  = raw[k] - smoothed[k];
        float alpha = (diff < 0.0f ? -diff : diff) >= ACTION_SNAP_THRESH
                        ? ACTION_ALPHA_FAST
                        : ACTION_ALPHA_SLOW;
        smoothed[k] += alpha * diff;
    }
}


int main(void)
{
    printf("═══ LAYER 0 (C): host-compiled nn.c vs golden vectors ═══\n\n");

    /* ── Test A: forward-pass parity ─────────────────────────── */
    printf("A. nn_forward() parity (%d vectors):\n", N_GOLDEN);
    float worst = 0.0f;
    int   worst_i = 0;
    for (int i = 0; i < N_GOLDEN; i++) {
        float out[4];
        nn_forward(golden_obs[i], out);
        for (int j = 0; j < 4; j++) {
            float d = fabsf(out[j] - golden_out[i][j]);
            if (d > worst) { worst = d; worst_i = i; }
        }
    }
    CHECK(worst < 5e-4f,
          "max |C - numpy| = %.2e (vector '%s'), tolerance 5e-4",
          worst, golden_names[worst_i]);

    /* ── Test B: IIR noise attenuation (slow regime) ────────── */
    printf("\nB. Adaptive IIR — noise regime (|delta| < %.2f):\n",
           ACTION_SNAP_THRESH);
    {
        /* white noise amplitude 0.10 — below snap threshold */
        float sm[4] = {0}, in[4];
        srand(1234);
        double in_var = 0, out_var = 0;
        int n = 5000;
        for (int t = 0; t < n; t++) {
            for (int k = 0; k < 4; k++)
                in[k] = 0.10f * (2.0f * rand() / (float)RAND_MAX - 1.0f);
            iir_step(sm, in);
            in_var  += in[0] * in[0];
            out_var += sm[0] * sm[0];
        }
        float atten = (float)sqrt(in_var / (out_var + 1e-12));
        CHECK(atten > 3.0f,
              "noise attenuation = %.1fx (want > 3x)", atten);
    }

    /* ── Test C: IIR step latency (fast regime) ─────────────── */
    printf("\nC. Adaptive IIR — step response (|delta| > snap):\n");
    {
        float sm[4] = {0};
        float in[4] = {0.5f, 0.5f, 0.5f, 0.5f};
        int t90 = -1;
        for (int t = 1; t <= 100; t++) {
            iir_step(sm, in);
            if (t90 < 0 && sm[0] >= 0.45f) t90 = t;
        }
        CHECK(t90 > 0 && t90 <= 6,
              "reaches 90%% of a 0.5 step in %d ticks = %d ms (want <= 60 ms)",
              t90, t90 * 10);
    }

    /* ── Test D: IIR *sub-threshold* lag — the trap ─────────── */
    printf("\nD. Adaptive IIR — SUB-threshold tracking lag:\n");
    {
        /* A genuine but gentle correction of 0.10 amplitude never crosses
         * SNAP_THRESH, so it gets the SLOW alpha. Compare with the fixed
         * alpha=0.3 EMA the network was TRAINED against (drone_env.py
         * TRAIN_ACTION_ALPHA). If the deployed filter is much slower than
         * the trained one, the NN's learned timing is wrong. */
        float sm_fw = 0.0f, sm_tr = 0.0f;
        int t90_fw = -1, t90_tr = -1;
        for (int t = 1; t <= 200; t++) {
            sm_fw += ((0.10f - sm_fw) >= ACTION_SNAP_THRESH
                      ? ACTION_ALPHA_FAST : ACTION_ALPHA_SLOW) * (0.10f - sm_fw);
            sm_tr += 0.30f * (0.10f - sm_tr);
            if (t90_fw < 0 && sm_fw >= 0.09f) t90_fw = t;
            if (t90_tr < 0 && sm_tr >= 0.09f) t90_tr = t;
        }
        printf("     firmware filter t90 = %d ms, trained filter t90 = %d ms\n",
               t90_fw * 10, t90_tr * 10);
        CHECK(t90_fw <= 2 * t90_tr,
              "deployed lag within 2x of trained lag (%d vs %d ms) "
              "— MISMATCH means the NN flies with a filter it never trained on",
              t90_fw * 10, t90_tr * 10);
    }

    /* ── Test E: mixer headroom arithmetic ──────────────────── */
    printf("\nE. Mixer authority budget (documents the physics ceiling):\n");
    {
        float hover = NN_HOVER_THROTTLE;
        float max_sustained = hover + ALT_ADJ_MAX;          /* NN ≈ 0 at equil. */
        float max_transient = hover + ALT_ADJ_MAX + 1.0f * NN_ACTION_SCALE;
        printf("     base hover cmd      = %.4f  (%.0f us)\n",
               hover, 1000 + hover * 1000);
        printf("     max SUSTAINED cmd   = %.4f  (%.0f us)  "
               "<- P-alt cap, NN residual ~0 at equilibrium\n",
               max_sustained, 1000 + max_sustained * 1000);
        printf("     max transient cmd   = %.4f  (%.0f us)\n",
               max_transient > 1 ? 1.f : max_transient,
               1000 + (max_transient > 1 ? 1.f : max_transient) * 1000);
        printf("     => If the real drone's hover point is above %.0f us,\n"
               "        it CANNOT sustain altitude. Measure it with TEST 14.\n",
               1000 + max_sustained * 1000);
        CHECK(1, "budget printed — compare against TEST 14 measurement");
    }

    printf("\n════════════════════════════════════════════\n");
    printf("Layer 0 (C) result: %s\n",
           failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return failures ? 1 : 0;
}

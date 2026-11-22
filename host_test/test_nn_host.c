/*
 * test_nn_host.c — LAYER 0 (C side): compile the ACTUAL flight code on the
 * host and verify it against golden vectors from Python.
 *
 * This compiles the very same nn.c + weights.h that go into the firmware
 * (they are pure C99 with no ARM dependencies), so it proves the C forward
 * pass — loops, indexing, ReLU, weight layout — is bit-for-bit sane.
 *
 * It also checks the action filter (fc_config.h — the SAME code main.c and
 * the harness compile) against the training filter in drone_env.py, and
 * prints the mixer authority budget.
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

#include "../fc_config.h"       /* action_filter_step, ALT_ADJ_MAX — shared with main.c */

static int failures = 0;

#define CHECK(cond, ...) do {                          \
    if (cond) { printf("  \033[32m✓ PASS\033[0m  ");    \
    } else    { printf("  \033[31m✗ FAIL\033[0m  "); failures++; } \
    printf(__VA_ARGS__); printf("\n");                  \
} while (0)


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

    /* ── Test B: filter == training filter ───────────────────── */
    printf("\nB. Action filter vs training (drone_env.py: s = 0.3*a + 0.7*s):\n");
    {
        float sm[4] = {0}, in[4];
        float ref = 0.0f, worst_f = 0.0f;
        srand(1234);
        for (int t = 0; t < 5000; t++) {
            float a = 2.0f * rand() / (float)RAND_MAX - 1.0f;
            for (int k = 0; k < 4; k++) in[k] = a;
            action_filter_step(sm, in);
            ref = 0.3f * a + 0.7f * ref;
            float d = fabsf(sm[0] - ref);
            if (d > worst_f) worst_f = d;
        }
        CHECK(fabsf(ACTION_FILTER_ALPHA - 0.3f) < 1e-6f && worst_f < 1e-5f,
              "alpha = %.3f, max |firmware - training| = %.1e over 5000 steps",
              ACTION_FILTER_ALPHA, worst_f);
    }

    /* ── Test C: step response ──────────────────────────────── */
    printf("\nC. Action filter — step response:\n");
    {
        float sm[4] = {0};
        float in[4] = {0.5f, 0.5f, 0.5f, 0.5f};
        int t90 = -1;
        for (int t = 1; t <= 100; t++) {
            action_filter_step(sm, in);
            if (t90 < 0 && sm[0] >= 0.45f) t90 = t;
        }
        CHECK(t90 == 7,
              "reaches 90%% of a step in %d ticks = %d ms (alpha 0.3 → 7 ticks)",
              t90, t90 * 10);
    }

    /* ── Test D: small corrections are NOT slowed down ──────── */
    printf("\nD. Action filter — small (0.10) correction, same speed as training:\n");
    {
        /* The old adaptive filter used alpha 0.08 for changes < 0.15,
         * making small corrections ~4x slower than training. With one
         * fixed alpha, a small step must settle exactly as fast as a big one. */
        float sm[4] = {0}, in[4] = {0.10f, 0.10f, 0.10f, 0.10f};
        int t90 = -1;
        for (int t = 1; t <= 200; t++) {
            action_filter_step(sm, in);
            if (t90 < 0 && sm[0] >= 0.09f) t90 = t;
        }
        CHECK(t90 == 7,
              "small step reaches 90%% in %d ms (training: 70 ms)", t90 * 10);
    }

    /* ── Test F: axis alignment is a real rotation ──────────── */
    printf("\nF. IMU axis alignment (fc_config.h) is a proper rotation:\n");
    {
        /* Gyro and accel sit on one die, so the SAME rotation must map both
         * to the body frame, and a rotation has determinant +1. The old
         * macros (accel x,-y,z / gyro -x,y,z) were a mirror image (det -1)
         * and differed between the sensors — that was the pitch inversion. */
        int A[3] = { ALIGN_AX(1), ALIGN_AY(1), ALIGN_AZ(1) };
        int G[3] = { ALIGN_GX(1), ALIGN_GY(1), ALIGN_GZ(1) };
        int same = (A[0] == G[0] && A[1] == G[1] && A[2] == G[2]);
        int det  = A[0] * A[1] * A[2];
        CHECK(same && det == 1 && A[2] == 1,
              "accel signs (%+d,%+d,%+d), gyro signs (%+d,%+d,%+d), det %+d "
              "(want identical, det +1, Z up)",
              A[0], A[1], A[2], G[0], G[1], G[2], det);
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

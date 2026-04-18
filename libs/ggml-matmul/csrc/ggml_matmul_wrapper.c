/*
 * Q4_0 matrix multiply using ggml's full computation graph API.
 *
 * Optimized version: reuses ggml context and work buffer across calls.
 * Creates a fresh lightweight graph per call but avoids context alloc/free.
 *
 * dst[m, n] = input_f32[m, k] × weight_q4[n, k]^T
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-cpu.h"

/* ─── Thread configuration ──────────────────────────────────────────── */

static int g_n_threads = 0;

void ggml_matmul_set_threads(int n) { g_n_threads = n; }
int  ggml_matmul_get_threads(void)  { return g_n_threads; }

/* ─── One-time initialization ───────────────────────────────────────── */

static int g_initialized = 0;

static void ensure_init(void) {
    if (!g_initialized) {
        ggml_cpu_init();
        g_initialized = 1;
        fprintf(stderr, "[ggml] CPU initialized: avx512=%d vnni=%d bf16=%d\n",
                ggml_cpu_has_avx512(),
                ggml_cpu_has_avx512_vnni(),
                ggml_cpu_has_avx512_bf16());
    }
}

static int get_threads(int requested) {
    int nt = requested > 0 ? requested : g_n_threads;
    if (nt <= 0) {
        nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nt <= 0) nt = 16;
    }
    return nt;
}

/* ─── Persistent work buffer ────────────────────────────────────────── */

static uint8_t * g_work_buf = NULL;
static size_t    g_work_cap = 0;

static void ensure_work_buf(size_t needed) {
    if (needed > g_work_cap) {
        free(g_work_buf);
        g_work_cap = needed + (needed >> 2); /* 25% headroom */
        g_work_buf = (uint8_t *)malloc(g_work_cap);
    }
}

/* ─── Public API ────────────────────────────────────────────────────── */

void ggml_q4_mul_mat(
    int m, int k, int n,
    const float * input,
    const void * weight,
    size_t w_nbytes,
    float * output,
    int n_threads
) {
    ensure_init();
    int nt = get_threads(n_threads);
    (void)w_nbytes;

    /* Lightweight context: just tensor descriptors + graph (no data alloc) */
    size_t ctx_size = 3 * ggml_tensor_overhead()
                    + ggml_graph_overhead()
                    + 256;

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    struct ggml_context * ctx = ggml_init(params);

    /* Weight tensor: [n, k] Q4_0 — external data */
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
    A->data = (void *)weight;

    /* Input tensor: [m, k] F32 — external data (const cast safe: ggml won't modify B) */
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    B->data = (void *)input;

    /* Result: [m, n] F32 — write directly to caller's buffer */
    struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    C->data = output;

    /* Build graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, C);

    /* Plan + compute (reuse work buffer) */
    struct ggml_cplan plan = ggml_graph_plan(graph, nt, NULL);
    if (plan.work_size > 0) {
        ensure_work_buf(plan.work_size);
        plan.work_data = g_work_buf;
    }

    ggml_graph_compute(graph, &plan);

    /* Debug: first call only */
    static int debug_done = 0;
    if (!debug_done) {
        debug_done = 1;
        fprintf(stderr, "[ggml] mul_mat OK: m=%d k=%d n=%d threads=%d first4=[%.4f,%.4f,%.4f,%.4f]\n",
                m, k, n, nt, output[0], output[1], output[2], output[3]);
    }

    ggml_free(ctx);
}

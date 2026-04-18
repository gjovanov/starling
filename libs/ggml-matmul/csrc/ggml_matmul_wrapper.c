/*
 * Q4_0 matrix multiply using ggml's full computation graph API.
 *
 * Uses ggml_mul_mat with ggml_graph_compute_with_ctx for correct,
 * optimized Q4×F32 matrix multiplication. This is the same code path
 * llama.cpp uses — handles threading, cache tiling, AVX-512 dispatch.
 *
 * dst[m, n] = input_f32[m, k] × weight_q4[n, k]^T
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ggml headers */
#include "ggml.h"
#include "ggml-cpu.h"

/* Thread count for compute graph. Default to all available CPUs. */
static int g_n_threads = 0; /* 0 = auto-detect */

void ggml_matmul_set_threads(int n) {
    g_n_threads = n;
}

int ggml_matmul_get_threads(void) {
    return g_n_threads;
}

/* One-time initialization */
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

/*
 * ggml_q4_mul_mat: Q4_0 × F32 matrix multiply using ggml graph compute.
 *
 * Arguments:
 *   m        - number of input rows (1 for decode, >1 for prefill)
 *   k        - inner dimension (must be multiple of 32)
 *   n        - number of output features
 *   input    - F32 input matrix [m, k] row-major
 *   weight   - Q4_0 weight data, [n, k] as raw Q4_0 blocks
 *   w_nbytes - byte size of weight data
 *   output   - F32 output matrix [m, n] row-major
 *   n_threads - number of threads (0 = use g_n_threads or auto)
 */
void ggml_q4_mul_mat(
    int m, int k, int n,
    const float * input,
    const void * weight,
    size_t w_nbytes,
    float * output,
    int n_threads
) {
    ensure_init();

    /* Default thread count: all available logical CPUs */
    int nt = n_threads > 0 ? n_threads : g_n_threads;
    if (nt <= 0) {
        nt = sysconf(_SC_NPROCESSORS_ONLN);
        if (nt <= 0) nt = 16; /* fallback */
    }

    /* Context for tensor descriptors and graph only (no data allocation).
     * Actual data is externally managed or stack-allocated.
     */
    size_t ctx_size = 0;
    ctx_size += 3 * ggml_tensor_overhead(); /* A, B, result */
    ctx_size += ggml_graph_overhead();       /* graph */
    ctx_size += 256;                         /* alignment padding */

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true, /* we manage data ourselves */
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[ggml] ERROR: ggml_init failed (requested %zu bytes)\n", ctx_size);
        memset(output, 0, m * n * sizeof(float));
        return;
    }

    /* Create weight tensor A: [n, k] Q4_0 — points to external data */
    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
    A->data = (void *)weight;

    /* Create input tensor B: [m, k] F32 — points to external data (const) */
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    B->data = (void *)input;

    /* Create mul_mat operation: result = A × B^T → [m, n] */
    struct ggml_tensor * result = ggml_mul_mat(ctx, A, B);
    /* Allocate result data — point to caller's output buffer */
    result->data = (void *)output;

    /* Build and compute graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    /* Use ggml_graph_plan + ggml_graph_compute for proper work buffer management */
    struct ggml_cplan plan = ggml_graph_plan(graph, nt, NULL);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }

    enum ggml_status status = ggml_graph_compute(graph, &plan);

    if (plan.work_data) {
        free(plan.work_data);
    }

    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ggml] ERROR: graph compute failed with status %d\n", status);
        memset(output, 0, m * n * sizeof(float));
        ggml_free(ctx);
        return;
    }

    /* Debug: first call validation */
    static int debug_done = 0;
    if (!debug_done) {
        debug_done = 1;
        float max_abs = 0.0f;
        for (int i = 0; i < m * n; i++) {
            float av = output[i] > 0 ? output[i] : -output[i];
            if (av > max_abs) max_abs = av;
        }
        fprintf(stderr, "[ggml] mul_mat OK: m=%d k=%d n=%d threads=%d max_abs=%.4f first4=[%.4f,%.4f,%.4f,%.4f]\n",
                m, k, n, nt, max_abs,
                output[0], output[1], output[2], output[3]);
    }

    ggml_free(ctx);
}

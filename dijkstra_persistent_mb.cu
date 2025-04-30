/**********************************************************************
 *  Cooperative‐launch persistent‐kernel Dijkstra with debugging
 *  Tests multiple graph sizes automatically
 *
 *  Compile:
 *      nvcc -O3 -arch=sm_70 test_grids_coop_debug.cu -o dijkstra_coop_debug \
 *           -Xcompiler "-pthread -lrt -lm"
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define DENSITY      20
#define MAX_WEIGHT   100
#define INF_DIST     1000000000
#define RAND_SEED    1234

typedef int data_t;

/* ---------------- timing helper ---------------------------------- */
double interval(struct timespec start, struct timespec end) {
    struct timespec temp;
    temp.tv_sec  = end.tv_sec  - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec--;
        temp.tv_nsec += 1000000000;
    }
    return (double)temp.tv_sec + (double)temp.tv_nsec * 1.0e-9;
}

/* ---------------- host helpers ----------------------------------- */
void setIntArrayValue(int *a, int n, int v) { for (int i = 0; i < n; ++i) a[i] = v; }
void setDataArrayValue(data_t *a, int n, data_t v) { for (int i = 0; i < n; ++i) a[i] = v; }

static void initializeGraphZero(data_t *g, int n) {
    size_t cells = (size_t)n * n;
    for (size_t i = 0; i < cells; ++i) g[i] = 0;
}

static void constructGraphEdge(data_t *g, int *deg, int n) {
    for (int i = 1; i < n; ++i) {
        int j = rand() % i;
        data_t w = (rand() % MAX_WEIGHT) + 1;
        g[(size_t)i*n + j] = g[(size_t)j*n + i] = w;
        deg[i]++; deg[j]++;
    }
    for (int v = 0; v < n; ++v) {
        while (deg[v] < DENSITY) {
            int u = rand() % n;
            if (u == v || g[(size_t)v*n + u]) continue;
            data_t w = (rand() % MAX_WEIGHT) + 1;
            g[(size_t)v*n + u] = g[(size_t)u*n + v] = w;
            deg[v]++; deg[u]++;
        }
    }
}

static int closest(const data_t *d, const int *vis, int n) {
    data_t best = INF_DIST + 1;
    int idx = -1;
    for (int v = 0; v < n; ++v) {
        if (!vis[v] && d[v] < best) { best = d[v]; idx = v; }
    }
    return idx;
}

static void dijkstraCPU(const data_t *g, data_t *d, int *par, int *vis, int n) {
    for (int i = 0; i < n; ++i) {
        int u = closest(d, vis, n);
        if (u < 0) break;
        vis[u] = 1;
        for (int v = 0; v < n; ++v) {
            if (vis[v]) continue;
            data_t w = g[(size_t)u*n + v];
            if (!w) continue;
            data_t nd = d[u] + w;
            if (nd < d[v]) { d[v] = nd; par[v] = u; }
        }
    }
}

/* ---------------- global device symbol -------------------------- */
__device__ unsigned long long globalmin;

/* ---------------- init kernel ----------------------------------- */
__global__ void initGlobals() {
    unsigned long long init_pack = ((unsigned long long)INF_DIST << 32) | 0xFFFFFFFFull;
    atomicExch(&globalmin, init_pack);
    __threadfence();
}

/* ---------------- Dijkstra kernel (cooperative) ----------------- */
__global__ void dijkstraKernel(const data_t *__restrict__ g,
                               data_t *dist, int *par, int *vis, int n)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ char shmem[];
    int warps_per_block = (blockDim.x + 31) >> 5;

    data_t *shdist = (data_t*)shmem;
    int    *shidx  = (int*)(shmem + warps_per_block * sizeof(data_t));

    int tid    = threadIdx.x;
    int gid    = blockIdx.x*blockDim.x + tid;
    int lane   = tid & 31;
    int warp   = tid >> 5;
    int stride = blockDim.x * gridDim.x;

    while (true) {
        // reduction to find global minimum 
        data_t local_min = INF_DIST; int local_idx = -1;
        for (int v = gid; v < n; v += stride) {
            if (!vis[v] && dist[v] < local_min) {
                local_min = dist[v];
                local_idx = v;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            data_t o  = __shfl_down_sync(0xFFFFFFFFu, local_min, off);
            int    oi = __shfl_down_sync(0xFFFFFFFFu, local_idx, off);
            if (o < local_min) { local_min = o; local_idx = oi; }
        }
        if (lane == 0) {
            shdist[warp] = local_min;
            shidx [warp] = local_idx;
        }
        __syncthreads();

        if (tid < 32) {
            data_t bm = INF_DIST; int bi = -1;
            if (tid < warps_per_block) { bm = shdist[tid]; bi = shidx[tid]; }
            for (int off = 16; off > 0; off >>= 1) {
                data_t o  = __shfl_down_sync(0xFFFFFFFFu, bm, off);
                int    oi = __shfl_down_sync(0xFFFFFFFFu, bi, off);
                if (o < bm) { bm = o; bi = oi; }
            }
            if (tid == 0) {
                unsigned long long pack = ((unsigned long long)bm << 32) | (unsigned long long)bi;
                atomicMin(&globalmin, pack);
            }
        }

        grid.sync();

        // extract winner 
        unsigned long long win = atomicAdd(&globalmin, 0ull);
        int    u      = (int)(win & 0xFFFFFFFFull);
        data_t u_dist = (data_t)(win >> 32);

        if (u == 0xFFFFFFFFu || u_dist >= INF_DIST) {
            grid.sync();
            return;
        }

        bool owner = (u == local_idx && u_dist == local_min);
        if (owner) atomicExch(&vis[u], 1);

        // relax row u 
        for (int v = gid; v < n; v += stride) {
            data_t w = g[(size_t)u*n + v];
            if (!w || vis[v]) continue;
            data_t nd  = u_dist + w;
            data_t old = atomicMin(&dist[v], nd);
            if (nd < old) par[v] = u;
        }

        if (owner)
            atomicExch(&globalmin, ((unsigned long long)INF_DIST << 32) | 0xFFFFFFFFull);

        grid.sync();
    }
}

/* ---------------- driver ---------------------------------------- */
int main(void) {
    int grid_sizes[]  = {16, 32, 64, 128, 200, 256, 400, 512};
    int numTests      = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    const int iterations = 5;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "ERROR: no coop launch support\n");
        return EXIT_FAILURE;
    }

    for (int t = 0; t < numTests; ++t) {
        int G = grid_sizes[t];
        int n = G * G;
        size_t M = (size_t)n * n;
        printf("=== Graph %4dx%4d (%8d) ===\n", G, G, n);

        /* host alloc */
        data_t *h_g    = (data_t*)malloc(M * sizeof(data_t));
        int    *h_deg  = (int*)   calloc(n, sizeof(int));
        data_t *h_ref  = (data_t*)malloc(n * sizeof(data_t));
        data_t *h_dist = (data_t*)malloc(n * sizeof(data_t));
        int    *h_par  = (int*)   malloc(n * sizeof(int));
        int    *h_vis  = (int*)   malloc(n * sizeof(int));

        /* device alloc */
        data_t *d_g, *d_dist; int *d_par, *d_vis;
        cudaMalloc(&d_g,   M * sizeof(data_t));
        cudaMalloc(&d_dist, n * sizeof(data_t));
        cudaMalloc(&d_par,  n * sizeof(int));
        cudaMalloc(&d_vis,  n * sizeof(int));

        double cpuTot  = 0,
               wallTot = 0,
               kernTot = 0;   /* kernel-only accumulator */

        size_t sharedBytes = (((256 + 31) >> 5) * sizeof(data_t))
                           + (((256 + 31) >> 5) * sizeof(int))
                           + (256 * sizeof(data_t));
        void*  args[] = { &d_g, &d_dist, &d_par, &d_vis, &n };

        for (int it = 0; it < iterations; ++it) {

            /* ------- graph generation -------------------------------- */
            setIntArrayValue(h_deg, n, 0);
            srand(RAND_SEED + it);
            initializeGraphZero(h_g, n);
            constructGraphEdge(h_g, h_deg, n);

            /* ------- CPU reference ----------------------------------- */
            setDataArrayValue(h_dist, n, INF_DIST); h_dist[0] = 0;
            setIntArrayValue(h_par,  n, -1);
            setIntArrayValue(h_vis,  n, 0);

            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC_RAW, &a);
            dijkstraCPU(h_g, h_dist, h_par, h_vis, n);
            clock_gettime(CLOCK_MONOTONIC_RAW, &b);
            cpuTot += interval(a, b);
            memcpy(h_ref, h_dist, n * sizeof(data_t));

            /* ------- GPU prep ---------------------------------------- */
            setDataArrayValue(h_dist, n, INF_DIST); h_dist[0] = 0;
            setIntArrayValue(h_par,  n, -1);
            setIntArrayValue(h_vis,  n, 0);
            cudaMemcpy(d_g,   h_g,   M * sizeof(data_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist,h_dist,n * sizeof(data_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_par, h_par, n * sizeof(int),    cudaMemcpyHostToDevice);
            cudaMemset(d_vis, 0,     n * sizeof(int));
            initGlobals<<<1,1>>>();
            cudaDeviceSynchronize();

            /* ------- GPU timing -------------------------------------- */
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);

            clock_gettime(CLOCK_MONOTONIC_RAW, &a);
            cudaEventRecord(e0);
            cudaLaunchCooperativeKernel((void*)dijkstraKernel,
                                        2, 256, args, sharedBytes, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            clock_gettime(CLOCK_MONOTONIC_RAW, &b);

            wallTot += interval(a, b);

            float k_ms;
            cudaEventElapsedTime(&k_ms, e0, e1);
            kernTot += k_ms * 1e-3;

            cudaEventDestroy(e0);
            cudaEventDestroy(e1);

            /* ------- verify ------------------------------------------ */
            cudaMemcpy(h_dist, d_dist, n * sizeof(data_t), cudaMemcpyDeviceToHost);
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                if (h_dist[i] != h_ref[i]) { ok = false; break; }
            }
            printf(" Iter %d: %s\n", it + 1, ok ? "PASS" : "FAIL");
        }

        printf(" Avg CPU        : %8.6f s\n", cpuTot  / iterations);
        printf(" Avg GPU wall   : %8.6f s\n", wallTot / iterations);
        printf(" Avg GPU kernel : %8.6f s\n\n", kernTot / iterations);

        /* cleanup */
        free(h_g); free(h_deg); free(h_ref);
        free(h_dist); free(h_par); free(h_vis);
        cudaFree(d_g); cudaFree(d_dist);
        cudaFree(d_par); cudaFree(d_vis);
    }
    return 0;
}


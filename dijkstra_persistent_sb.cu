/**********************************************************************
 *  Single-block persistent-kernel Dijkstra (dense matrix)
 *
 *  Compile:
 *      nvcc -O3 -arch=sm_70 dijkstra_persistent_sb.cu -o dijkstra_persistent_sb \
 *           -Xcompiler "-pthread -lrt -lm"
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

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
void setIntArrayValue(int *a, int n, int v) {
    for (int i = 0; i < n; ++i) a[i] = v;
}
void setDataArrayValue(data_t *a, int n, data_t v) {
    for (int i = 0; i < n; ++i) a[i] = v;
}

static void initializeGraphZero(data_t *g, int n) {
    size_t cells = (size_t)n * n;
    for (size_t i = 0; i < cells; ++i) g[i] = 0;
}
static void constructGraphEdge(data_t *g, int *deg, int n) {
    for (int i = 1; i < n; ++i) {
        int j = rand() % i, w = (rand() % MAX_WEIGHT) + 1;
        g[(size_t)i*n + j] = g[(size_t)j*n + i] = w;
        deg[i]++; deg[j]++;
    }
    for (int v = 0; v < n; ++v) {
        while (deg[v] < DENSITY) {
            int u = rand() % n;
            if (u == v || g[(size_t)v*n + u]) continue;
            int w = (rand() % MAX_WEIGHT) + 1;
            g[(size_t)v*n + u] = g[(size_t)u*n + v] = w;
            deg[v]++; deg[u]++;
        }
    }
}

static int closest(const data_t *d, const int *vis, int n) {
    data_t best = INF_DIST + 1;
    int idx = -1;
    for (int v = 0; v < n; ++v) {
        if (!vis[v] && d[v] < best) {
            best = d[v];
            idx = v;
        }
    }
    return idx;
}
static void dijkstraCPU(const data_t *g, data_t *d, int *par, int *vis, int n) {
    for (int i = 0; i < n; ++i) {
        int u = closest(d, vis, n);
        if (u == -1) break;
        vis[u] = 1;
        for (int v = 0; v < n; ++v) {
            if (vis[v]) continue;
            data_t w = g[(size_t)u*n + v];
            if (!w) continue;
            data_t nd = d[u] + w;
            if (nd < d[v]) {
                d[v]   = nd;
                par[v] = u;
            }
        }
    }
}

/* ---------------- single-block kernel ---------------------------- */
__global__ void dijkstraKernelSB(
    const data_t *__restrict__ g,
    data_t *dist, int *par, int *vis, int n
) {
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp    = tid >> 5;
    int threads = blockDim.x;

    extern __shared__ char shmem[];
    data_t *shdist = (data_t*)shmem;
    int    *shidx  = (int*)(shmem + sizeof(data_t)*((threads+31)>>5));

    while (true) {
        // local min per thread 
        data_t lmin = INF_DIST; int lidx = -1;
        for (int v = tid; v < n; v += threads) {
            if (!vis[v] && dist[v] < lmin) {
                lmin = dist[v];
                lidx = v;
            }
        }
        // warp reduce 
        for (int off = 16; off > 0; off >>= 1) {
            data_t ov = __shfl_down_sync(0xFFFFFFFFu, lmin, off);
            int    oi = __shfl_down_sync(0xFFFFFFFFu, lidx, off);
            if (ov < lmin) {
                lmin = ov;
                lidx = oi;
            }
        }
        if (lane == 0) {
            shdist[warp] = lmin;
            shidx [warp] = lidx;
        }
        __syncthreads();

        // block reduce
        if (tid < 32) {
            data_t best = INF_DIST; int best_idx = -1;
            int wcnt = (threads+31)>>5;
            if (tid < wcnt) {
                best     = shdist[tid];
                best_idx = shidx[tid];
            }
            for (int off = 16; off > 0; off >>= 1) {
                data_t ov = __shfl_down_sync(0xFFFFFFFFu, best, off);
                int    oi = __shfl_down_sync(0xFFFFFFFFu, best_idx, off);
                if (ov < best) {
                    best     = ov;
                    best_idx = oi;
                }
            }
            if (tid == 0) {
                shdist[0] = best;
                shidx [0] = best_idx;
            }
        }
        __syncthreads();

        int   u      = shidx[0];
        data_t u_dist = shdist[0];
        if (u < 0 || u_dist >= INF_DIST) break;  // done 

        if (tid == 0) vis[u] = 1;  // visited
        __syncthreads();

        // relax all outgoing edges 
        for (int v = tid; v < n; v += threads) {
            data_t w = g[(size_t)u*n + v];
            if (!w || vis[v]) continue;
            data_t nd  = u_dist + w;
            data_t old = atomicMin(&dist[v], nd);
            if (nd < old) par[v] = u;
        }
        __syncthreads();
    }
}

/* ---------------- driver ---------------------------------------- */
int main(void) {
    int grid_sizes[] = {16,32,64,128,200,256,400,512};
    int numTests     = sizeof(grid_sizes)/sizeof(grid_sizes[0]);
    const int iterations = 5;

    for (int t = 0; t < numTests; ++t) {
        int grid = grid_sizes[t];
        int n    = grid*grid;
        size_t M = (size_t)n * n;
        printf("=== Grid %4d×%4d (%8d nodes) ===\n", grid, grid, n);

        /* host alloc */
        data_t *g_h   = (data_t*)malloc(M * sizeof(data_t));
        int    *deg   = (int*)malloc(n * sizeof(int));
        data_t *d_h   = (data_t*)malloc(n * sizeof(data_t));
        int    *p_h   = (int*)malloc(n * sizeof(int));
        int    *v_h   = (int*)malloc(n * sizeof(int));
        data_t *d_gpu = (data_t*)malloc(n * sizeof(data_t));
        data_t *d_ref = (data_t*)malloc(n * sizeof(data_t));

        /* device alloc */
        data_t *g_d, *dist_d;
        int    *par_d, *vis_d;
        cudaMalloc(&g_d,    M * sizeof(data_t));
        cudaMalloc(&dist_d, n * sizeof(data_t));
        cudaMalloc(&par_d,  n * sizeof(int));
        cudaMalloc(&vis_d,  n * sizeof(int));

        double cpuTot = 0, wallTot = 0, kernTot = 0;
        const int threads = 256;
        int warps = (threads + 31) >> 5;
        size_t shm = warps * (sizeof(data_t) + sizeof(int));

        for (int it = 0; it < iterations; ++it) {
            srand(RAND_SEED + it);
            initializeGraphZero(g_h, n);
            setIntArrayValue(deg, n, 0);
            constructGraphEdge(g_h, deg, n);

            /* CPU reference */
            setDataArrayValue(d_h, n, INF_DIST);
            d_h[0] = 0;
            setIntArrayValue(p_h, n, -1);
            setIntArrayValue(v_h, n, 0);
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
            dijkstraCPU(g_h, d_h, p_h, v_h, n);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
            cpuTot += interval(t0, t1);
            memcpy(d_ref, d_h, n * sizeof(data_t));

            /* reset & copy to GPU */
            setDataArrayValue(d_h, n, INF_DIST);
            d_h[0] = 0;
            setIntArrayValue(p_h, n, -1);

            cudaMemcpy(g_d,    g_h,    M * sizeof(data_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dist_d, d_h,    n * sizeof(data_t), cudaMemcpyHostToDevice);
            cudaMemcpy(par_d,  p_h,    n * sizeof(int),    cudaMemcpyHostToDevice);
            // clear visited flags on the device:
            cudaMemset(vis_d, 0, n * sizeof(int));

            /* GPU run */
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
            cudaEventRecord(e0);
            dijkstraKernelSB<<<1,threads,shm>>>(g_d, dist_d, par_d, vis_d, n);
            cudaEventRecord(e1);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
            wallTot += interval(t0, t1);
            float k_ms;
            cudaEventElapsedTime(&k_ms, e0, e1);
            kernTot += k_ms * 1e-3;

            /* verify */
            cudaMemcpy(d_gpu, dist_d, n * sizeof(data_t), cudaMemcpyDeviceToHost);
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                if (d_gpu[i] != d_ref[i]) { ok = false; break; }
            }
            printf("  Iter %d: CPU/GPU %s\n", it+1, ok ? "MATCH" : "MISMATCH");
        }

        printf("Avg CPU        : %10.6f s\n", cpuTot / iterations);
        printf("Avg GPU wall   : %10.6f s\n", wallTot / iterations);
        printf("Avg GPU kernel : %10.6f s\n\n", kernTot / iterations);

        /* cleanup */
        free(g_h); free(deg); free(d_h); free(p_h); free(v_h);
        free(d_gpu); free(d_ref);
        cudaFree(g_d); cudaFree(dist_d);
        cudaFree(par_d); cudaFree(vis_d);
    }
    return 0;
}

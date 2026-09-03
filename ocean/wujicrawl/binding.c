/* Wuji Hand2 CRAWL, GPU-NATIVE via mujoco_warp: free-floating hand learns to
 * crawl on the ground, tracking a commanded planar velocity. Task lives in
 * wujicrawl_warp.py; the C/CUDA mechanism is the shared src/mjwarp_host.cuh
 * (via wujicrawl_host.cu).
 */
#include <string.h>

typedef struct {
    float episode_return;
    float episode_length;
    float perf;   /* mean velocity-tracking kernel over the episode */
    float score;  /* displacement along the commanded heading (meters) */
    float n;      /* required last field */
} Log;

typedef struct {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;
} WujiCrawl;

#define Env WujiCrawl
#define OBS_SIZE 75  /* rot6d 6 | linvel 3 | angvel 3 | z 1 | qpos 20 |
                        qvel 20 | prev 20 | cmd 2 */
#define NUM_ATNS 20
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor
#define MY_GPU_NATIVE 1

void c_reset(WujiCrawl* env);
void c_step(WujiCrawl* env);
void c_close(WujiCrawl* env);
void c_render(WujiCrawl* env);

#include "vecenv.h"

/* Implemented in wujicrawl_host.cu (src/mjwarp_host.cuh): stashes the [env]
 * kwargs Dict; my_gpu_init forwards every key to wujicrawl_warp.py init(). */
void mjwarp_set_env_kwargs(Dict* kwargs);

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    mjwarp_set_env_kwargs(kwargs);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
}

/* CPU-path stubs: never invoked under MY_GPU_NATIVE. */
void c_reset(Env* env) { (void)env; }
void c_step(Env* env) { (void)env; }
void c_close(Env* env) { (void)env; }

/* Live viewer for `puffer eval wujicrawl` (mjwarp_host.cuh -> Python). */
void mjwarp_render(void);
void c_render(Env* env) {
    (void)env;
    mjwarp_render();
}

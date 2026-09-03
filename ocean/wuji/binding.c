/* Wuji Hand2 fingertip-reach: thin vecenv glue over the kalki_env.h C ABI.
 *
 * All physics/task logic lives in libwuji_env.so — the generic core
 * (rl/env/kalki_env.cc) plus the wuji task (ocean/wuji/wuji.cc), built by:
 *   bazel build //rl/PufferLib/ocean/wuji:wuji_env_shared
 * This file only adapts PufferLib's env contract to that ABI; it holds no
 * robot- or task-specific logic beyond sizes and kwarg names.
 */
#include <stdio.h>
#include <stdlib.h>

#include "kalki_env.h"

typedef struct {
    float episode_return;
    float episode_length;
    float perf;
    float score;
    float n;  /* required last field */
} Log;

typedef struct {
    /* Required by vecenv (slices into the flat vec buffers). */
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    KalkiEnv* impl;
} Wuji;

#define Env Wuji
#define OBS_SIZE 75  /* qpos 20 | qvel 20 | prev_action 20 | 5x(tgt-tip) 15 */
#define NUM_ATNS 20
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor

void c_reset(Wuji* env);
void c_step(Wuji* env);
void c_close(Wuji* env);
void c_render(Wuji* env);

#include "vecenv.h"

/* Shared, immutable after first my_init (my_init calls run sequentially). */
static KalkiModel* wuji_model = NULL;

static void shared_init(void) {
    if (wuji_model) return;
    const char* candidates[] = {
        getenv("WUJI_MJCF"),  /* explicit override wins */
        "../../control/robots/wuji_hand2/mjcf_right_with_mount_rl.xml", /* cwd = rl/PufferLib */
        "control/robots/wuji_hand2/mjcf_right_with_mount_rl.xml",       /* cwd = repo root */
    };
    char err[256] = "no candidate path exists";
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (!candidates[i]) continue;
        wuji_model = kalki_model_load(candidates[i], err, sizeof(err));
        if (wuji_model) break;
    }
    if (!wuji_model) {
        fprintf(stderr, "wuji: failed to load model (set WUJI_MJCF): %s\n", err);
        exit(1);
    }
    if (kalki_model_obs_size(wuji_model) != OBS_SIZE ||
        kalki_model_action_size(wuji_model) != NUM_ATNS) {
        fprintf(stderr, "wuji: model obs/action %d/%d != compiled %d/%d\n",
                kalki_model_obs_size(wuji_model),
                kalki_model_action_size(wuji_model), OBS_SIZE, NUM_ATNS);
        exit(1);
    }
}

static void set_param_checked(KalkiEnv* impl, Dict* kwargs, const char* key) {
    if (kalki_env_set_param(impl, key, dict_get(kwargs, key)->value) != 0) {
        fprintf(stderr, "wuji: env rejected param '%s'\n", key);
        exit(1);
    }
}

void my_init(Env* env, Dict* kwargs) {
    shared_init();

    env->num_agents = 1;
    /* vecenv seeds rng with the env index; 0 is fixed up inside the ABI. */
    env->impl = kalki_env_create(wuji_model, env->rng);
    if (!env->impl) {
        fprintf(stderr, "wuji: kalki_env_create failed\n");
        exit(1);
    }

    /* Keys = the [env] section of config/wuji.ini. dict_get asserts if one
     * is missing there; set_param_checked exits if the env rejects one. */
    set_param_checked(env->impl, kwargs, "max_episode_len");
    set_param_checked(env->impl, kwargs, "decimation");
    set_param_checked(env->impl, kwargs, "action_scale");
    set_param_checked(env->impl, kwargs, "w_reach");
    set_param_checked(env->impl, kwargs, "w_action_rate");
    set_param_checked(env->impl, kwargs, "success_radius");
}

void c_reset(Wuji* env) {
    kalki_env_reset(env->impl, env->observations);
}

void c_step(Wuji* env) {
    float reward = 0.0f, done = 0.0f;
    kalki_env_step(env->impl, env->actions, env->observations, &reward, &done);
    env->rewards[0] = reward;
    if (done > 0.5f) {
        env->terminals[0] = 1.0f;
        KalkiEnvLog l;
        kalki_env_log_drain(env->impl, &l);
        env->log.episode_return += l.episode_return;
        env->log.episode_length += l.episode_length;
        env->log.perf += l.perf;
        env->log.score += l.score;
        env->log.n += l.n;
    }
}

void c_close(Wuji* env) {
    kalki_env_close(env->impl);
    env->impl = NULL;
}

void c_render(Wuji* env) {
    (void)env;  /* no renderer; use mujoco.viewer on a checkpoint instead */
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
}

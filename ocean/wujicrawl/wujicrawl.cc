// Copyright 2026 Kalki Robotics, Inc. All Rights Reserved.
//
// Wuji Hand2 CRAWL task, CPU path (kalki_task.h implementation).
//
// Mirrors the GPU-native task in wujicrawl_warp.py: free-floating right hand
// on a ground plane learns to crawl, tracking a commanded planar velocity
// (world frame) sampled per episode. Keep the two in sync — this is the
// C-MuJoCo reference for sim2sim checks against mjwarp and for consumers of
// the kalki_env.h ABI without CUDA.
//
// Model: control/robots/wuji_hand2/mjcf_right_crawl_rl.xml (built by
// make_crawl_scene.py): free joint + 20 finger DOF, settled palm-down pose
// stored as keyframe "home" (required — its quat defines the "down" axis for
// the flip check). Free-joint qvel: linear world-frame, angular body-frame
// (only |omega|^2 is used, which is frame-invariant).
//
//   reward = max(0, w_track * exp(-|v_xy - cmd|^2 / 0.002) + w_vel * progress
//                + w_alive + w_action_rate * rate + w_qvel * qv_pen
//                + w_ang_vel * |omega|^2)
//            + w_flip * [flipped]
//
// This hand is PASSIVELY STABLE: doing nothing costs nothing physically, so
// the reward must make standing still clearly unprofitable. The tracking
// kernel is sharp (sigma^2 = 0.002) and the dominant term is linear progress
// toward the command, CLIPPED at the commanded speed. The per-step total is
// floored at 0 (a terminal bootstraps V=0, so sustained negative income
// makes suicide-by-flip optimal); the one-time flip penalty is the only
// negative reward the policy can see.
//
// Known gap vs the GPU task: no w_overflow term. kalki_env.cc clamps actions
// to [-1,1] before the task sees anything (kalki_task_reward only receives
// the post-clamp action rate), so the raw-output overflow penalty cannot be
// reproduced here without changing the generic env.
//
// Task observations (2): commanded (vx, vy), world frame. Base pose/velocity
// arrive via the generic qpos | qvel block. Note qpos includes world x, y —
// a non-stationarity leak the curated GPU observation avoids.

#include <cmath>
#include <cstdio>
#include <cstring>

#include <mujoco/mujoco.h>

#include "kalki_task.h"

namespace {

constexpr int kNq = 27;  // 7 free + 20 fingers
constexpr int kNv = 26;  // 6 free + 20 fingers
constexpr int kNu = 20;

// Resolved once by kalki_task_init; read-only afterwards.
int g_home_key = -1;
mjtNum g_down_body[3];        // settled pose's floor-facing axis, body frame
int g_finger_qposadr[kNu];    // actuator i -> qpos address of its joint
int g_finger_dofadr[kNu];     // actuator i -> dof address of its joint

// Global across envs, set via kalki_task_set_param. Defaults mirror
// config/wujicrawl.ini.
float g_w_track = 1.0f;
float g_w_vel = 4.0f;
float g_w_alive = 0.1f;
float g_w_action_rate = -0.01f;
float g_w_ang_vel = -0.001f;
float g_w_qvel = -0.005f;
float g_w_flip = -10.0f;
float g_cmd_vx_lo = -0.15f;  // backward (-x): the easy direction for this hand
float g_cmd_vx_hi = -0.08f;

float g_cmd_vy_max = 0.0f;
float g_reset_noise = 0.2f;
float g_success_vel_radius = 0.05f;  // m/s, |v_xy - cmd| for success logging

// Per-env task state: this episode's velocity command.
struct CrawlState {
  float cmd[2];
};

}  // namespace

int kalki_task_init(const mjModel* m, char* err, int err_sz) {
  auto fail = [&](const char* msg) {
    if (err && err_sz > 0) snprintf(err, err_sz, "wujicrawl: %s", msg);
    return -1;
  };

  if (m->nq != kNq || m->nv != kNv || m->nu != kNu) {
    return fail("expected free-joint crawl model (nq 27, nv 26, nu 20); "
                "generate it with make_crawl_scene.py");
  }
  if (m->njnt < 1 || m->jnt_type[0] != mjJNT_FREE || m->jnt_qposadr[0] != 0) {
    return fail("first joint must be the free base joint");
  }
  for (int i = 0; i < kNu; i++) {
    if (m->actuator_trntype[i] != mjTRN_JOINT) {
      return fail("all actuators must be joint transmissions");
    }
    int j = m->actuator_trnid[2 * i];
    if (j <= 0 || j >= m->njnt || m->jnt_type[j] != mjJNT_HINGE) {
      return fail("actuators must drive finger hinge joints");
    }
    g_finger_qposadr[i] = m->jnt_qposadr[j];
    g_finger_dofadr[i] = m->jnt_dofadr[j];
  }

  g_home_key = mj_name2id(m, mjOBJ_KEY, "home");
  if (g_home_key < 0) {
    return fail("missing keyframe 'home' (make_crawl_scene.py)");
  }

  // Body-frame "down" axis of the settled pose: rotate world -z into the
  // settled base frame. The flip check asks when it points up instead.
  mjtNum wq[4] = {m->key_qpos[g_home_key * m->nq + 3],
                  m->key_qpos[g_home_key * m->nq + 4],
                  m->key_qpos[g_home_key * m->nq + 5],
                  m->key_qpos[g_home_key * m->nq + 6]};
  mju_normalize4(wq);
  mjtNum conj[4];
  mju_negQuat(conj, wq);
  const mjtNum down_world[3] = {0.0, 0.0, -1.0};
  mju_rotVecQuat(g_down_body, down_world, conj);
  return 0;
}

int kalki_task_state_size(const mjModel* m) {
  (void)m;
  return (int)sizeof(CrawlState);
}

int kalki_task_obs_size(const mjModel* m) {
  (void)m;
  return 2;  // command (vx, vy)
}

int kalki_task_set_param(const char* key, double value) {
  if (!strcmp(key, "w_track")) g_w_track = (float)value;
  else if (!strcmp(key, "w_vel")) g_w_vel = (float)value;
  else if (!strcmp(key, "w_alive")) g_w_alive = (float)value;
  else if (!strcmp(key, "w_action_rate")) g_w_action_rate = (float)value;
  else if (!strcmp(key, "w_ang_vel")) g_w_ang_vel = (float)value;
  else if (!strcmp(key, "w_qvel")) g_w_qvel = (float)value;
  else if (!strcmp(key, "w_flip")) g_w_flip = (float)value;
  else if (!strcmp(key, "cmd_vx_lo")) g_cmd_vx_lo = (float)value;
  else if (!strcmp(key, "cmd_vx_hi")) g_cmd_vx_hi = (float)value;
  else if (!strcmp(key, "cmd_vy_max")) g_cmd_vy_max = (float)value;
  else if (!strcmp(key, "reset_noise")) g_reset_noise = (float)value;
  else if (!strcmp(key, "success_vel_radius")) g_success_vel_radius = (float)value;
  else return -1;
  return 0;
}

// Settled keyframe + reset randomization (finger pose jitter within actuator
// limits, small random base push) + fresh command. Every env starting from
// the identical frozen pose makes the stand-still optimum stickier; noise
// decorrelates the batch and lets early policies experience "already moving"
// states where the progress reward is visible.
void kalki_task_reset(const mjModel* m, mjData* d, void* state,
                      unsigned int* rng) {
  CrawlState* s = (CrawlState*)state;

  mj_resetDataKeyframe(m, d, g_home_key);
  mju_zero(d->qvel, m->nv);

  for (int i = 0; i < kNu; i++) {
    mjtNum lo = m->actuator_ctrlrange[2 * i];
    mjtNum hi = m->actuator_ctrlrange[2 * i + 1];
    mjtNum j = d->qpos[g_finger_qposadr[i]] +
               (mjtNum)kalki_rng_uniform(rng, -g_reset_noise, g_reset_noise);
    d->qpos[g_finger_qposadr[i]] = j < lo ? lo : (j > hi ? hi : j);
  }
  // Small random planar base push (up to ~0.5 * reset_noise m/s).
  d->qvel[0] = (mjtNum)kalki_rng_uniform(rng, -0.5f * g_reset_noise,
                                         0.5f * g_reset_noise);
  d->qvel[1] = (mjtNum)kalki_rng_uniform(rng, -0.5f * g_reset_noise,
                                         0.5f * g_reset_noise);

  s->cmd[0] = kalki_rng_uniform(rng, g_cmd_vx_lo, g_cmd_vx_hi);
  s->cmd[1] = kalki_rng_uniform(rng, -g_cmd_vy_max, g_cmd_vy_max);
}

void kalki_task_obs(const mjModel* m, const mjData* d, const void* state,
                    float* out) {
  (void)m;
  (void)d;
  const CrawlState* s = (const CrawlState*)state;
  out[0] = s->cmd[0];
  out[1] = s->cmd[1];
}

float kalki_task_reward(const mjModel* m, const mjData* d, void* state,
                        float action_rate, int* success, int* terminate) {
  (void)m;
  const CrawlState* s = (const CrawlState*)state;

  const float vx = (float)d->qvel[0];  // world frame
  const float vy = (float)d->qvel[1];
  const float wx = (float)d->qvel[3];  // body frame; |omega|^2 only
  const float wy = (float)d->qvel[4];
  const float wz = (float)d->qvel[5];

  // Flip check: the body axis that pointed at the floor in the settled pose
  // now points up -> the hand is on its back.
  mjtNum q[4] = {d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]};
  mju_normalize4(q);
  mjtNum down_world[3];
  mju_rotVecQuat(down_world, g_down_body, q);
  *terminate = down_world[2] > 0.5;

  // Sharp tracking kernel: standing still under the minimum 0.08 m/s
  // command earns only ~0.04; see the header comment on passive stability.
  const float dvx = vx - s->cmd[0];
  const float dvy = vy - s->cmd[1];
  const float track = expf(-(dvx * dvx + dvy * dvy) / 0.002f);

  // Linear progress toward the command, clipped at the commanded speed
  // (linear gradient from v=0, no overshoot incentive beyond the command).
  const float cnorm =
      sqrtf(s->cmd[0] * s->cmd[0] + s->cmd[1] * s->cmd[1]);
  float progress = 0.0f;
  if (cnorm > 1e-6f) {
    progress = (vx * s->cmd[0] + vy * s->cmd[1]) / cnorm;
    progress = progress < cnorm ? progress : cnorm;
  }

  // Joint-velocity limit penalty (bounded): thrashing costs, gait speeds
  // (< 6 rad/s) are free. Cap 25/joint — see header on the suicide exploit.
  float qv_pen = 0.0f;
  for (int i = 0; i < kNu; i++) {
    float ex = fabsf((float)d->qvel[g_finger_dofadr[i]]) - 6.0f;
    if (ex > 0.0f) {
      float p = ex * ex;
      qv_pen += p < 25.0f ? p : 25.0f;
    }
  }

  float reward = g_w_track * track + g_w_vel * progress + g_w_alive +
                 g_w_action_rate * action_rate + g_w_qvel * qv_pen +
                 g_w_ang_vel * (wx * wx + wy * wy + wz * wz);
  if (reward != reward) {  // never leak NaN into the trainer or the logs
    reward = 0.0f;
  }
  // Floor at 0, then the one-time flip penalty (see header comment).
  if (reward < 0.0f) {
    reward = 0.0f;
  }
  if (*terminate) {
    reward += g_w_flip;
  }

  const float r = g_success_vel_radius;
  *success = (dvx * dvx + dvy * dvy) < r * r;
  return reward;
}

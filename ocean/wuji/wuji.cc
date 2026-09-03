// Copyright 2026 Kalki Robotics, Inc. All Rights Reserved.
//
// Wuji Hand2 fingertip-reach task (kalki_task.h implementation).
//
// Each episode samples a random joint configuration within joint limits,
// runs FK, and takes the five fingertip positions as targets — every goal is
// reachable by construction. Reward drives all five fingertips to their
// targets and penalizes action rate.
//
// Model: control/robots/wuji_hand2/mjcf_{left,right}[_with_mount].xml
// (20 DOF / 20 position actuators). Fingertip sites are resolved with both
// r_/l_ prefixes, so either hand loads unchanged. Optional keyframe "home"
// for the start pose.
//
// Task observations (15): 5 x (target - fingertip).
// Params: w_reach, w_action_rate, success_radius.

#include <cmath>
#include <cstdio>
#include <cstring>

#include <mujoco/mujoco.h>

#include "kalki_task.h"

namespace {

constexpr int kNumFingers = 5;
const char* const kTipSiteNames[kNumFingers] = {
    "thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip",
    "pinky_tip"};

// Resolved once by kalki_task_init; read-only afterwards.
int g_tip_site[kNumFingers];
int g_home_key = -1;

// Global across envs, set via kalki_task_set_param.
float g_w_reach = 1.0f;
float g_w_action_rate = -0.01f;
float g_success_radius = 0.02f;  // meters, per fingertip

// Per-env task state: this episode's fingertip goals.
struct WujiState {
  float target[3 * kNumFingers];
};

// Fingertip sites carry a hand-side prefix (r_thumb_tip / l_thumb_tip).
int find_tip_site(const mjModel* m, const char* name) {
  char buf[64];
  snprintf(buf, sizeof(buf), "r_%s", name);
  int id = mj_name2id(m, mjOBJ_SITE, buf);
  if (id >= 0) return id;
  snprintf(buf, sizeof(buf), "l_%s", name);
  return mj_name2id(m, mjOBJ_SITE, buf);
}

void reset_pose(const mjModel* m, mjData* d) {
  if (g_home_key >= 0) {
    mj_resetDataKeyframe(m, d, g_home_key);
  } else {
    mj_resetData(m, d);
  }
}

}  // namespace

int kalki_task_init(const mjModel* m, char* err, int err_sz) {
  for (int f = 0; f < kNumFingers; f++) {
    g_tip_site[f] = find_tip_site(m, kTipSiteNames[f]);
    if (g_tip_site[f] < 0) {
      if (err && err_sz > 0) {
        snprintf(err, err_sz, "wuji: missing fingertip site {r_|l_}%s",
                 kTipSiteNames[f]);
      }
      return -1;
    }
  }
  g_home_key = mj_name2id(m, mjOBJ_KEY, "home");
  return 0;
}

int kalki_task_state_size(const mjModel* m) {
  (void)m;
  return (int)sizeof(WujiState);
}

int kalki_task_obs_size(const mjModel* m) {
  (void)m;
  return 3 * kNumFingers;
}

int kalki_task_set_param(const char* key, double value) {
  if (!strcmp(key, "w_reach")) g_w_reach = (float)value;
  else if (!strcmp(key, "w_action_rate")) g_w_action_rate = (float)value;
  else if (!strcmp(key, "success_radius")) g_success_radius = (float)value;
  else return -1;
  return 0;
}

void kalki_task_reset(const mjModel* m, mjData* d, void* state,
                      unsigned int* rng) {
  WujiState* s = (WujiState*)state;

  // Sample a random joint configuration within limits and record its
  // fingertip positions as this episode's targets — feasible by
  // construction. FK only needs frames, not full dynamics.
  reset_pose(m, d);
  for (int j = 0; j < m->njnt; j++) {
    if (m->jnt_type[j] != mjJNT_HINGE && m->jnt_type[j] != mjJNT_SLIDE) {
      continue;  // leave free/ball joints at their reset pose
    }
    float lo = -0.5f, hi = 0.5f;
    if (m->jnt_limited[j]) {
      lo = (float)m->jnt_range[2 * j];
      hi = (float)m->jnt_range[2 * j + 1];
    }
    d->qpos[m->jnt_qposadr[j]] = (mjtNum)kalki_rng_uniform(rng, lo, hi);
  }
  mj_kinematics(m, d);
  for (int f = 0; f < kNumFingers; f++) {
    const mjtNum* p = &d->site_xpos[3 * g_tip_site[f]];
    for (int i = 0; i < 3; i++) s->target[3 * f + i] = (float)p[i];
  }

  // Restore the start pose (the env runs mj_forward after we return).
  reset_pose(m, d);
}

void kalki_task_obs(const mjModel* m, const mjData* d, const void* state,
                    float* out) {
  (void)m;
  const WujiState* s = (const WujiState*)state;
  int k = 0;
  for (int f = 0; f < kNumFingers; f++) {
    const mjtNum* p = &d->site_xpos[3 * g_tip_site[f]];
    for (int i = 0; i < 3; i++) {
      out[k++] = s->target[3 * f + i] - (float)p[i];
    }
  }
}

float kalki_task_reward(const mjModel* m, const mjData* d, void* state,
                        float action_rate, int* success, int* terminate) {
  (void)m;
  const WujiState* s = (const WujiState*)state;

  // Mean over fingers of a kernel that is 1 at the target. tanh(10*d)
  // saturates ~0.3 m out; fingertip errors live at the cm scale.
  float reach = 0.0f;
  bool all_within = true;
  for (int f = 0; f < kNumFingers; f++) {
    const mjtNum* tip = &d->site_xpos[3 * g_tip_site[f]];
    float dx = s->target[3 * f + 0] - (float)tip[0];
    float dy = s->target[3 * f + 1] - (float)tip[1];
    float dz = s->target[3 * f + 2] - (float)tip[2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    reach += 1.0f - tanhf(10.0f * dist);
    all_within = all_within && dist < g_success_radius;
  }
  reach /= (float)kNumFingers;

  *success = all_within;
  *terminate = 0;  // reach-and-hold: episodes run to the time limit
  return g_w_reach * reach + g_w_action_rate * action_rate;
}

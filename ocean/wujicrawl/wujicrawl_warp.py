"""Wuji Hand2 CRAWL: free-floating hand locomotion on the ground, GPU-native
via mujoco_warp (in-process mjwarp, one captured CUDA graph per rollout
buffer, C launches them GIL-free; C side is the shared src/mjwarp_host.cuh).

Model: control/robots/wuji_hand2/mjcf_right_crawl_rl.xml (built by
make_crawl_scene.py): right hand with a free joint on a ground plane,
settled palm-down pose stored as keyframe "home".
  nq = 7 + 20 (free joint + fingers), nv = 6 + 20, nu = 20.

Task: track a commanded planar velocity, sampled per episode in POLAR form:
speed uniform in [cmd_speed_lo, cmd_speed_hi], heading uniform in a sector of
half-width cmd_heading_jitter_deg around cmd_heading_deg. The speed FLOOR is
load-bearing — the box sampling it replaces (vx, vy drawn independently from
+/-0.08 and +/-0.15) left 24% of episodes with |cmd| < 0.06 m/s, and under a
sigma^2 = 0.002 tracking kernel a near-zero command is maximized by freezing.
A quarter of the batch was training the exact stand-still optimum every other
term in this reward exists to defeat.

Commands point along the Y AXIS by default (cmd_heading_deg = -90). At the
settled pose the wrist's body -z axis — the fingertip direction — points
along world -y, so a y command is served by the MCP/PIP/DIP FLEXORS (+/-2 N.m
forcerange at the MCP) pulling the palm toward planted fingertips. An x
command runs across the knuckles and has to come from the ABDUCTION joints,
forcerange +/-0.2 N.m: ten times weaker. That actuator asymmetry, not a quirk
of the search, is why y is the easy direction for this morphology.

cmd_frame picks the frame the command and the tracked velocity live in:
  0 = world (default) — heading measured from world +x, fixed all episode.
  1 = heading — measured from the hand's own facing (the body axis that
      pointed along world +x at the settled pose), so a y command stays on
      the fingertip axis as the hand yaws. In world frame the command is
      pinned while the hand rotates under it, so any yaw drift the gait
      induces steadily rotates the task off the flexor axis that made y
      easy in the first place.

  reward = max(0, w_track * exp(-|v_track - cmd|^2 / 0.002) + w_vel * progress
               + w_alive + w_action_rate * rate + w_overflow * over
               + w_qvel * qv_pen + w_ang_vel * |omega_body|^2)
           + w_flip * [flipped]
  (floored at 0 so living never pays worse than a terminal's V=0 bootstrap;
  the one-time flip penalty is the only negative reward.)
  terminate on flip (settled "down" axis points up) or divergence;
  truncate at max_episode_len.

score integrates velocity along the commanded direction over the episode
(meters, unclipped). Final-position displacement is only correct when the
commanded world direction is constant; integrating stays correct under
cmd_frame = 1, where it rotates with the hand.

Observation (75):
  base rot 6D (2 world-frame rotation matrix columns)  6
  base linear velocity (xy in the cmd_frame, z world)  3
  base angular velocity, body frame                    3
  base height z                                        1
  finger qpos                                         20
  finger qvel                                         20
  prev_action                                         20
  command (vx, vy), world frame                        2
"""

import os

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

NQ = 27   # 7 free + 20 fingers
NV = 26   # 6 free + 20 fingers
NU = 20
FQ = 7    # finger qpos offset
FV = 6    # finger qvel offset
OBS = 6 + 3 + 3 + 1 + 20 + 20 + 20 + 2  # 75

state = {}  # holds refs to everything the captured graphs point into


def _model_path():
    for p in (os.environ.get("WUJI_CRAWL_MJCF"),
              "../../control/robots/wuji_hand2/mjcf_right_crawl_rl.xml",
              "control/robots/wuji_hand2/mjcf_right_crawl_rl.xml"):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("wujicrawl: MJCF not found (set WUJI_CRAWL_MJCF); "
                            "generate it with make_crawl_scene.py")


@wp.kernel
def k_pre(actions: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
          prev: wp.array2d(dtype=float), rate: wp.array(dtype=float),
          over: wp.array(dtype=float),
          ctrl_lo: wp.array(dtype=float), ctrl_hi: wp.array(dtype=float),
          action_scale: float):
    e = wp.tid()
    r = float(0.0)
    ov = float(0.0)
    for i in range(NU):
        raw = actions[e, i]
        if raw != raw:
            raw = 0.0
        ex = wp.abs(raw) - 1.0
        if ex > 0.0:
            # Huber wall: quadratic near the boundary, linear past ex=2.
            # Bounded slope, but a restoring gradient at ANY overshoot — the
            # old hard cap min(ex^2, 4) had zero gradient past |raw|~3, and
            # the mean sat at |mu|~1e5 with nothing pulling it back.
            if ex < 2.0:
                ov += ex * ex
            else:
                ov += 4.0 * ex - 4.0
        a = wp.clamp(raw, -1.0, 1.0)
        mid = 0.5 * (ctrl_lo[i] + ctrl_hi[i])
        half = 0.5 * (ctrl_hi[i] - ctrl_lo[i])
        ctrl[e, i] = mid + a * action_scale * half
        da = a - prev[e, i]
        r += da * da
        prev[e, i] = a
    rate[e] = r
    over[e] = ov


@wp.func
def base_quat(qpos: wp.array2d(dtype=float), e: int) -> wp.quat:
    # MuJoCo stores wxyz; warp quats are xyzw.
    return wp.quat(qpos[e, 4], qpos[e, 5], qpos[e, 6], qpos[e, 3])


@wp.func
def xs32(x: wp.uint32) -> wp.uint32:
    x ^= x << wp.uint32(13)
    x ^= x >> wp.uint32(17)
    x ^= x << wp.uint32(5)
    return x


@wp.func
def u01(x: wp.uint32) -> float:
    return float(x >> wp.uint32(8)) * (1.0 / 16777216.0)


# Planar velocity in the frame commands are expressed in. cmd_frame 0 leaves
# it in world; 1 rotates it into the hand's heading frame, where "heading" is
# the world-xy direction of fwd_body (the body axis that pointed along world
# +x at the settled pose). Uses the sin/cos of the yaw directly off that
# vector — no atan2, and no branch on quadrant. Degenerate when the hand is
# nose-up/nose-down and fwd has no xy component; world frame there.
@wp.func
def track_vel(q: wp.quat, linvel: wp.vec3, fwd_body: wp.vec3,
              cmd_frame: int) -> wp.vec3:
    vx = linvel[0]
    vy = linvel[1]
    if cmd_frame != 0:
        f = wp.quat_rotate(q, fwd_body)
        n = wp.sqrt(f[0] * f[0] + f[1] * f[1])
        if n > 1.0e-6:
            c = f[0] / n
            s = f[1] / n
            vx = c * linvel[0] + s * linvel[1]
            vy = -s * linvel[0] + c * linvel[1]
    return wp.vec3(vx, vy, linvel[2])


# Episode reset shared by k_reset and k_post's auto-reset: settled keyframe
# + reset randomization (finger pose jitter within actuator limits, small
# random base push) + fresh command. Every env starting from the identical
# frozen pose makes the stand-still optimum stickier; noise decorrelates the
# batch and lets early policies experience "already moving" states where the
# progress reward is visible. Returns the advanced rng state.
@wp.func
def reset_env(e: int,
              qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
              prev: wp.array2d(dtype=float), cmd: wp.array2d(dtype=float),
              key_qpos: wp.array(dtype=float),
              ctrl_lo: wp.array(dtype=float), ctrl_hi: wp.array(dtype=float),
              reset_noise: float, cmd_speed_lo: float, cmd_speed_hi: float,
              cmd_heading: float, cmd_jitter: float,
              x: wp.uint32) -> wp.uint32:
    for i in range(NQ):
        qpos[e, i] = key_qpos[i]
    for i in range(NV):
        qvel[e, i] = 0.0
    for i in range(NU):
        prev[e, i] = 0.0
        x = xs32(x)
        j = key_qpos[FQ + i] + reset_noise * (2.0 * u01(x) - 1.0)
        qpos[e, FQ + i] = wp.clamp(j, ctrl_lo[i], ctrl_hi[i])
    # Small random planar base push (up to ~0.5 * reset_noise m/s).
    x = xs32(x)
    qvel[e, 0] = 0.5 * reset_noise * (2.0 * u01(x) - 1.0)
    x = xs32(x)
    qvel[e, 1] = 0.5 * reset_noise * (2.0 * u01(x) - 1.0)
    # Polar: a speed FLOOR (cmd_speed_lo > 0) guarantees every episode is
    # off-command while standing still. Sampling vx and vy independently in
    # boxes does not — it concentrates mass near |cmd| = 0, where freezing
    # maximizes the tracking kernel.
    x = xs32(x)
    speed = cmd_speed_lo + u01(x) * (cmd_speed_hi - cmd_speed_lo)
    x = xs32(x)
    ang = cmd_heading + cmd_jitter * (2.0 * u01(x) - 1.0)
    cmd[e, 0] = speed * wp.cos(ang)
    cmd[e, 1] = speed * wp.sin(ang)
    return x


@wp.func
def write_crawl_obs(obs: wp.array2d(dtype=float), e: int,
                    q: wp.quat, tvel: wp.vec3, angvel: wp.vec3, z: float,
                    qpos: wp.array2d(dtype=float),
                    qvel: wp.array2d(dtype=float),
                    prev: wp.array2d(dtype=float),
                    cmd: wp.array2d(dtype=float)):
    cx = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    cy = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    obs[e, 0] = cx[0]
    obs[e, 1] = cx[1]
    obs[e, 2] = cx[2]
    obs[e, 3] = cy[0]
    obs[e, 4] = cy[1]
    obs[e, 5] = cy[2]
    # xy in the same frame the command and the reward use (see track_vel),
    # so the policy is not left to infer the rotation from rot6d itself; z
    # is world either way.
    obs[e, 6] = tvel[0]
    obs[e, 7] = tvel[1]
    obs[e, 8] = tvel[2]
    obs[e, 9] = 0.25 * angvel[0]
    obs[e, 10] = 0.25 * angvel[1]
    obs[e, 11] = 0.25 * angvel[2]
    obs[e, 12] = z
    k = int(13)
    for i in range(NU):
        obs[e, k] = qpos[e, FQ + i]
        k += 1
    for i in range(NU):
        obs[e, k] = 0.1 * qvel[e, FV + i]  # O(1) scale for the network
        k += 1
    for i in range(NU):
        obs[e, k] = prev[e, i]
        k += 1
    obs[e, k] = cmd[e, 0]
    obs[e, k + 1] = cmd[e, 1]


@wp.kernel
def k_post(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
           prev: wp.array2d(dtype=float), rate: wp.array(dtype=float),
           over: wp.array(dtype=float),
           cmd: wp.array2d(dtype=float),
           tick: wp.array(dtype=int), track_sum: wp.array(dtype=float),
           prog_sum: wp.array(dtype=float),
           ep_ret: wp.array(dtype=float), rng: wp.array(dtype=wp.uint32),
           obs: wp.array2d(dtype=float), rewards: wp.array(dtype=float),
           terminals: wp.array(dtype=float),
           key_qpos: wp.array(dtype=float), down_body: wp.vec3,
           fwd_body: wp.vec3,
           ctrl_lo: wp.array(dtype=float), ctrl_hi: wp.array(dtype=float),
           log: wp.array(dtype=float),
           max_len: int, w_track: float, w_vel: float, w_alive: float,
           w_rate: float, w_ang: float, cmd_speed_lo: float,
           cmd_speed_hi: float, cmd_heading: float, cmd_jitter: float,
           cmd_frame: int, dt: float, reset_noise: float, w_overflow: float,
           w_qvel: float, w_flip: float):
    e = wp.tid()

    bad = int(0)
    for i in range(NQ):
        v = qpos[e, i]
        if v != v or wp.abs(v) > 1.0e6:
            bad = 1

    q = base_quat(qpos, e)
    linvel = wp.vec3(qvel[e, 0], qvel[e, 1], qvel[e, 2])   # world frame
    angvel = wp.vec3(qvel[e, 3], qvel[e, 4], qvel[e, 5])   # body frame
    z = qpos[e, 2]

    # Flip check: the body axis that pointed at the floor in the settled pose
    # now points up -> the hand is on its back.
    down_world = wp.quat_rotate(q, down_body)
    flipped = int(0)
    if down_world[2] > 0.5:
        flipped = 1

    # This hand is PASSIVELY STABLE: unlike a biped, doing nothing costs
    # nothing physically, so the reward must make standing still clearly
    # unprofitable. Tracking kernel is sharp (sigma^2 = 0.002: standing under
    # the slowest sampled command, cmd_speed_lo = 0.08 m/s, earns only ~0.04
    # — which is why that floor must stay above ~0.06), and the dominant term
    # is linear progress toward the command, CLIPPED at the commanded speed
    # (linear gradient from v=0, no overshoot incentive beyond it).
    tv = track_vel(q, linvel, fwd_body, cmd_frame)
    dvx = tv[0] - cmd[e, 0]
    dvy = tv[1] - cmd[e, 1]
    track = wp.exp(-(dvx * dvx + dvy * dvy) / 0.002)
    cnorm = wp.sqrt(cmd[e, 0] * cmd[e, 0] + cmd[e, 1] * cmd[e, 1])
    vdot = float(0.0)  # signed speed along the command, unclipped (score)
    progress = float(0.0)
    if cnorm > 1.0e-6:
        vdot = (tv[0] * cmd[e, 0] + tv[1] * cmd[e, 1]) / cnorm
        if vdot != vdot:  # guard before it accumulates into prog_sum
            vdot = 0.0
        progress = wp.min(vdot, cnorm)
    # w_overflow penalizes RAW policy outputs beyond [-1,1] (see k_pre):
    # without it the action means drift far outside the clamp, where PPO's
    # gradient signal about behavior vanishes (observed |mu| ~ 38).
    # Joint-velocity limit penalty (bounded): thrashing costs, gait speeds
    # (< 6 rad/s) are free. Cap 25/joint — at 100 the penalty dominated the
    # whole reward (observed ~-13/step) and made ending the episode optimal.
    qv_pen = float(0.0)
    for i in range(NU):
        exq = wp.abs(qvel[e, FV + i]) - 6.0
        if exq > 0.0:
            qv_pen += wp.min(exq * exq, 25.0)
    reward = (w_track * track + w_vel * progress + w_alive + w_rate * rate[e]
              + w_overflow * over[e] + w_qvel * qv_pen
              + w_ang * wp.dot(angvel, angvel))
    if reward != reward:  # never leak NaN into the trainer or the logs
        reward = 0.0
    # Floor at 0: a terminal bootstraps V=0, so any sustained negative
    # income makes suicide-by-flip the optimum (observed: flip in 0.25 s).
    # Living must never pay worse than being dead; penalties shape within
    # the positive region, and the one-time flip penalty below is the only
    # negative reward the policy can ever see.
    reward = wp.max(reward, 0.0)
    if flipped == 1:
        reward += w_flip

    t = tick[e] + 1
    ts = track_sum[e] + track
    ps = prog_sum[e] + vdot * dt  # meters travelled along the command
    ret = ep_ret[e] + reward

    done = int(0)
    if bad == 1 or flipped == 1 or t >= max_len:
        done = 1

    rewards[e] = reward
    terminals[e] = float(done)

    if done == 1:
        wp.atomic_add(log, 0, ret)                # episode_return
        wp.atomic_add(log, 1, float(t))           # episode_length
        wp.atomic_add(log, 2, ts / float(t))      # perf: mean tracking kernel
        # score: metres travelled along the commanded direction, integrated
        # over the episode. Final-position displacement only works while the
        # commanded WORLD direction is constant; under cmd_frame = 1 it
        # rotates with the hand, and the integral is the frame-correct form.
        # Positive is always good, whatever heading was sampled.
        wp.atomic_add(log, 3, ps)
        wp.atomic_add(log, 4, 1.0)                # n
        rng[e] = reset_env(e, qpos, qvel, prev, cmd, key_qpos, ctrl_lo,
                           ctrl_hi, reset_noise, cmd_speed_lo, cmd_speed_hi,
                           cmd_heading, cmd_jitter, rng[e])
        q = base_quat(qpos, e)
        tv = track_vel(q, wp.vec3(qvel[e, 0], qvel[e, 1], 0.0), fwd_body,
                       cmd_frame)
        angvel = wp.vec3(0.0, 0.0, 0.0)
        z = qpos[e, 2]
        t = 0
        ts = 0.0
        ps = 0.0
        ret = 0.0

    tick[e] = t
    track_sum[e] = ts
    prog_sum[e] = ps
    ep_ret[e] = ret

    write_crawl_obs(obs, e, q, tv, angvel, z, qpos, qvel, prev, cmd)


@wp.kernel
def k_reset(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
            prev: wp.array2d(dtype=float), cmd: wp.array2d(dtype=float),
            tick: wp.array(dtype=int), track_sum: wp.array(dtype=float),
            prog_sum: wp.array(dtype=float),
            ep_ret: wp.array(dtype=float), rng: wp.array(dtype=wp.uint32),
            obs: wp.array2d(dtype=float),
            key_qpos: wp.array(dtype=float), fwd_body: wp.vec3,
            ctrl_lo: wp.array(dtype=float), ctrl_hi: wp.array(dtype=float),
            cmd_speed_lo: float, cmd_speed_hi: float, cmd_heading: float,
            cmd_jitter: float, cmd_frame: int, reset_noise: float):
    e = wp.tid()
    rng[e] = reset_env(e, qpos, qvel, prev, cmd, key_qpos, ctrl_lo, ctrl_hi,
                       reset_noise, cmd_speed_lo, cmd_speed_hi, cmd_heading,
                       cmd_jitter, rng[e])
    tick[e] = 0
    track_sum[e] = 0.0
    prog_sum[e] = 0.0
    ep_ret[e] = 0.0
    q = base_quat(qpos, e)
    tv = track_vel(q, wp.vec3(qvel[e, 0], qvel[e, 1], 0.0), fwd_body,
                   cmd_frame)
    write_crawl_obs(obs, e, q, tv, wp.vec3(0.0, 0.0, 0.0), qpos[e, 2],
                    qpos, qvel, prev, cmd)


def _wrap(ptr, shape, dev):
    return wp.array(ptr=ptr, dtype=wp.float32, shape=shape, device=dev)


def init(total_agents, num_buffers, seed,
         act_ptr, obs_ptr, rew_ptr, term_ptr,
         max_episode_len, decimation, action_scale,
         w_track, w_vel, w_alive, w_action_rate, w_ang_vel,
         cmd_speed_lo, cmd_speed_hi, cmd_heading_deg, cmd_heading_jitter_deg,
         cmd_frame, reset_noise, w_overflow, w_qvel, w_flip):
    wp.init()
    dev = "cuda:0"
    assert total_agents % num_buffers == 0
    apb = total_agents // num_buffers

    mjm = mujoco.MjModel.from_xml_path(_model_path())
    assert mjm.nq == NQ and mjm.nv == NV and mjm.nu == NU, (mjm.nq, mjm.nv, mjm.nu)
    key = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key >= 0, "crawl model must have keyframe 'home' (make_crawl_scene.py)"
    key_qpos_np = mjm.key_qpos[key].astype(np.float32)

    # Body-frame "down" axis of the settled pose (for the flip check):
    # rotate world -z into the settled base frame.
    wq = key_qpos_np[3:7].astype(np.float64)  # wxyz
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, wq)
    R = rot.reshape(3, 3)
    down_body_np = R.T @ np.array([0.0, 0.0, -1.0])
    # Body-frame heading reference: the body axis that pointed along world +x
    # at the settled pose. Its world-xy direction is the hand's "facing", so
    # heading 0 means the same thing in both cmd_frames at reset.
    fwd_body_np = R.T @ np.array([1.0, 0.0, 0.0])

    cmd_speed_lo = float(cmd_speed_lo)
    cmd_speed_hi = float(cmd_speed_hi)
    assert 0.0 < cmd_speed_lo <= cmd_speed_hi, (
        "cmd_speed_lo must be > 0 (a zero-speed command makes standing still "
        "optimal under the tracking kernel) and <= cmd_speed_hi")
    cmd_heading = np.radians(float(cmd_heading_deg))
    cmd_jitter = np.radians(float(cmd_heading_jitter_deg))
    # Control-step duration, for integrating the score in metres.
    step_dt = float(mjm.opt.timestep) * int(decimation)

    consts = {
        "key_qpos": wp.array(key_qpos_np, dtype=wp.float32, device=dev),
        "ctrl_lo": wp.array(mjm.actuator_ctrlrange[:, 0].astype(np.float32), dtype=wp.float32, device=dev),
        "ctrl_hi": wp.array(mjm.actuator_ctrlrange[:, 1].astype(np.float32), dtype=wp.float32, device=dev),
        "log": wp.zeros(5, dtype=wp.float32, device=dev),
    }
    down_body = wp.vec3(*[float(v) for v in down_body_np])
    fwd_body = wp.vec3(*[float(v) for v in fwd_body_np])

    wm = mjw.put_model(mjm)
    wm.opt.warn_overflow = False  # solver caps are deliberate (RL settings)

    buffers = []
    graphs = []
    for b in range(num_buffers):
        d = mjw.make_data(mjm, nworld=apb, nconmax=192, njmax=384)
        st = {
            "d": d,
            "cmd": wp.zeros((apb, 2), dtype=wp.float32, device=dev),
            "prev": wp.zeros((apb, NU), dtype=wp.float32, device=dev),
            "rate": wp.zeros(apb, dtype=wp.float32, device=dev),
            "over": wp.zeros(apb, dtype=wp.float32, device=dev),
            "tick": wp.zeros(apb, dtype=wp.int32, device=dev),
            "track_sum": wp.zeros(apb, dtype=wp.float32, device=dev),
            "prog_sum": wp.zeros(apb, dtype=wp.float32, device=dev),
            "ep_ret": wp.zeros(apb, dtype=wp.float32, device=dev),
            "rng": wp.array((np.arange(apb, dtype=np.uint32) * 2654435761 + seed + b + 1) | 1,
                            dtype=wp.uint32, device=dev),
            "actions": _wrap(act_ptr + b * apb * NU * 4, (apb, NU), dev),
            "obs": _wrap(obs_ptr + b * apb * OBS * 4, (apb, OBS), dev),
            "rewards": _wrap(rew_ptr + b * apb * 4, (apb,), dev),
            "terminals": _wrap(term_ptr + b * apb * 4, (apb,), dev),
        }
        buffers.append(st)

    def launch_reset(st):
        wp.launch(k_reset, dim=apb,
                  inputs=[st["d"].qpos, st["d"].qvel, st["prev"], st["cmd"],
                          st["tick"], st["track_sum"], st["prog_sum"],
                          st["ep_ret"], st["rng"],
                          st["obs"], consts["key_qpos"], fwd_body,
                          consts["ctrl_lo"], consts["ctrl_hi"],
                          float(cmd_speed_lo), float(cmd_speed_hi),
                          float(cmd_heading), float(cmd_jitter),
                          int(cmd_frame), float(reset_noise)], device=dev)

    def launch_step(st):
        wp.launch(k_pre, dim=apb,
                  inputs=[st["actions"], st["d"].ctrl, st["prev"], st["rate"],
                          st["over"], consts["ctrl_lo"], consts["ctrl_hi"],
                          float(action_scale)], device=dev)
        for _ in range(int(decimation)):
            mjw.step(wm, st["d"])
        wp.launch(k_post, dim=apb,
                  inputs=[st["d"].qpos, st["d"].qvel, st["prev"], st["rate"],
                          st["over"], st["cmd"], st["tick"], st["track_sum"],
                          st["prog_sum"], st["ep_ret"],
                          st["rng"], st["obs"], st["rewards"], st["terminals"],
                          consts["key_qpos"], down_body, fwd_body,
                          consts["ctrl_lo"], consts["ctrl_hi"], consts["log"],
                          int(max_episode_len), float(w_track), float(w_vel),
                          float(w_alive), float(w_action_rate), float(w_ang_vel),
                          float(cmd_speed_lo), float(cmd_speed_hi),
                          float(cmd_heading), float(cmd_jitter),
                          int(cmd_frame), float(step_dt), float(reset_noise),
                          float(w_overflow), float(w_qvel),
                          float(w_flip)], device=dev)

    # Eager warmup (module loads/allocs are illegal inside capture).
    for st in buffers:
        launch_reset(st)
        launch_step(st)
        launch_reset(st)
    consts["log"].zero_()
    wp.synchronize()

    for st in buffers:
        with wp.ScopedCapture() as cap:
            launch_step(st)
        st["graph"] = cap.graph
        wp.capture_launch(cap.graph)  # instantiate graph_exec
        h = cap.graph.graph_exec
        graphs.append(int(h.value if hasattr(h, "value") else h))

    for st in buffers:  # undo the instantiation step
        launch_reset(st)
    consts["log"].zero_()
    wp.synchronize()

    state.update(consts=consts, buffers=buffers, wm=wm, mjm=mjm, apb=apb,
                 launch_reset=launch_reset, dev=dev)
    return {
        "graphs": graphs,
        "agents_per_buffer": apb,
        "log_ptr": int(consts["log"].ptr),
    }


def reset():
    for st in state["buffers"]:
        state["launch_reset"](st)
    wp.synchronize()


_viewer = {}


def render():
    """Mirror world 0 of buffer 0 into a live mujoco.viewer window.

    Called (with the GIL) once per control step by `puffer eval wujicrawl`
    via c_render. The viewed env's velocity command is pinned so the hand
    crawls consistently instead of resampling per episode; override with
    WUJICRAWL_VIEW_VX / WUJICRAWL_VIEW_VY.
    """
    import time
    import mujoco.viewer

    st = state["buffers"][0]
    if "v" not in _viewer:
        d = mujoco.MjData(state["mjm"])
        _viewer["d"] = d
        _viewer["v"] = mujoco.viewer.launch_passive(state["mjm"], d)
        # Default matches the trained command: y axis, at the top of the
        # sampled speed range. Components are in the cmd_frame the run used.
        _viewer["cmd"] = (float(os.environ.get("WUJICRAWL_VIEW_VX", "0.0")),
                          float(os.environ.get("WUJICRAWL_VIEW_VY", "-0.15")))
        _viewer["t"] = time.perf_counter()
        _viewer["dt"] = 0.008  # control step (dt 0.004 x decimation 2)

    # Pin the viewed env's command (auto-resets keep resampling it).
    cmd = st["cmd"].numpy()
    cmd[0] = _viewer["cmd"]
    st["cmd"].assign(cmd)

    # WUJICRAWL_DEBUG=1: periodic policy/behavior diagnostics across ALL envs
    # of buffer 0 (action saturation, actual velocities vs commands).
    if os.environ.get("WUJICRAWL_DEBUG"):
        _viewer["n"] = _viewer.get("n", 0) + 1
        if _viewer["n"] % 100 == 1:
            import numpy as np
            a = st["actions"].numpy()
            qv = st["d"].qvel.numpy()
            sat = float((np.abs(a) > 0.99).mean())
            print(f"[dbg] act: mean|a|={np.abs(a).mean():.2f} std={a.std():.2f} "
                  f"saturated={100*sat:.0f}%  "
                  f"vx: mean={qv[:,0].mean():+.3f} p90={np.percentile(qv[:,0],90):+.3f}  "
                  f"cmd0={cmd[0]}  |w0 vx={qv[0,0]:+.3f}")

    d = _viewer["d"]
    d.qpos[:] = st["d"].qpos.numpy()[0]
    mujoco.mj_forward(state["mjm"], d)
    if not _viewer["v"].is_running():
        os._exit(0)
    _viewer["v"].sync()

    # Real-time pacing.
    wait = _viewer["t"] + _viewer["dt"] - time.perf_counter()
    if wait > 0:
        time.sleep(wait)
    _viewer["t"] = time.perf_counter()

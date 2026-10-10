"""Timelapse videos of training, rebuilt after the fact from saved checkpoints.

Nothing is recorded during training. Every run keeps a snapshot of its policy
every 250 iterations, and this script replays those snapshots in the CPU
simulator (the same one md-eval uses) and films them from the same camera, so
every condition, seed and snapshot walks the same course and is comparable.

    uv run python scripts/make_timelapse.py --list           # what can be filmed
    uv run python scripts/make_timelapse.py learning         # C2/C3/C4 learning to walk, seed 1
    uv run python scripts/make_timelapse.py exam             # every final policy vs the held-out challenges
    uv run python scripts/make_timelapse.py seeds            # the three seeds of each condition side by side
    uv run python scripts/make_timelapse.py all              # all three
    uv run python scripts/make_timelapse.py all --quick      # small, short test versions (a few minutes)

Options: --seed N (default 1), --out DIR (default videos/), --conditions c2 c3 ...,
--fast (no shadows, about 2x faster without a GPU), --workers N (parallel filming).
Writes MP4s (H.264, 25 fps). Rendering uses the GPU when one is reachable
(EGL, or GLFW under WSLg) and falls back to software OpenGL (libosmesa6,
installed by setup/setup.sh); the choice is printed at the start.

New kinds of training appear on their own (src/microduck_pretrain/runs.py);
add a line to runs.EXPERIMENTS to give them a friendly title and caption.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path


SOFTWARE_GL = ("llvmpipe", "softpipe", "swrast", "software")


def _pick_gl_backend() -> tuple[str, bool]:
    """(backend, is_software). A GPU when this machine offers one (egl, then
    glfw under WSLg), else whichever software renderer works. Respects a
    MUJOCO_GL already set."""
    import subprocess

    test = ("import mujoco, OpenGL.GL as gl; "
            "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); "
            "r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); r.update_scene(d); r.render(); "
            "print('GL_RENDERER=' + gl.glGetString(gl.GL_RENDERER).decode(), flush=True); import os; os._exit(0)")
    preset = os.environ.get("MUJOCO_GL")
    working: list[tuple[str, bool]] = []
    for backend in ([preset] if preset else ["egl", "glfw", "osmesa"]):
        env = {k: v for k, v in os.environ.items() if k != "PYOPENGL_PLATFORM"}
        env["MUJOCO_GL"] = backend
        if backend != "glfw":
            env["PYOPENGL_PLATFORM"] = backend
        try:
            out = subprocess.run([sys.executable, "-c", test], env=env, capture_output=True, text=True, timeout=60).stdout
        except Exception:
            continue
        name = next((ln.split("=", 1)[1] for ln in out.splitlines() if ln.startswith("GL_RENDERER=")), None)
        if name is None:
            continue
        software = backend == "osmesa" or any(t in name.lower() for t in SOFTWARE_GL)
        if not software:
            return backend, False
        working.append((backend, True))
    return working[0] if working else (preset or "osmesa", True)


if __name__ == "__main__" and not os.environ.get("MD_TIMELAPSE_WORKER"):
    _gl, _soft = _pick_gl_backend()
    os.environ["MUJOCO_GL"] = _gl
    os.environ["MD_TIMELAPSE_SOFTWARE_GL"] = "1" if _soft else ""
    os.environ["MD_TIMELAPSE_WORKER"] = "1"  # children inherit the choice
if os.environ.get("MUJOCO_GL") and os.environ["MUJOCO_GL"] != "glfw":
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from microduck_pretrain import evaluate as ev  # noqa: E402
from microduck_pretrain import runs as R  # noqa: E402

FPS = 25
STEPS_PER_FRAME = int(round(1.0 / FPS / ev.CONTROL_DT))  # 50 Hz control -> 2 steps per frame
DAYS_PER_ITERATION = 4096 * 24 * ev.CONTROL_DT / 86400  # simulated practice per PPO iteration

# Colours (RGB): a dark broadcast look so the blue MuJoCo floor reads well.
BG = (17, 21, 28)
PANEL_BG = (24, 29, 38)
INK = (236, 239, 244)
QUIET = (150, 160, 175)
ACCENT = (255, 140, 66)   # the duck's own orange
BAD = (235, 87, 87)
GOOD = (88, 196, 132)


@dataclass
class Layout:
    panel_w: int = 640
    image_h: int = 352
    header_h: int = 56
    footer_h: int = 40
    banner_h: int = 64
    clip_s: float = 3.0       # learning video: seconds per snapshot
    exam_s: float = 6.0       # exam and seeds videos: seconds per challenge
    card_s: float = 5.0       # title cards
    fast: bool = False        # drop shadows (about 2x faster in software rendering)
    speed: int = 2            # playback speed: 2 = each video second shows 2 s of walking

    @property
    def panel_h(self) -> int:
        return self.header_h + self.image_h + self.footer_h


FULL = Layout()
QUICK = Layout(panel_w=320, image_h=184, header_h=32, footer_h=24, banner_h=32, clip_s=2.0, exam_s=4.0, card_s=1.0)


def font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=px)


# --------------------------------------------------------------------------
# Challenges (values from sites/held_out.toml; pushes come faster so they show)
# The film order mirrors the held-out site, so the video lines up with the report.
# --------------------------------------------------------------------------
WALK = [ev.Segment(vx=0.30, seconds=4.0), ev.Segment(wz=0.8, seconds=2.0)]


@dataclass
class Challenge:
    name: str
    caption: str
    scenario: ev.Scenario


def challenges() -> list[Challenge]:
    base = ev.Scenario(name="home", delay_steps=1, commands=WALK)
    return [
        Challenge("Home turf", "Flat floor, the conditions every duck trained on", base),
        Challenge("Slippery floor", "Polished floor: foot friction 0.4", replace(base, name="slippery", foot_friction=0.4)),
        Challenge("Shoved", "A 0.5 m/s shove every 2 to 3 seconds",
                  replace(base, name="shoved", push_speed_mps=0.5, push_interval_s=(2.0, 3.0))),
        Challenge("Heavy backpack", "150 g strapped on, about 20% of its weight", replace(base, name="payload", payload_kg=0.15)),
        Challenge("Bumpy floor", "Random bumps up to 10 mm", replace(base, name="rough", rough_height_mm=10.0)),
        Challenge("Worn gears", "Every joint has a little play in it", replace(base, name="worn", scene="backlash")),
        Challenge("Weak battery", "Battery nearly flat: 6.3 V, sagging under load",
                  replace(base, name="battery", vin=6.3, vin_drop_gain=0.25)),
        Challenge("Laggy connection", "Commands reach the motors 40 ms late", replace(base, name="laggy", delay_steps=2)),
        Challenge("Everything at once", "Bumps, worn gears, 100 g, slick floor, weak battery",
                  replace(base, name="combined", rough_height_mm=6.0, scene="backlash", payload_kg=0.10,
                          foot_friction=0.6, vin=6.8)),
    ]


# --------------------------------------------------------------------------
# One panel: a policy walking a scenario, filmed
# --------------------------------------------------------------------------
@contextlib.contextmanager
def scene_extras(sc: ev.Scenario):
    """Things added to the scene for viewers only (no effect on physics):
    the payload as an orange block on the trunk (the evaluator already adds
    its mass), and the target marker: a translucent orange disc with a
    pointer, moved each step to where the commands say the duck should be."""
    orig = mujoco.MjSpec.from_file

    def patched(path, *a, **k):
        spec = orig(path, *a, **k)
        if sc.payload_kg > 0:
            g = spec.body("trunk_base").add_geom()
            g.name = "md_payload_visual"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [0.025, 0.03, 0.012 + 0.04 * sc.payload_kg]
            g.pos = [-0.01, 0.0, 0.045]
            g.rgba = [1.0, 0.55, 0.26, 1.0]
            g.contype = g.conaffinity = 0
            g.mass = 1e-6
        ghost = spec.worldbody.add_body()
        ghost.name = "md_target"
        ghost.mocap = True
        z = sc.rough_height_mm / 1000.0 + 0.002
        for name, typ, size, pos in (("md_target_disc", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.07, 0.0015, 0], [0, 0, z]),
                                     ("md_target_arrow", mujoco.mjtGeom.mjGEOM_BOX, [0.04, 0.01, 0.0016], [0.075, 0, z])):
            g = ghost.add_geom()
            g.name, g.type, g.size, g.pos = name, typ, size, pos
            g.rgba = [1.0, 0.55, 0.26, 0.55]
            g.contype = g.conaffinity = 0
        return spec

    mujoco.MjSpec.from_file = patched
    try:
        yield
    finally:
        mujoco.MjSpec.from_file = orig


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class KeptUp:
    """How much of the commanded motion the duck actually did, as a running
    share: for each command segment, distance covered along the commanded
    direction (or angle turned) over what was asked, weighted by time. The
    first 0.5 s of a segment is not judged (everyone needs a moment). A fall
    scores the interrupted segment as zero."""

    def __init__(self):
        self.done_w = self.done_r = 0.0
        self.seg = None

    def start(self, seg: ev.Segment, x: float, y: float, yaw: float):
        self.close()
        self.seg, self.t, self.p0, self.yaw0, self.turned, self.prev = seg, 0.0, (x, y), yaw, 0.0, yaw

    def step(self, dt: float, x: float, y: float, yaw: float):
        self.t += dt
        self.turned += _wrap(yaw - self.prev)
        self.prev = yaw
        self.pos = (x, y)

    def _ratio(self) -> float | None:
        s = self.seg
        if s is None or self.t < 0.5:
            return None
        v = math.hypot(s.vx, s.vy)
        if v > 1e-6:
            c, sn = math.cos(self.yaw0), math.sin(self.yaw0)
            ux, uy = (c * s.vx - sn * s.vy) / v, (sn * s.vx + c * s.vy) / v
            along = (self.pos[0] - self.p0[0]) * ux + (self.pos[1] - self.p0[1]) * uy
            r = along / (v * self.t)
        elif abs(s.wz) > 1e-6:
            r = self.turned / (s.wz * self.t)
        else:
            return None
        return min(max(r, 0.0), 1.25)

    def close(self):
        r = self._ratio()
        if r is not None:
            self.done_r += r * self.t
            self.done_w += self.t
        self.seg = None

    def fail(self):
        """A fall: the segment counts as nothing done (at least 0.5 s of it)."""
        if self.seg is not None and (self.seg.vx or self.seg.vy or self.seg.wz):
            self.done_w += max(self.t, 0.5)
        self.seg = None

    def value(self) -> float | None:
        r = self._ratio()
        w = self.done_w + (self.t if r is not None else 0.0)
        return None if w == 0 else (self.done_r + (r * self.t if r is not None else 0.0)) / w


def film(policy: Path, sc: ev.Scenario, seconds: float, lay: Layout, seed: int = 0):
    """Run `policy` through `sc` and film `seconds` of video at `lay.speed`x.
    Returns frames, the running fall count and the running kept-up share."""
    rng = np.random.default_rng(seed)
    with scene_extras(sc):
        model, data, ctrl = ev.build_sim(sc, rng)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, lay.panel_w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, lay.image_h)
    if sc.rough_height_mm > 0:  # the bumpy floor lies on the flat one: hide the flat one (looks only)
        plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        bumps = model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
        if plane.any():
            model.geom_matid[bumps] = model.geom_matid[plane][0]  # same checkered floor, now bumpy
        model.geom_rgba[plane, 3] = 0.0
    software = bool(os.environ.get("MD_TIMELAPSE_SOFTWARE_GL", "1"))
    if software:
        model.vis.quality.offsamples = 0  # anti-aliasing is the slowest part in software
    policy_rt = ev.make_policy(model, data, ctrl, policy, sc.delay_steps)
    renderer = mujoco.Renderer(model, lay.image_h, lay.panel_w)
    if software:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # keep shadows: they show footfalls
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(not lay.fast)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "md_target")])

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[qadr: qadr + 3] = [0.0, 0.0, ev.SPAWN_HEIGHT_M]
        data.qpos[qadr + 3: qadr + 7] = [1, 0, 0, 0]
        for i, idx in enumerate(policy_rt.joint_qpos_indices):
            data.qpos[idx] = policy_rt.default_pose[i]
        ctrl.reset(data.qpos)
        policy_rt.last_action[:] = 0.0
        policy_rt.set_position_targets(policy_rt.default_pose)
        mujoco.mj_forward(model, data)

    def pose():
        return float(data.qpos[qadr]), float(data.qpos[qadr + 1]), _yaw(data.qpos[qadr + 3: qadr + 7])

    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.0, 120.0, -18.0
    reset()
    look = data.qpos[qadr: qadr + 3].copy()
    look[2] = 0.11
    cycle = sum(s.seconds for s in sc.commands)
    frames, falls_seen, kept_seen, falls = [], [], [], 0
    next_push = rng.uniform(*sc.push_interval_s) if sc.push_speed_mps > 0 else math.inf
    last = None
    kept = KeptUp()
    gx = gy = gyaw = 0.0
    n_frames = int(round(seconds * FPS))
    t = 0.0
    for _ in range(n_frames):
        for _ in range(STEPS_PER_FRAME * lay.speed):
            tt = t % cycle
            seg = sc.commands[-1]
            for s in sc.commands:
                if tt < s.seconds:
                    seg = s
                    break
                tt -= s.seconds
            if seg is not last:
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    policy_rt.set_vel_cmd(seg.vx, seg.vy, seg.wz)
                last = seg
                gx, gy, gyaw = pose()  # the target restarts from the duck at every new command
                kept.start(seg, gx, gy, gyaw)
            if t >= next_push:
                ang = rng.uniform(0, 2 * math.pi)
                data.qvel[vadr] = sc.push_speed_mps * math.cos(ang)
                data.qvel[vadr + 1] = sc.push_speed_mps * math.sin(ang)
                next_push = t + rng.uniform(*sc.push_interval_s)
            policy_rt.apply_action(policy_rt.infer())
            for _ in range(ev.DECIMATION):
                ctrl.update()
                mujoco.mj_step(model, data)
            t += ev.CONTROL_DT
            gyaw += seg.wz * ev.CONTROL_DT
            c, sn = math.cos(gyaw), math.sin(gyaw)
            gx += (c * seg.vx - sn * seg.vy) * ev.CONTROL_DT
            gy += (sn * seg.vx + c * seg.vy) * ev.CONTROL_DT
            kept.step(ev.CONTROL_DT, *pose())
            gz = float(policy_rt.get_projected_gravity()[2])
            if data.qpos[qadr + 2] < ev.FALL_HEIGHT_M or gz > ev.FALL_TILT_GZ:
                falls += 1
                kept.fail()
                reset()
                last = None
        data.mocap_pos[mocap] = [gx, gy, 0.0]
        data.mocap_quat[mocap] = [math.cos(gyaw / 2), 0.0, 0.0, math.sin(gyaw / 2)]
        trunk = data.qpos[qadr: qadr + 3]
        aim = np.array([gx - trunk[0], gy - trunk[1]])
        aim *= 0.5 * min(1.0, 0.7 / max(1e-6, float(np.linalg.norm(aim))))  # frame duck and target together
        look[:2] += 0.15 * (trunk[:2] + aim - look[:2])
        cam.lookat[:] = look
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
        falls_seen.append(falls)
        kept_seen.append(kept.value())
    renderer.close()
    return frames, falls_seen, kept_seen


def _film_job(job):
    frames, falls, kept = film(*job)
    return np.stack(frames), falls, kept


_POOL = None


def film_many(jobs: list) -> list[tuple[list[np.ndarray], list[int], list[float | None]]]:
    """Film several panels at once, one process per panel (up to the CPU count)."""
    global _POOL
    if len(jobs) == 1 or WORKERS <= 1:
        return [film(*j) for j in jobs]
    if _POOL is None:
        import concurrent.futures as cf
        import multiprocessing as mp

        _POOL = cf.ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"))
    return [(list(fr), fl, kp) for fr, fl, kp in _POOL.map(_film_job, jobs)]


WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
AMBER = (240, 190, 80)


def status(kept: float | None, falls: int) -> list[tuple[str, tuple]]:
    """Footer scoreboard: share of the commanded walking done, and falls."""
    if kept is None:
        k = ("kept up  -", QUIET)
    else:
        k = (f"kept up {100 * kept:.0f}%", INK)  # no traffic light: compare the panels side by side
    return [k, (f"falls {falls}", BAD if falls else GOOD)]


def panel_image(img: np.ndarray, title: str, caption: str, footer_left: str, footer_right,
                lay: Layout, right_color=INK) -> Image.Image:
    """footer_right: a string (drawn in right_color) or [(text, colour), ...]."""
    p = Image.new("RGB", (lay.panel_w, lay.panel_h), PANEL_BG)
    p.paste(Image.fromarray(img), (0, lay.header_h))
    d = ImageDraw.Draw(p)
    s = lay.header_h / 56
    d.text((int(14 * s), int(6 * s)), title, font=font(int(22 * s)), fill=INK)
    d.text((int(14 * s), int(32 * s)), caption, font=font(int(15 * s)), fill=QUIET)
    fy = lay.header_h + lay.image_h + int(9 * s)
    d.text((int(14 * s), fy), footer_left, font=font(int(16 * s)), fill=QUIET)
    parts = [(footer_right, right_color)] if isinstance(footer_right, str) else (footer_right or [])
    x = lay.panel_w - int(14 * s)
    for text, colour in reversed(parts):
        w = d.textlength(text, font=font(int(16 * s)))
        x -= w
        d.text((x, fy), text, font=font(int(16 * s)), fill=colour)
        x -= int(22 * s)
    return p


def grid(panels: list[Image.Image], cols: int, banner: str, banner_right: str, lay: Layout) -> np.ndarray:
    rows = math.ceil(len(panels) / cols)
    W, H = cols * lay.panel_w, lay.banner_h + rows * lay.panel_h
    frame = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(frame)
    s = lay.banner_h / 64
    d.text((int(18 * s), int(16 * s)), banner, font=font(int(28 * s)), fill=INK)
    if banner_right:
        w = d.textlength(banner_right, font=font(int(24 * s)))
        d.text((W - w - int(18 * s), int(19 * s)), banner_right, font=font(int(24 * s)), fill=ACCENT)
    for i, p in enumerate(panels):
        frame.paste(p, ((i % cols) * lay.panel_w, lay.banner_h + (i // cols) * lay.panel_h))
    return np.asarray(frame)


def card(size: tuple[int, int], title: str, lines: list[str], lay: Layout) -> np.ndarray:
    W, H = size
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    s = lay.banner_h / 64
    y = int(H * 0.28)
    d.text((int(80 * s), y), title, font=font(int(52 * s)), fill=INK)
    y += int(90 * s)
    for line in lines:
        d.text((int(80 * s), y), line, font=font(int(26 * s)), fill=QUIET)
        y += int(42 * s)
    return np.asarray(img)


class Writer:
    def __init__(self, path: Path):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8,
                                    pixelformat="yuv420p", macro_block_size=16)
        self.n = 0

    def add(self, frame: np.ndarray, repeat: int = 1):
        for _ in range(repeat):
            self.w.append_data(frame)
            self.n += 1

    def close(self):
        self.w.close()
        print(f"Wrote {self.path.relative_to(REPO) if self.path.is_relative_to(REPO) else self.path}  "
              f"({self.n / FPS:.0f} s)")


def practice(it: int) -> str:
    days = it * DAYS_PER_ITERATION
    return f"{days:.0f} days of practice" if days >= 1.5 else f"{days * 24:.0f} hours of practice"


# --------------------------------------------------------------------------
# Videos
# --------------------------------------------------------------------------
def video_learning(found, seed, keys, lay, out: Path):
    chosen = [r for k in keys if (r := R.find(found, k, seed))]
    if not chosen:
        print(f"learning: no runs for {keys} seed {seed}; skipped")
        return
    its = sorted({i for r in chosen for i in r.iterations})
    if lay.panel_w == QUICK.panel_w:
        its = its[:: max(1, len(its) // 3)][:3] or its
    cols = min(3, len(chosen))
    base = challenges()[0].scenario
    w = Writer(out / f"learning_seed{seed}.mp4")
    size = (cols * lay.panel_w, lay.banner_h + math.ceil(len(chosen) / cols) * lay.panel_h)
    w.add(card(size, "How a robot learns to walk", [
        "Each duck starts knowing nothing and practises in simulation, 4096 copies at once.",
        "Every few seconds of this video jumps ahead in training to the next saved snapshot.",
        "Same floor, same camera, same command for every duck: walk forward, then turn.",
        "The orange disc is where the duck was told to be. 'Kept up' is how much of that walking it actually did.",
    ], lay), int(lay.card_s * FPS))
    for it in its:
        t0 = time.perf_counter()
        snaps = [r.nearest(it - r.experiment.start_iteration) for r in chosen]
        filmed = film_many([(R.snapshot_onnx(r, sn), base, lay.clip_s, lay, seed) for r, sn in zip(chosen, snaps)])
        clips = [(r, sn, fr, fl, kp) for r, sn, (fr, fl, kp) in zip(chosen, snaps, filmed)]
        for f in range(len(clips[0][2])):
            panels = [panel_image(fr[f], r.experiment.title, r.experiment.caption,
                                  f"snapshot {snap + r.experiment.start_iteration:,}",
                                  status(kp[f], fl[f]), lay)
                      for r, snap, fr, fl, kp in clips]
            w.add(grid(panels, cols, f"Training iteration {it:,}", f"{practice(it)}  ·  {lay.speed}x speed", lay))
        print(f"  learning: iteration {it:>5} filmed ({time.perf_counter() - t0:.0f} s)", flush=True)
    w.close()


def final_lineup(found, seed, keys):
    lineup = []
    if R.VENDOR_ONNX.exists():  # always shown: the shipped policy is the yardstick
        lineup.append((R.VENDOR, R.VENDOR_ONNX))
    for r in found:
        if r.seed != seed or r.experiment.key == "steadycam" or (keys and r.experiment.key not in keys):
            continue
        lineup.append((r.experiment, R.snapshot_onnx(r, r.last)))
    return lineup


def video_exam(found, seed, keys, lay, out: Path):
    lineup = final_lineup(found, seed, keys)
    if not lineup:
        print("exam: nothing to film; skipped")
        return
    cols = 3 if len(lineup) > 4 else min(2, len(lineup)) if len(lineup) == 4 else len(lineup)
    size = (cols * lay.panel_w, lay.banner_h + math.ceil(len(lineup) / cols) * lay.panel_h)
    w = Writer(out / f"exam_seed{seed}.mp4")
    w.add(card(size, "The final exam", [
        "Every finished policy faces the same surprises it never trained for.",
        "A fall resets the duck and adds to its count. Fewer falls and steady walking = ready on delivery.",
        "The orange disc is where the duck was told to be. 'Kept up' is how much of that walking it actually did.",
    ], lay), int(lay.card_s * FPS))
    chs = [challenges()[0], challenges()[-1]] if lay.panel_w == QUICK.panel_w else challenges()
    for ch in chs:
        t0 = time.perf_counter()
        filmed = film_many([(path, ch.scenario, lay.exam_s, lay, seed) for _, path in lineup])
        clips = [(exp, fr, fl, kp) for (exp, _), (fr, fl, kp) in zip(lineup, filmed)]
        for f in range(len(clips[0][1])):
            panels = [panel_image(fr[f], exp.title, exp.caption, ch.name.lower(),
                                  status(kp[f], fl[f]), lay) for exp, fr, fl, kp in clips]
            w.add(grid(panels, cols, ch.name, f"{ch.caption}  ·  {lay.speed}x speed", lay))
        print(f"  exam: {ch.name} filmed ({time.perf_counter() - t0:.0f} s)", flush=True)
    w.close()


def video_seeds(found, keys, lay, out: Path):
    by_exp: dict[str, list] = {}
    for r in found:
        if r.seed is None or r.experiment.key == "steadycam" or (keys and r.experiment.key not in keys):
            continue
        by_exp.setdefault(r.experiment.folder, []).append(r)
    if not by_exp:
        print("seeds: nothing to film; skipped")
        return
    ch = challenges()[-1]
    lay_cols = 3
    w = None
    for folder, rs in by_exp.items():
        rs = sorted(rs, key=lambda r: r.seed)[:3]
        if w is None:
            size = (lay_cols * lay.panel_w, lay.banner_h + lay.panel_h)
            w = Writer(out / "seeds.mp4")
            w.add(card(size, "Same recipe, three tries", [
                "Training has luck in it. Each recipe was trained three times from different random starts.",
                f"Here every try faces the hardest challenge: {ch.caption.lower()}.",
                "The orange disc is where the duck was told to be. 'Kept up' is how much of that walking it actually did.",
            ], lay), int(lay.card_s * FPS))
        t0 = time.perf_counter()
        filmed = film_many([(R.snapshot_onnx(r, r.last), ch.scenario, lay.exam_s, lay, 1) for r in rs])
        clips = [(r, fr, fl, kp) for r, (fr, fl, kp) in zip(rs, filmed)]
        exp = rs[0].experiment
        for f in range(len(clips[0][1])):
            panels = [panel_image(fr[f], f"{exp.title}, try {r.seed}", exp.caption, ch.name.lower(),
                                  status(kp[f], fl[f]), lay) for r, fr, fl, kp in clips]
            while len(panels) < lay_cols:
                panels.append(Image.new("RGB", (lay.panel_w, lay.panel_h), PANEL_BG))
            w.add(grid(panels, lay_cols, exp.title, f"{exp.caption}  ·  {lay.speed}x speed", lay))
        print(f"  seeds: {exp.title} filmed ({time.perf_counter() - t0:.0f} s)", flush=True)
    w.close()


def main(argv: list[str] | None = None) -> int:
    global WORKERS
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video", nargs="?", choices=["learning", "exam", "seeds", "all"], default="all")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--conditions", nargs="+", help="condition keys to include, e.g. c2 c3 c4 (default: all found)")
    p.add_argument("--quick", action="store_true", help="small, short versions to check everything works")
    p.add_argument("--out", type=Path, default=REPO / "videos")
    p.add_argument("--list", action="store_true", help="list the runs that can be filmed and exit")
    p.add_argument("--fast", action="store_true", help="no shadows: about 2x faster with software rendering")
    p.add_argument("--workers", type=int, help=f"parallel filming processes (default {WORKERS})")
    args = p.parse_args(argv)
    if args.workers:
        WORKERS = max(1, args.workers)

    if args.list:
        R.main()
        return 0
    found = R.discover()
    lay = replace(QUICK if args.quick else FULL, fast=args.fast)
    kind = "software" if os.environ.get("MD_TIMELAPSE_SOFTWARE_GL", "1") else "GPU"
    print(f"Rendering with {os.environ.get('MUJOCO_GL')} ({kind}), {WORKERS} parallel workers", flush=True)
    out = args.out / "quick" if args.quick else args.out
    keys = [k.lower() for k in args.conditions] if args.conditions else None
    t0 = time.perf_counter()
    if args.video in ("learning", "all"):
        video_learning(found, args.seed, keys or ["c2", "c3", "c4"], lay, out)
    if args.video in ("exam", "all"):
        video_exam(found, args.seed, keys, lay, out)
    if args.video in ("seeds", "all"):
        video_seeds(found, keys, lay, out)
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")
    if _POOL is not None:
        _POOL.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

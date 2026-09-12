"""Design sweep for v3.1: find a band/paddle where a memoryless oracle fails but the full oracle does not."""
import numpy as np, itertools, sys
from worldsim.bouncing_box import BouncingBox, BoxConfig, EVENT_PADDLE
from wm.controller import OracleController, WaitAndSeeOracleController, StayController
from wm.eval_controller import floor_visit_stats, contact_runs

def run(cfg, ctrl, episodes=60, steps=200, seed_base=5000):
    inter, visits = 0.0, 0.0
    for i in range(episodes):
        env = BouncingBox(cfg); env.reset(seed=seed_base + i)
        st = [env.state()]; hits = []
        ctrl.reset(1, seed=seed_base + i) if hasattr(ctrl, "reset") else None
        for t in range(steps):
            a = int(np.asarray(ctrl.act(np.zeros((1, 16)), np.zeros((1, 256)), state=st[-1][None]))[0])
            _, s, ev = env.step(a); st.append(s); hits.append(float((ev & EVENT_PADDLE) != 0))
        states = np.stack(st)[None]; fv = floor_visit_stats(states, paddle_h=cfg.paddle_h)
        key = [k for k in fv if "visit" in k][0]
        visits += float(np.sum(fv[key])); inter += float(contact_runs(np.array(hits)[None])[0])
    return inter / max(visits, 1), visits / episodes

configs = []
for lo, hi in [(0.13, 0.63), (0.16, 0.64), (0.13, 0.55)]:
    for pw in [0.16, 0.12]:
        configs.append((lo, hi, pw))
print(f"{'band':>12} {'paddle_w':>8} {'oracle':>8} {'wait&see':>9} {'stay':>6} {'visits/ep':>9}")
for lo, hi, pw in configs:
    cfg = BoxConfig(res=64, ball_radius=0.08, occluder=True, occluder_y=(lo, hi), paddle_w=pw)
    o, v = run(cfg, OracleController()); w, _ = run(cfg, WaitAndSeeOracleController()); st, _ = run(cfg, StayController())
    print(f"({lo:.2f},{hi:.2f}) {pw:8.2f} {o:8.2f} {w:9.2f} {st:6.2f} {v:9.2f}", flush=True)

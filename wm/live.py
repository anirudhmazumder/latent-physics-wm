"""Play the real game while watching the world model think, in real time.

    python -m wm.live                     # arrow keys; needs pygame
    python -m wm.live --autopilot         # let the dream-trained controller play
    python -m wm.live --record out.gif --steps 300 --autopilot   # headless, no pygame
    python -m wm.live --v2                # the mass-from-colour world + v2 checkpoints
    python -m wm.live --v3                # the occlusion-band world + v3 checkpoints

Four panels, left to right, all showing the SAME instant:

    REAL        the actual game frame the agent is looking at
    V SEES      decode(encode(frame)) -- the VAE's reconstruction. What survives
                the 16-number bottleneck; everything downstream sees only this.
    M PREDICTED what the dynamics model said, one frame ago, this frame would
                look like (given the action you took). Re-synced to reality
                every step, so this is pure one-step prediction quality.
    M DREAMS    an open-loop dream: synced to reality once, then running on its
                own predictions, fed only your actions. Watch it drift. Press S
                to re-sync; it also auto-resyncs every --resync frames.

Under the panels: the contact probability M assigns to the transition that is
about to happen (a bar that should light up *before* the ball reaches the
paddle), the predicted dense reward, and counters.

Keys:  arrows move   A autopilot on/off   S re-sync the dream   D freeze/unfreeze dream resync
       [ / ] dream temperature   R reset   Q quit

The timing convention is the one the controller was trained with (see
``wm/dream_env.py``): the controller and the one-step predictor both see
``[z_t, h_pre]``, where ``h_pre`` is the LSTM state after consuming
``(z_{t-1}, a_{t-1})``; then ``(z_t, a_t)`` is pushed through the LSTM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from worldsim.bouncing_box import EVENT_PADDLE, LEFT, RIGHT, STAY, BouncingBox, BoxConfig
from worldsim.render import upscale

from .analyze import load_ckpt
from .controller import load_controller
from .rnn import MDNRNN, load_rnn

PANELS = ("REAL", "V SEES", "M PREDICTED", "M DREAMS")
BG = (28, 28, 34)
GRID = (60, 60, 70)


# ------------------------------------------------------------------ the model


class LiveWorldModel:
    """Holds V, M, optionally C, and the two hidden states (tracking + dream)."""

    def __init__(
        self,
        vae_path: str,
        rnn_path: str,
        ctrl_path: Optional[str],
        ctrl_which: str = "params_best_real",
        device: str = "cpu",
        temperature: float = 0.0,
    ):
        self.device = device
        self.vae, _, _ = load_ckpt(vae_path, device)
        self.rnn, self.cfg = load_rnn(rnn_path, device)
        self.ctrl = load_controller(ctrl_path, which=ctrl_which) if ctrl_path else None
        self.eye = torch.eye(self.cfg.n_actions, device=device)
        self.temperature = temperature
        self.reset()

    def reset(self) -> None:
        self.h = self.rnn.init_hidden(1, self.device)      # tracks reality
        self.h_d = None                                     # dream state
        self.z_d = None
        self.dream_age = 0
        self.predicted: Optional[np.ndarray] = None        # frame M predicted for "now"
        self.p_hit = 0.0
        self.r_dense = 0.0

    # --- V ---------------------------------------------------------------

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        mu, _ = self.vae.encode(x.to(self.device))
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> np.ndarray:
        x = self.vae.decode(z).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (x * 255 + 0.5).astype(np.uint8)

    # --- C ---------------------------------------------------------------

    def act(self, mu: torch.Tensor) -> int:
        h_pre = self.h[0][0].cpu().numpy()
        return int(self.ctrl.act(mu.cpu().numpy(), h_pre)[0])

    # --- M ---------------------------------------------------------------

    @torch.no_grad()
    def step(self, mu: torch.Tensor, action: int):
        """Push (z_t, a_t) through both the tracking LSTM and the dream.

        Returns (predicted_next_frame, dream_frame_for_next_step).
        """
        a = self.eye[action][None]
        if self.z_d is None:
            # Must happen BEFORE the tracking state advances, so the dream
            # inherits the hidden state that has not yet consumed z_t.
            self.resync(mu)

        # Dream state: fed its OWN latent. After a re-sync it equals reality;
        # from then on it only ever sees what it dreamed, plus your actions.
        parts_d, self.h_d = self.rnn.step(self.z_d, a, self.h_d)
        self.z_d = self.rnn.sample_next(parts_d, temperature=self.temperature)[:, 0]
        self.dream_age += 1

        # Tracking state: fed the REAL latent. Its output is M's belief about
        # the next frame -- the "M PREDICTED" panel one step later -- and the
        # contact / reward heads for the transition about to happen.
        parts, self.h = self.rnn.step(mu, a, self.h)
        self.p_hit = float(torch.sigmoid(parts["hit_logit"])[0, 0, 0])
        self.r_dense = float(parts["reward"][0, 0, 0])
        z_pred = self.rnn.most_likely_mean(parts)[:, 0]
        self.predicted = self.decode(z_pred)
        return self.decode(self.z_d)

    def resync(self, mu: torch.Tensor) -> None:
        """Snap the dream back onto reality: same latent, same hidden state."""
        self.z_d = mu.clone()
        self.h_d = tuple(t.clone() for t in self.h)
        self.dream_age = 0


# ------------------------------------------------------------------ the loop


class LiveGame:
    def __init__(self, wm: LiveWorldModel, seed: Optional[int], resync_every: int,
                 mass_from_color: bool = False, occluder: bool = False,
                 occluder_y=(0.28, 0.58)):
        self.wm = wm
        self.env = BouncingBox(
            BoxConfig(
                res=64,
                ball_radius=0.08,
                mass_from_color=mass_from_color,
                occluder=occluder,
                occluder_y=tuple(occluder_y),
            ),
            seed=seed,
        )
        self.resync_every = resync_every
        self.freeze_resync = False
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.frame = self.env.reset(seed)
        self.wm.reset()
        self.mu = self.wm.encode(self.frame)
        self.wm.resync(self.mu)
        self.dream_frame = self.wm.decode(self.mu)
        self.t = 0
        self.hits = 0
        self.last_action = STAY
        self.last_hit_t = -99

    def tick(self, action: int) -> None:
        """One frame: act, advance the world, advance the model."""
        if self.resync_every and not self.freeze_resync and self.wm.dream_age >= self.resync_every:
            self.wm.resync(self.mu)
        self.dream_frame = self.wm.step(self.mu, action)
        self.frame, _, ev = self.env.step(action)
        if ev & EVENT_PADDLE:
            self.hits += 1
            self.last_hit_t = self.t
        self.mu = self.wm.encode(self.frame)
        self.t += 1
        self.last_action = action

    # --- drawing ---------------------------------------------------------

    def panels(self) -> list[np.ndarray]:
        wm = self.wm
        recon = wm.decode(self.mu)
        pred = wm.predicted if wm.predicted is not None else recon
        return [self.frame, recon, pred, self.dream_frame]

    def status_lines(self, autopilot: bool) -> list[str]:
        wm = self.wm
        a = {LEFT: "left", STAY: "stay", RIGHT: "right"}[self.last_action]
        st = self.env.state()
        names = self.env.state_names
        mass = (f"   mass {st[names.index('mass')]:.2f}"
                if "mass" in names else "")
        # v3: how much of the ball is on screen right now. The number to watch
        # in the M PREDICTED / M DREAMS panels is what those do while this is 0.
        seen = (f"   seen {st[names.index('ball_visible')]:.0%}"
                if "ball_visible" in names else "")
        return [
            f"step {self.t:4d}   hits {self.hits}   action {a:5s}{mass}{seen}   "
            f"{'AUTOPILOT (' + wm.ctrl.name + ')' if autopilot else 'MANUAL'}",
            f"P(contact next) {wm.p_hit:5.2f}   predicted reward {wm.r_dense:5.2f}   "
            f"dream age {wm.dream_age:3d}  tau {wm.temperature:.1f}  "
            f"{'resync frozen' if self.freeze_resync else f'resync every {self.resync_every}'}",
        ]


def compose(panels, scale: int, p_hit: float, pad: int = 6, header: int = 22, footer: int = 54):
    """Tile the four panels into one RGB canvas; text is drawn by the caller."""
    big = [upscale(p, scale) for p in panels]
    h, w = big[0].shape[:2]
    W = len(big) * (w + pad) + pad
    H = header + h + footer
    canvas = np.full((H, W, 3), BG, np.uint8)
    for i, img in enumerate(big):
        x0 = pad + i * (w + pad)
        canvas[header : header + h, x0 : x0 + w] = img
    # Contact-probability bar under the M PREDICTED panel.
    x0 = pad + 2 * (w + pad)
    y0 = header + h + 6
    canvas[y0 : y0 + 8, x0 : x0 + w] = GRID
    fill = int(round(p_hit * w))
    if fill > 0:
        col = (90, 200, 120) if p_hit > 0.5 else (200, 170, 80)
        canvas[y0 : y0 + 8, x0 : x0 + fill] = col
    return canvas


def panel_xs(scale: int, pad: int = 6):
    w = 64 * scale
    return [pad + i * (w + pad) for i in range(len(PANELS))]


# ------------------------------------------------------------------ headless


def record(game: LiveGame, steps: int, out: str, scale: int, autopilot: bool, fps: int, seed: int):
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    # Manual mode without a keyboard: sticky random actions, like data collection.
    hold, action = 0, STAY
    frames = []
    for _ in range(steps):
        if autopilot and game.wm.ctrl is not None:
            action = game.wm.act(game.mu)
        else:
            if hold <= 0:
                action, hold = int(rng.integers(0, 3)), int(rng.geometric(1 / 8))
            hold -= 1
        game.tick(action)
        canvas = compose(game.panels(), scale, game.wm.p_hit)
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        for name, x in zip(PANELS, panel_xs(scale)):
            d.text((x + 2, 4), name, fill=(220, 220, 230))
        y = 22 + 64 * scale + 18
        for k, line in enumerate(game.status_lines(autopilot)):
            d.text((6, y + 14 * k), line, fill=(200, 200, 210))
        frames.append(img)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_p, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"wrote {len(frames)} frames to {out_p}   hits={game.hits}")


# ------------------------------------------------------------------ pygame


def play(game: LiveGame, scale: int, autopilot: bool, fps: int):
    try:
        import pygame
    except ImportError:
        raise SystemExit("interactive mode needs pygame in this env: pip install pygame\n"
                         "(or use --record out.gif for the headless version)")

    pygame.init()
    canvas = compose(game.panels(), scale, 0.0)
    screen = pygame.display.set_mode((canvas.shape[1], canvas.shape[0]))
    pygame.display.set_caption("world model, live -- arrows | A autopilot | S resync | D freeze | [ ] tau | R reset | Q quit")
    font = pygame.font.SysFont("menlo,monaco,consolas,monospace", 13)
    clock = pygame.time.Clock()
    xs = panel_xs(scale)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_r:
                    game.reset()
                elif ev.key == pygame.K_a and game.wm.ctrl is not None:
                    autopilot = not autopilot
                elif ev.key == pygame.K_s:
                    game.wm.resync(game.mu)
                elif ev.key == pygame.K_d:
                    game.freeze_resync = not game.freeze_resync
                elif ev.key == pygame.K_LEFTBRACKET:
                    game.wm.temperature = max(0.0, round(game.wm.temperature - 0.1, 1))
                elif ev.key == pygame.K_RIGHTBRACKET:
                    game.wm.temperature = min(2.0, round(game.wm.temperature + 0.1, 1))

        keys = pygame.key.get_pressed()
        if autopilot and game.wm.ctrl is not None:
            action = game.wm.act(game.mu)
        elif keys[pygame.K_LEFT]:
            action = LEFT
        elif keys[pygame.K_RIGHT]:
            action = RIGHT
        else:
            action = STAY
        game.tick(action)

        canvas = compose(game.panels(), scale, game.wm.p_hit)
        surf = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        for name, x in zip(PANELS, xs):
            screen.blit(font.render(name, True, (220, 220, 230)), (x + 2, 3))
        y = 22 + 64 * scale + 18
        for k, line in enumerate(game.status_lines(autopilot)):
            screen.blit(font.render(line, True, (200, 200, 210)), (6, y + 15 * k))
        if game.t - game.last_hit_t < 6:
            screen.blit(font.render("CONTACT", True, (90, 220, 120)), (xs[0] + 2, 22 + 64 * scale - 18))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae", default="runs/vae_b1/vae.pt")
    p.add_argument("--rnn", default="runs/rnn_v1/rnn.pt")
    p.add_argument("--ctrl", default="runs/ctrl_v1/controller.pt",
                   help="controller for autopilot; '' to disable")
    p.add_argument("--ctrl-which", default="params_best_real")
    p.add_argument("--autopilot", action="store_true", help="start with the controller driving")
    p.add_argument("--temperature", type=float, default=0.0, help="dream sampling temperature")
    p.add_argument("--resync", type=int, default=60,
                   help="auto re-sync the dream to reality every N frames (0 = never)")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--record", default=None, help="write a GIF instead of opening a window")
    p.add_argument("--steps", type=int, default=300, help="frames to record with --record")
    p.add_argument("--device", default="cpu")
    p.add_argument("--v2", action="store_true",
                   help="v2 world (mass from colour) with the v2 checkpoints: shorthand for "
                        "--mass-from-color --vae runs/vae_v2/vae.pt --rnn runs/rnn_v2/rnn.pt "
                        "--ctrl runs/ctrl_v2/controller.pt")
    p.add_argument("--mass-from-color", action="store_true", help="v2 environment")
    p.add_argument("--v3", action="store_true",
                   help="v3 world (occlusion band) with the v3 checkpoints: shorthand for "
                        "--occluder --vae runs/vae_v3/vae.pt --rnn runs/rnn_v3/rnn.pt "
                        "--ctrl runs/ctrl_v3/controller.pt. Checkpoints that do not "
                        "exist yet are skipped, so this works at every stage of v3.")
    p.add_argument("--occluder", action="store_true", help="v3 environment")
    p.add_argument("--occluder-y", type=float, nargs=2, default=(0.28, 0.58),
                   metavar=("LO", "HI"))
    a = p.parse_args()
    if a.v3:
        a.occluder = True
        # Same "only override what the user did not set" rule as --v2, so
        # `--v3 --vae something_else.pt` still does what it says.
        for attr, path in (("vae", "runs/vae_v3/vae.pt"),
                           ("rnn", "runs/rnn_v3/rnn.pt"),
                           ("ctrl", "runs/ctrl_v3/controller.pt")):
            if getattr(a, attr) == p.get_default(attr):
                setattr(a, attr, path)
    if a.v2:
        a.mass_from_color = True
        if a.vae == p.get_default("vae"):
            a.vae = "runs/vae_v2/vae.pt"
        if a.rnn == p.get_default("rnn"):
            a.rnn = "runs/rnn_v2/rnn.pt"
        if a.ctrl == p.get_default("ctrl"):
            a.ctrl = "runs/ctrl_v2/controller.pt"
    if a.ctrl and not Path(a.ctrl).exists():
        print(f"(no controller at {a.ctrl}; autopilot disabled)")
        a.ctrl = ""
    # During v3 stage one only the VAE exists. Rather than refuse to start,
    # fall back to the matching v1/v2 RNN so the REAL and V SEES panels are
    # still usable for eyeballing the encoder -- with a loud note that the two
    # right-hand panels are then meaningless.
    if not Path(a.rnn).exists():
        raise SystemExit(
            f"no dynamics model at {a.rnn}. Stage two of v3 has not been run "
            "yet; pass --rnn explicitly to borrow another one, or use "
            "`python -m worldsim.play --occluder` to just play the world."
        )

    wm = LiveWorldModel(a.vae, a.rnn, a.ctrl or None, a.ctrl_which, a.device, a.temperature)
    game = LiveGame(wm, a.seed, a.resync, mass_from_color=a.mass_from_color,
                    occluder=a.occluder, occluder_y=tuple(a.occluder_y))
    if a.record:
        record(game, a.steps, a.record, a.scale, a.autopilot, a.fps, a.seed or 0)
    else:
        play(game, a.scale, a.autopilot, a.fps)


if __name__ == "__main__":
    main()

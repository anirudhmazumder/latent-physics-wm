"""Play the real environment with the arrow keys.

    python -m worldsim.play

Worth doing once before you train anything -- it is the fastest way to notice
that e.g. the paddle barely affects the ball, or that the ball is too fast to
track at 64x64. Requires pygame (``pip install pygame``); everything else in
this package is numpy-only.

Later, the same loop with ``env`` swapped for a decoder + dynamics model is how
you play inside the model's imagination.
"""

from __future__ import annotations

import argparse

import numpy as np

from .bouncing_box import LEFT, RIGHT, STAY, BouncingBox, BoxConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=64, help="simulation resolution")
    p.add_argument("--scale", type=int, default=8, help="display upscale factor")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mass-from-color", action="store_true",
                   help="v2: random mass each reset, shown as ball colour. "
                        "Press R a few times -- a yellow ball is visibly fast "
                        "and easy to deflect, a purple one slow and stubborn.")
    a = p.parse_args()

    try:
        import pygame
    except ImportError:
        raise SystemExit("play mode needs pygame:  pip install pygame")

    env = BouncingBox(
        BoxConfig(res=a.res, mass_from_color=a.mass_from_color), seed=a.seed
    )
    side = a.res * a.scale

    pygame.init()
    screen = pygame.display.set_mode((side, side))
    title = "worldsim v2" if a.mass_from_color else "worldsim v1"
    pygame.display.set_caption(f"{title} - arrows to move, R to reset, Q to quit")
    clock = pygame.time.Clock()

    frame = env.render()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_r:
                    frame = env.reset()
                    if a.mass_from_color:
                        print(f"mass {env.mass:.3f}  speed {env.speed:.4f}/frame  "
                              f"colour {env.ball_color}", flush=True)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = LEFT
        elif keys[pygame.K_RIGHT]:
            action = RIGHT
        else:
            action = STAY

        frame, _, _ = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (side, side))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(a.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
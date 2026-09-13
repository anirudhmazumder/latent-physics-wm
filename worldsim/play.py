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

from .bouncing_box import (
    EVENT_FLIP,
    LEFT,
    RIGHT,
    STAY,
    BouncingBox,
    BoxConfig,
)


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
    p.add_argument("--occluder", action="store_true",
                   help="v3: an opaque band hides the ball across the middle "
                        "of the box. Try to catch it -- you will find you have "
                        "to commit to a side before it reappears, which is "
                        "exactly the problem the controller is given.")
    p.add_argument("--occluder-y", type=float, nargs=2, default=(0.28, 0.58),
                   metavar=("LO", "HI"))
    p.add_argument("--gravity", type=float, default=0.0,
                   help="v4: vertical acceleration magnitude (try 0.0001). Its "
                        "DIRECTION is hidden and flips every time you hit the "
                        "ball -- worth playing once, because you will find "
                        "yourself keeping count of your own contacts, which is "
                        "precisely the state the world model has to carry.")
    p.add_argument("--gravity-sign-init", choices=["random", "down", "up"],
                   default="random",
                   help="v4: fix the starting direction instead of drawing it")
    p.add_argument("--launch-min-angle", type=float, default=None,
                   help="minimum launch angle from horizontal in degrees. "
                        "Defaults to 40 when --gravity is on (below that the "
                        "ball cannot cross the box against the pull) and to "
                        "the v1 rule otherwise.")
    a = p.parse_args()

    try:
        import pygame
    except ImportError:
        raise SystemExit("play mode needs pygame:  pip install pygame")

    # The v4 launch angle defaults to 40 whenever gravity is on, because the
    # combination "gravity on, v1 launch angle" is not a variant anyone wants
    # to play -- the ball cannot climb and the game is a floor-skim.
    launch = a.launch_min_angle
    if launch is None:
        launch = 40.0 if a.gravity > 0.0 else BoxConfig().launch_min_angle_deg
    env = BouncingBox(
        BoxConfig(
            res=a.res,
            mass_from_color=a.mass_from_color,
            occluder=a.occluder,
            occluder_y=tuple(a.occluder_y),
            gravity=a.gravity,
            gravity_sign_init=a.gravity_sign_init,
            launch_min_angle_deg=launch,
        ),
        seed=a.seed,
    )
    side = a.res * a.scale

    pygame.init()
    screen = pygame.display.set_mode((side, side))
    title = "worldsim v1"
    if a.mass_from_color:
        title = "worldsim v2"
    if a.occluder:
        title = "worldsim v3" if not a.mass_from_color else "worldsim v2+v3"
    if a.gravity > 0.0:
        title = "worldsim v4"
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
                    if a.gravity > 0.0:
                        print(f"gravity now pulls "
                              f"{'DOWN' if env.gravity_sign < 0 else 'UP'}",
                              flush=True)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = LEFT
        elif keys[pygame.K_RIGHT]:
            action = RIGHT
        else:
            action = STAY

        frame, _, events = env.step(action)
        if a.gravity > 0.0 and (events & EVENT_FLIP):
            # The only announcement of the hidden variable you will ever get.
            print(f"  FLIP -> gravity now pulls "
                  f"{'DOWN' if env.gravity_sign < 0 else 'UP'}", flush=True)

        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (side, side))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(a.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
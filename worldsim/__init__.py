"""Tiny controllable environments for world-model experiments.

v1: BouncingBox -- a ball bouncing in a unit box with a paddle you can move.
v2: the same box with ``BoxConfig(mass_from_color=True)`` -- the ball's colour
    encodes its mass, and mass sets its speed and how much english the paddle
    can impart. Default BoxConfig() is still exactly v1.
v3: the same box with ``BoxConfig(occluder=True)`` -- an opaque band across the
    middle of the frame hides the ball for part of every traverse. Physics
    untouched; the state gains a ``ball_visible`` column and the event mask an
    ``EVENT_HIDDEN`` bit. The two switches compose.
Designed so that v4 (gravity switch) stays an additive change to BoxConfig /
BouncingBox rather than a rewrite.
"""

from .bouncing_box import (
    ACTIONS,
    LEFT,
    RIGHT,
    STAY,
    STATE_NAMES,
    STATE_NAMES_V2,
    STATE_NAMES_V3,
    BouncingBox,
    BoxConfig,
    EVENT_HIDDEN,
    EVENT_PADDLE,
    EVENT_WALL_X,
    EVENT_WALL_Y,
    color_to_mass,
    mass_to_color,
    mass_to_u,
    u_to_mass,
    visible_fraction,
)
from .policies import MixedPolicy, sticky_random_actions, tracking_action

__all__ = [
    "ACTIONS",
    "LEFT",
    "RIGHT",
    "STAY",
    "STATE_NAMES",
    "STATE_NAMES_V2",
    "STATE_NAMES_V3",
    "BouncingBox",
    "BoxConfig",
    "EVENT_HIDDEN",
    "EVENT_PADDLE",
    "EVENT_WALL_X",
    "EVENT_WALL_Y",
    "sticky_random_actions",
    "MixedPolicy",
    "tracking_action",
    "mass_to_color",
    "color_to_mass",
    "mass_to_u",
    "u_to_mass",
    "visible_fraction",
]
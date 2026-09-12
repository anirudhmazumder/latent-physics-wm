"""Tiny controllable environments for world-model experiments.

v1: BouncingBox -- a ball bouncing in a unit box with a paddle you can move.
v2: the same box with ``BoxConfig(mass_from_color=True)`` -- the ball's colour
    encodes its mass, and mass sets its speed and how much english the paddle
    can impart. Default BoxConfig() is still exactly v1.
Designed so that v3 (occlusion band) and v4 (gravity switch) stay additive
changes to BoxConfig / BouncingBox rather than rewrites.
"""

from .bouncing_box import (
    ACTIONS,
    LEFT,
    RIGHT,
    STAY,
    STATE_NAMES,
    STATE_NAMES_V2,
    BouncingBox,
    BoxConfig,
    EVENT_PADDLE,
    EVENT_WALL_X,
    EVENT_WALL_Y,
    color_to_mass,
    mass_to_color,
    mass_to_u,
    u_to_mass,
)
from .policies import MixedPolicy, sticky_random_actions, tracking_action

__all__ = [
    "ACTIONS",
    "LEFT",
    "RIGHT",
    "STAY",
    "STATE_NAMES",
    "STATE_NAMES_V2",
    "BouncingBox",
    "BoxConfig",
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
]
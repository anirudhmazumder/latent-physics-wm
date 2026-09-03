"""Tiny controllable environments for world-model experiments.

v1: BouncingBox -- a ball bouncing in a unit box with a paddle you can move.
Designed so that v2 (mass-from-color), v3 (occlusion band) and v4 (gravity
switch) are additive changes to BoxConfig / BouncingBox rather than rewrites.
"""

from .bouncing_box import (
    ACTIONS,
    LEFT,
    RIGHT,
    STAY,
    STATE_NAMES,
    BouncingBox,
    BoxConfig,
    EVENT_PADDLE,
    EVENT_WALL_X,
    EVENT_WALL_Y,
)
from .policies import sticky_random_actions

__all__ = [
    "ACTIONS",
    "LEFT",
    "RIGHT",
    "STAY",
    "STATE_NAMES",
    "BouncingBox",
    "BoxConfig",
    "EVENT_PADDLE",
    "EVENT_WALL_X",
    "EVENT_WALL_Y",
    "sticky_random_actions",
]

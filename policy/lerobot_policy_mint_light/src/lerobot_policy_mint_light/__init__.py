#!/usr/bin/env python

"""MINT-Light policy plugin for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("Install lerobot==0.5.1 before using MINT-Light.") from exc

from .configuration_mint_light import MINTLightConfig
from .modeling_mint_light import MINTLightPolicy
from .processor_mint_light import make_mint_light_pre_post_processors

__all__ = ["MINTLightConfig", "MINTLightPolicy", "make_mint_light_pre_post_processors"]

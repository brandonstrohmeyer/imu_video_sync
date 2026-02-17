from .registry import LOG_SOURCES, VIDEO_SOURCES, register_source, resolve_source

# Import built-in sources to register them.
from . import aim_csv  # noqa: F401
from . import gopro_gpmf  # noqa: F401
from . import racechrono_csv  # noqa: F401
from . import dji_osmo_action  # noqa: F401

__all__ = [
    "LOG_SOURCES",
    "VIDEO_SOURCES",
    "register_source",
    "resolve_source",
]

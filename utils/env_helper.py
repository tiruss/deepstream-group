#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# env_helper.py
# @author Seonho Oh
# @description Configuration helper
# @created 2024-01-29 10:26:00
#

from __future__ import annotations

import os


def get_bool(key: str, default: bool) -> bool:
    """Get environment variable as boolean type."""
    var = os.getenv(key)
    if var is None:
        return default
    return {"true": True, "false": False, "yes": True, "no": False}.get(var.strip().casefold(), default)


def get_float(key: str, default: float) -> float:
    """Get environment variable as float type."""
    var = os.getenv(key)
    if var is None:
        return default
    return float(var)


def get_int(key: str, default: int) -> int:
    """Get environment variable as int type."""
    var = os.getenv(key)
    if var is None:
        return default
    return int(var)


def get_str(key: str, default: str) -> str:
    """Get environment variable as int type."""
    var = os.getenv(key)
    if var is None:
        return default
    return var

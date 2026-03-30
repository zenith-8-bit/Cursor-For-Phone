"""
cursor_phone/logger.py
──────────────────────
Minimal coloured logger. Uses ANSI codes — works in Termux.
"""

from enum import Enum

class LogLevel(Enum):
    DEBUG   = 0
    INFO    = 1
    STEP    = 2
    SUCCESS = 3
    WARN    = 4
    ERROR   = 5

_COLORS = {
    LogLevel.DEBUG:   "\033[90m",    # dark grey
    LogLevel.INFO:    "\033[36m",    # cyan
    LogLevel.STEP:    "\033[34m",    # blue
    LogLevel.SUCCESS: "\033[32m",    # green
    LogLevel.WARN:    "\033[33m",    # yellow
    LogLevel.ERROR:   "\033[31m",    # red
}
_RESET = "\033[0m"

_GLOBAL_LEVEL = LogLevel.DEBUG

def set_level(level: LogLevel):
    global _GLOBAL_LEVEL
    _GLOBAL_LEVEL = level

def log(msg: str, level: LogLevel = LogLevel.INFO):
    if level.value < _GLOBAL_LEVEL.value:
        return
    color = _COLORS.get(level, "")
    print(f"{color}{msg}{_RESET}")

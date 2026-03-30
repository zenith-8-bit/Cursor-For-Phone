"""
cursor_phone/agent.py
─────────────────────
Main orchestrator. Reads a .yaml script, executes each step
using the screen XML as context. If a step fails it enters
failsafe mode and asks the user what to do next.
"""

import sys
import time
import argparse
from pathlib import Path

from context import ScreenContext
from executor import Executor
from llm     import LLMPlanner
from failsafe import Failsafe
from logger  import log, LogLevel


def run_script(script_path: str, dry_run: bool = False):
    import yaml

    script = yaml.safe_load(Path(script_path).read_text())
    steps  = script.get("steps", [])
    config = script.get("config", {})

    ctx      = ScreenContext()
    exe      = Executor(dry_run=dry_run)
    planner  = LLMPlanner(config.get("model", "claude-haiku-4-5-20251001"))
    failsafe = Failsafe(ctx, exe, planner)

    log(f"Running script: {script.get('name','unnamed')}  ({len(steps)} steps)", LogLevel.INFO)

    for i, step in enumerate(steps):
        intent = step.get("do")
        log(f"\n── Step {i+1}/{len(steps)}: {intent}", LogLevel.STEP)

        success = execute_step(intent, ctx, exe, planner, failsafe,
                               max_retries=config.get("max_retries", 3))
        if not success:
            log(f"Step {i+1} failed after retries. Aborting script.", LogLevel.ERROR)
            sys.exit(1)

        wait = step.get("wait", config.get("default_wait", 1.5))
        time.sleep(wait)

    log("\n✓ Script completed successfully.", LogLevel.SUCCESS)


def execute_step(intent: str, ctx: ScreenContext, exe: Executor,
                 planner: LLMPlanner, failsafe: Failsafe,
                 max_retries: int = 3) -> bool:

    for attempt in range(1, max_retries + 1):
        log(f"  Attempt {attempt}/{max_retries}", LogLevel.DEBUG)

        # 1. Capture screen state
        scene = ctx.capture()
        if scene is None:
            log("  Could not capture screen.", LogLevel.WARN)
            time.sleep(1)
            continue

        # 2. Ask LLM what action to take
        action = planner.plan(intent, scene)
        if action is None:
            log("  LLM returned no action.", LogLevel.WARN)
        else:
            log(f"  → Action: {action}", LogLevel.DEBUG)
            result = exe.execute(action, scene)
            if result.success:
                log(f"  ✓ {result.message}", LogLevel.SUCCESS)
                return True
            log(f"  ✗ {result.message}", LogLevel.WARN)

        # 3. Failsafe on last attempt
        if attempt == max_retries:
            return failsafe.handle(intent, scene)

        time.sleep(1.2)

    return False


def interactive_mode():
    """REPL — type one instruction at a time."""
    ctx      = ScreenContext()
    exe      = Executor()
    planner  = LLMPlanner()
    failsafe = Failsafe(ctx, exe, planner)

    log("Cursor-for-Phone  [interactive mode]", LogLevel.INFO)
    log("Type an instruction or 'quit' to exit.\n", LogLevel.INFO)

    while True:
        try:
            intent = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not intent or intent.lower() in ("quit", "exit", "q"):
            break
        execute_step(intent, ctx, exe, planner, failsafe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cursor-for-Phone agent")
    parser.add_argument("script", nargs="?", help="Path to .yaml script file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and plan actions without executing them")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive REPL instead of running a script")
    args = parser.parse_args()

    if args.interactive or args.script is None:
        interactive_mode()
    else:
        run_script(args.script, dry_run=args.dry_run)

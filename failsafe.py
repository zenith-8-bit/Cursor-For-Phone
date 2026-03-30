"""
cursor_phone/failsafe.py
─────────────────────────
When the agent can't complete a step after max_retries, this
module kicks in:
  1. Prints what it sees on screen right now.
  2. Tells the user what it was trying to do.
  3. Offers options: skip / retry with new instruction / abort.
"""

from context import ScreenContext, Scene
from executor import Executor
from llm      import LLMPlanner
from logger   import log, LogLevel


class Failsafe:
    def __init__(self, ctx: ScreenContext, exe: Executor, planner: LLMPlanner):
        self.ctx     = ctx
        self.exe     = exe
        self.planner = planner

    def handle(self, original_intent: str, last_scene: Scene) -> bool:
        """
        Interactive failsafe. Returns True if recovery succeeded,
        False if user chose to abort.
        """
        print()
        log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", LogLevel.WARN)
        log("⚠  FAILSAFE — could not complete step:", LogLevel.WARN)
        log(f"   Intent: {original_intent}", LogLevel.WARN)
        log("", LogLevel.WARN)
        log("Current screen state:", LogLevel.INFO)
        print(last_scene.summary())
        log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", LogLevel.WARN)
        print()

        while True:
            print("Options:")
            print("  [r] Retry this step with a new/refined instruction")
            print("  [s] Skip this step and continue")
            print("  [a] Abort the script")
            print("  [d] Describe what you see and let me re-plan")
            choice = input("Choice [r/s/a/d]: ").strip().lower()

            if choice == "s":
                log("Skipping step.", LogLevel.INFO)
                return True   # continue script

            if choice == "a":
                log("Aborting.", LogLevel.ERROR)
                return False

            if choice == "r":
                new_intent = input("New instruction: ").strip()
                if not new_intent:
                    continue
                # Re-capture screen and try once with new intent
                scene = self.ctx.capture()
                if scene is None:
                    log("Could not capture screen.", LogLevel.ERROR)
                    continue
                action = self.planner.plan(new_intent, scene)
                if action is None:
                    log("LLM returned no action.", LogLevel.WARN)
                    continue
                result = self.exe.execute(action, scene)
                log(f"Result: {result.message}",
                    LogLevel.SUCCESS if result.success else LogLevel.WARN)
                if result.success:
                    return True

            if choice == "d":
                description = input(
                    "Describe what you see (the agent will use this as context): "
                ).strip()
                if not description:
                    continue
                # Inject the user's description as extra context
                scene = self.ctx.capture()
                if scene is None:
                    scene = last_scene
                # Prepend user description to the scene summary
                augmented_summary = (
                    f"USER OBSERVATION: {description}\n\n" + scene.summary()
                )

                class AugmentedScene:
                    def summary(self_inner):
                        return augmented_summary
                    package  = scene.package
                    activity = scene.activity
                    nodes    = scene.nodes

                action = self.planner.plan(original_intent, AugmentedScene())
                if action is None:
                    log("LLM returned no action.", LogLevel.WARN)
                    continue
                result = self.exe.execute(action, scene)
                log(f"Result: {result.message}",
                    LogLevel.SUCCESS if result.success else LogLevel.WARN)
                if result.success:
                    return True

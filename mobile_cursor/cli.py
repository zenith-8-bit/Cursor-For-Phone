import argparse
import json
from .agent import Agent
from .bridge import PhoneBridge

def main():
    p = argparse.ArgumentParser(description="Mobile Cursor Qwen phone agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("state")
    sub.add_parser("xml")
    sub.add_parser("watch")

    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--visualize", action="store_true")
    run.add_argument("--max-steps", type=int, default=None)

    args = p.parse_args()
    bridge = PhoneBridge()

    if args.cmd == "health":
        print(json.dumps(bridge.health(), indent=2))
    elif args.cmd == "state":
        print(json.dumps(bridge.state().model_dump(), indent=2, ensure_ascii=False))
    elif args.cmd == "xml":
        print(bridge.xml())
    elif args.cmd == "watch":
        Agent(bridge=bridge)._display_state(bridge.state())
    elif args.cmd == "run":
        ok, message = Agent(bridge=bridge).run(
            args.task,
            dry_run=args.dry_run,
            visualize=args.visualize,
            max_steps=args.max_steps,
        )
        print(f"\nTask {'finished' if ok else 'stopped'}: {message}")
        raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()

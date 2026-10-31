"""Generate and optionally install a macOS scheduler for the guarded retraining CLI."""

import argparse
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from heart_disease.config import PROJECT_ROOT
from heart_disease.retrain_settings import load_settings


def scheduler_definition(config: Path) -> dict[str, Any]:
    """Use monthly local calendar time and inbox changes for short-lived jobs."""
    settings = load_settings(config)
    settings.incoming_dir.mkdir(parents=True, exist_ok=True)
    logs = config.resolve().parent / "ops/local"
    logs.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(str(config.resolve()).encode()).hexdigest()[:8]
    return {
        "Label": f"local.heartml.retrain.{suffix}",
        "ProgramArguments": [
            sys.executable,
            "-m",
            "heart_disease.retrain",
            "--config",
            str(config.resolve()),
            "--trigger",
            "launchd",
        ],
        "WorkingDirectory": str(config.resolve().parent),
        "StartCalendarInterval": {"Day": 1, "Hour": 3, "Minute": 0},
        "WatchPaths": [str(settings.incoming_dir)],
        "RunAtLoad": True,
        "ThrottleInterval": 60,
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "retrain.stdout.log"),
        "StandardErrorPath": str(logs / "retrain.stderr.log"),
    }


def main(argv: list[str] | None = None) -> None:
    """Generate a reviewable plist by default; --install activates the user LaunchAgent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "retraining.toml")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    definition = scheduler_definition(args.config)
    output = args.config.resolve().parent / "ops/local" / f"{definition['Label']}.plist"
    output.write_bytes(plistlib.dumps(definition))
    if args.install:
        if sys.platform != "darwin":
            raise RuntimeError("Automatic installation supports macOS launchd only")
        destination = Path.home() / "Library/LaunchAgents" / output.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, destination)
        domain = f"gui/{os.getuid()}"
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{definition['Label']}"],
            check=False,
            capture_output=True,
        )
        subprocess.run(["launchctl", "bootstrap", domain, str(destination)], check=True)
        print(f"Installed {definition['Label']}: {destination}")
    else:
        print(f"Generated {output}; use --install to activate")


if __name__ == "__main__":
    main()

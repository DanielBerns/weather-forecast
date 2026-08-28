#!/usr/bin/env python3
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path dynamically
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from forecast_system.config import is_path_in_git_repo, get_git_repo_root


def create_external_config_dir(target_dir_path):
    target_dir = Path(target_dir_path).resolve()
    repo_root = get_git_repo_root(script_dir)

    in_repo, git_root = is_path_in_git_repo(target_dir)
    if in_repo:
        print(
            f"Error: Target directory '{target_dir}' is inside the Git repository at '{git_root}'.\n"
            f"Configuration directories must be created outside the Git repository.",
            file=sys.stderr
        )
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created external configuration directory at: {target_dir}")

    # Copy reset.yaml and resume.yaml from repo root
    source_reset = (repo_root or script_dir) / "reset.yaml"
    source_resume = (repo_root or script_dir) / "resume.yaml"

    copied_files = []
    if source_reset.exists():
        dest_reset = target_dir / "reset.yaml"
        shutil.copy2(source_reset, dest_reset)
        copied_files.append(dest_reset.name)
        print(f"✓ Copied {source_reset.name} -> {dest_reset}")
    else:
        print(f"Warning: Source file '{source_reset}' not found in repo.", file=sys.stderr)

    if source_resume.exists():
        dest_resume = target_dir / "resume.yaml"
        shutil.copy2(source_resume, dest_resume)
        copied_files.append(dest_resume.name)
        print(f"✓ Copied {source_resume.name} -> {dest_resume}")
    else:
        print(f"Warning: Source file '{source_resume}' not found in repo.", file=sys.stderr)

    print("\nInitialization complete!")
    print(f"You can now run the pipeline using one of your external configuration files, e.g.:")
    print(f"  uv run forecast-pipeline -c {target_dir / 'reset.yaml'}")
    print(f"  uv run forecast-pipeline -c {target_dir / 'resume.yaml'}")



def main():
    parser = argparse.ArgumentParser(
        description="Create an external configuration directory outside the Git repository with copies of reset.yaml and resume.yaml."
    )
    parser.add_argument(
        "path_to_directory",
        type=str,
        help="Target directory path outside the Git repository."
    )

    args = parser.parse_args()
    create_external_config_dir(args.path_to_directory)


if __name__ == "__main__":
    main()

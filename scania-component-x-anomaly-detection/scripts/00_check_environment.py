from pathlib import Path

from scania_anomaly.config import ensure_directories, load_config


def main() -> None:
    config = load_config()
    ensure_directories(config)
    print("Project:", config["project"]["name"])
    print("Drive root:", config["paths"]["drive_root"])
    print("Configured dataset files:")
    for alias, filename in config["dataset"]["files"].items():
        print(f"- {alias}: {filename}")

    raw_dir = Path(config["paths"]["raw_dir"])
    print("Raw dir exists:", raw_dir.exists())


if __name__ == "__main__":
    main()

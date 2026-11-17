"""离线检查配置继承和消融的基本约束。"""
import json
from _bootstrap import PROJECT_ROOT
from src.utils.config import load_config, validate_config


def main() -> None:
    checked = []
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        if path.name == "data.yaml":
            continue
        config = load_config(path)
        if config["experiment"]["stage"] == "sft" and config["experiment"]["level"] > 0 and not config["data"]["fingerprint"]:
            checked.append({"file": path.name, "experiment": config["experiment"]["name"], "status": "requires_tokenized_data_fingerprint_before_formal_training"})
            continue
        validate_config(config)
        checked.append({"file": path.name, "experiment": config["experiment"]["name"], "status": "schema_valid_only"})
    print(json.dumps(checked, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

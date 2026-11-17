"""可由 pytest 收集；缺少 pytest 时也可用 unittest 实际运行。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.utils.artifacts import create_run_dir, file_sha256, write_json
from src.utils.config import deep_merge, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_experiment_configs_validate(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            if path.name != "data.yaml":
                with self.subTest(path=path.name):
                    config = load_config(path)
                    if config['experiment']['stage']=='sft' and config['experiment']['level']>0:
                        # 正式训练会在真实 Tokenizer 前处理后填入所选编码数据指纹。
                        config["data"]["fingerprint"] = "test-only-resolved-fingerprint"
                    validate_config(config)

    def test_single_mechanisms_keep_other_training_parameters(self):
        base = load_config(ROOT / "configs/grpo.yaml")
        for filename, field, value in (("dapo_clip_higher.yaml", "epsilon_high", 0.28), ("dapo_token_level.yaml", "loss_type", "dapo")):
            config = load_config(ROOT / "configs" / filename)
            expected = deepcopy(base["trainer"])
            expected[field] = value
            self.assertEqual(config["trainer"], expected)
            self.assertEqual(config["mechanisms"], base["mechanisms"])
        for filename, mechanism in (("dapo_dynamic_sampling.yaml", "dynamic_sampling"), ("dapo_overlong.yaml", "overlong")):
            config = load_config(ROOT / "configs" / filename)
            expected = deepcopy(base["mechanisms"])
            expected[mechanism]["enabled"] = True
            self.assertEqual(config["trainer"], base["trainer"])
            self.assertEqual(config["mechanisms"], expected)
            self.assertEqual(config["initialization"], base["initialization"])

    def test_merge_does_not_mutate_parents(self):
        base = {"a": {"b": [1, 2]}}
        merged = deep_merge(base, {"c": 2})
        merged["a"]["b"].append(3)
        self.assertEqual(base, {"a": {"b": [1, 2]}})

    def test_cycle_and_non_mapping_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            a, b = Path(directory) / "a.yaml", Path(directory) / "b.yaml"
            a.write_text('{"extends":"b.yaml"}', encoding="utf-8")
            b.write_text('{"extends":"a.yaml"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "循环"):
                load_config(a)
            a.write_text('[]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "字典"):
                load_config(a)

    def test_invalid_group_batch_rejected(self):
        config = load_config(ROOT / "configs/grpo.yaml")
        config["trainer"]["num_generations"] = 3
        with self.assertRaisesRegex(ValueError, "整除"):
            validate_config(config)

    def test_formal_requires_revisions_data_and_shared_sft(self):
        config = load_config(ROOT / "configs/grpo.yaml")
        config["model"]["revision"] = None
        config["experiment"]["level"] = 1
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_config(config)
        config["model"]["revision"] = "example-revision-for-test-only"
        with self.assertRaisesRegex(ValueError, "数据指纹"):
            validate_config(config)
        config["data"]["fingerprint"] = "example-data-for-test-only"
        config["initialization"]["sft_adapter"] = None
        config["initialization"]["sft_adapter_sha256"] = None
        with self.assertRaisesRegex(ValueError, "SFT"):
            validate_config(config)

    def test_sampling_and_update_invariants(self):
        for field, value in (("num_iterations", 1), ("use_vllm", True)):
            config = load_config(ROOT / "configs/grpo.yaml")
            config["trainer"][field] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config["trainer"]["use_vllm"] = False
        config["trainer"]["generation_kwargs"]["do_sample"] = False
        with self.assertRaisesRegex(ValueError, "采样"):
            validate_config(config)

    def test_dynamic_and_overlong_bounds(self):
        config = load_config(ROOT / "configs/dapo_dynamic_sampling.yaml")
        config["mechanisms"]["dynamic_sampling"]["max_generated_tokens_per_batch"] = 0
        with self.assertRaisesRegex(ValueError, "预算"):
            validate_config(config)
        config = load_config(ROOT / "configs/dapo_overlong.yaml")
        config["mechanisms"]["overlong"]["soft_length"] = config["trainer"]["max_completion_length"]
        with self.assertRaisesRegex(ValueError, "阈值"):
            validate_config(config)

    def test_data_budget_and_test_isolation_plan(self):
        config = load_config(ROOT / "configs/data.yaml")
        gsm = config["gsm8k"]
        self.assertEqual(sum(gsm[name] for name in ("sft_count", "rl_count", "validation_count", "reserved_count")), 7473)
        stats = config["statistics"]
        self.assertEqual(sum(stats[name] for name in ("sft_count", "rl_count", "validation_count", "test_count")), stats["target_count"])
        self.assertEqual(stats["split_unit"], "template_family")


class ArtifactTests(unittest.TestCase):
    def test_run_ids_do_not_collide_and_reject_unsafe_stage(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            self.assertNotEqual(create_run_dir(directory, "sft"), create_run_dir(directory, "sft"))
            with self.assertRaises(ValueError):
                create_run_dir(directory, "../outside")

    def test_json_roundtrip_fingerprint_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            path = Path(directory) / "test.json"
            write_json(path, {"任务": "环境检查", "measured": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["任务"], "环境检查")
            self.assertEqual(len(file_sha256(path)), 64)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

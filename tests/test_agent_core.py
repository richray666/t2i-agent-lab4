import json
import inspect
import sys
import unittest
import uuid
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


CN_SCENE = "一座雨后的江南水乡。"
CN_FOCUS = "中文场景理解"
MISSING_EN = "缺少英文提示词。"
FIELD_CHECK = "字段校验"


class AgentCoreTests(unittest.TestCase):
    def test_load_prompts_requires_expected_fields(self):
        prompt_file = self._prompt_file(
            {
                "id": "cn_scene",
                "backend_category": "bilingual_scene",
                "prompt_zh": CN_SCENE,
                "prompt_en": "A Jiangnan water town after rain.",
                "evaluation_focus": CN_FOCUS,
            }
        )

        from t2i_agent.prompts import load_prompts

        prompts = load_prompts(prompt_file)

        self.assertEqual(prompts["cn_scene"].prompt_zh, CN_SCENE)
        self.assertEqual(prompts["cn_scene"].prompt_en, "A Jiangnan water town after rain.")
        self.assertEqual(prompts["cn_scene"].evaluation_focus, CN_FOCUS)

    def test_load_prompts_reports_missing_required_field(self):
        prompt_file = self._prompt_file(
            {
                "id": "broken",
                "backend_category": "text_rendering",
                "prompt_zh": MISSING_EN,
                "evaluation_focus": FIELD_CHECK,
            }
        )

        from t2i_agent.prompts import load_prompts

        with self.assertRaisesRegex(ValueError, "prompt_en"):
            load_prompts(prompt_file)

    def test_resolve_backend_returns_model_defaults(self):
        from t2i_agent.backends import resolve_backend

        cases = [
            ("offline", None, "local/offline-demo", 4, 1.0),
            ("kolors", None, "Kwai-Kolors/Kolors-diffusers", 25, 5.0),
            ("lens", "turbo", "microsoft/Lens-Turbo", 4, 1.0),
            ("lens", "full", "microsoft/Lens", 20, 5.0),
        ]

        for backend, variant, repo_id, steps, cfg in cases:
            with self.subTest(backend=backend, variant=variant):
                spec = resolve_backend(backend, variant=variant)
                self.assertEqual(spec.repo_id, repo_id)
                self.assertEqual(spec.default_steps, steps)
                self.assertEqual(spec.default_cfg, cfg)

    def test_resolve_backend_rejects_unknown_backend(self):
        from t2i_agent.backends import resolve_backend

        with self.assertRaises(ValueError):
            resolve_backend("midjourney")

    def test_save_generation_artifacts_writes_image_and_metadata(self):
        from t2i_agent.metadata import GenerationRecord, save_generation_artifacts

        output_dir = self._tmp_dir()
        image = Image.new("RGB", (8, 8), color=(12, 34, 56))
        record = GenerationRecord(
            backend="kolors",
            repo_id="Kwai-Kolors/Kolors-diffusers",
            prompt_id="cn_scene",
            prompt=CN_SCENE,
            seed=66,
            steps=25,
            cfg=5.0,
            width=1024,
            height=1024,
            runtime_seconds=1.25,
            offload=False,
        )

        image_path, metadata_path = save_generation_artifacts(
            image=image,
            record=record,
            output_dir=output_dir,
        )

        self.assertTrue(image_path.exists())
        self.assertTrue(metadata_path.exists())
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["image_path"], str(image_path))
        self.assertEqual(payload["backend"], "kolors")
        self.assertEqual(payload["seed"], 66)
        self.assertEqual(payload["runtime_seconds"], 1.25)

    def test_save_error_record_serializes_path_options(self):
        from t2i_agent.metadata import save_error_record

        output_dir = self._tmp_dir()
        error_path = save_error_record(
            output_dir,
            backend="kolors",
            prompt_id="cn_scene",
            error="download failed",
            options={"prompt_file": Path("prompts/test_prompts.jsonl")},
        )

        payload = json.loads(error_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["options"]["prompt_file"], "prompts/test_prompts.jsonl")

    def test_kolors_tokenizer_patch_accepts_padding_side(self):
        from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer
        from t2i_agent.generator import _patch_kolors_tokenizer_padding_side

        _patch_kolors_tokenizer_padding_side()

        signature = inspect.signature(ChatGLMTokenizer._pad)
        self.assertIn("padding_side", signature.parameters)

    def test_lens_pipeline_loader_finds_cloned_repo(self):
        from t2i_agent.generator import _load_lens_pipeline_class

        pipeline_class = _load_lens_pipeline_class()

        self.assertEqual(pipeline_class.__name__, "LensPipeline")

    def test_offline_backend_generates_real_artifacts(self):
        from t2i_agent.generator import GenerateOptions, generate

        output_dir = self._tmp_dir()
        prompt_file = self._prompt_file(
            {
                "id": "cn_scene",
                "backend_category": "bilingual_scene",
                "prompt_zh": CN_SCENE,
                "prompt_en": "A Jiangnan water town after rain.",
                "evaluation_focus": CN_FOCUS,
            }
        )

        options = GenerateOptions(
            backend="offline",
            prompt_id="cn_scene",
            prompt_file=prompt_file,
            output_dir=output_dir,
            seed=66,
        )
        image_path, metadata_path = generate(options)

        self.assertTrue(image_path.exists())
        self.assertTrue(metadata_path.exists())
        with Image.open(image_path) as img:
            self.assertEqual(img.size, (768, 768))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["backend"], "offline")
        self.assertEqual(payload["repo_id"], "local/offline-demo")

    def test_offline_backend_is_deterministic(self):
        from t2i_agent.generator import GenerateOptions, generate

        prompt_file = self._prompt_file(
            {
                "id": "cn_scene",
                "backend_category": "bilingual_scene",
                "prompt_zh": CN_SCENE,
                "prompt_en": "A Jiangnan water town after rain.",
                "evaluation_focus": CN_FOCUS,
            }
        )

        def run():
            out = self._tmp_dir()
            image_path, _ = generate(
                GenerateOptions(
                    backend="offline",
                    prompt_id="cn_scene",
                    prompt_file=prompt_file,
                    output_dir=out,
                    seed=66,
                )
            )
            return image_path.read_bytes()

        self.assertEqual(run(), run())

    def _prompt_file(self, payload):
        output_dir = self._tmp_dir()
        prompt_file = output_dir / "prompts.jsonl"
        prompt_file.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return prompt_file

    def _tmp_dir(self):
        output_dir = Path(__file__).resolve().parent / ".tmp" / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


if __name__ == "__main__":
    unittest.main()

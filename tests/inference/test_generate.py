import unittest

from parameterized import parameterized_class
from transformers import GenerationConfig

from src.inference import AnyTokenizer, apply_chat_template
from src.inference.transformers import generate, load_model


MODELS = [
    ("microsoft/Phi-4-mini-instruct"),
    ("google/gemma-2-9b-it"),
    ("Qwen/Qwen2.5-0.5B-Instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3"),
    ("CohereForAI/aya-expanse-8b"),
    ("meta-llama/Llama-3.2-1B-Instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct"),
]


@parameterized_class(("model_name",), [(model,) for model in MODELS])
class BaseTransformersGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_name = cls.model_name  # type: ignore  # noqa: PGH003

        cls.model, cls.tokenizer = load_model(model_name)
        cls.model_config = GenerationConfig.from_pretrained(model_name)

    def greedy_config(self, max_len: int):
        return GenerationConfig(
            do_sample=False,
            max_new_tokens=max_len,
            max_len=max_len,
            pad_token_id=self.model_config.pad_token_id,
            eos_token_id=self.model_config.eos_token_id,
            bos_token_id=self.model_config.bos_token_id,
        )

    def test_truncation(self):
        max_len = 4
        text = "Write a long paragraph about something."

        generated = generate(
            self.model,
            self.tokenizer,
            text,
            config=self.greedy_config(max_len),
            preprocess=pre_process,
            postprocess=str.strip,
        )

        self.assertEqual(len(self.tokenizer.encode(generated)), max_len)

    def test_generation(self):
        config = self.greedy_config(1024)
        text = "The quick brown fox jumps over the lazy dog"

        generated = generate(
            self.model,
            self.tokenizer,
            text,
            config=config,
            preprocess=pre_process,
            postprocess=str.strip,
        )

        self.assertEqual(type(generated), str)
        self.assertGreater(len(generated), 0)

    def test_batching(self):
        config = self.greedy_config(1024)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Hello, how are you?",
            "Write a long paragraph about the history of the universe.",
            "What is the meaning of life?",
        ]

        batch_1_generated = generate(
            self.model,
            self.tokenizer,
            texts,
            config=config,
            batch_size=1,
            return_prompt=True,
            preprocess=pre_process,
            postprocess=str.strip,
        )

        batch_2_generated = generate(
            self.model,
            self.tokenizer,
            texts,
            config=config,
            batch_size=2,
            return_prompt=True,
            preprocess=pre_process,
            postprocess=str.strip,
        )

        batch_4_generated = generate(
            self.model,
            self.tokenizer,
            texts,
            config=config,
            batch_size=4,
            return_prompt=True,
            preprocess=pre_process,
            postprocess=str.strip,
        )

        self.assertEqual(len(batch_1_generated), len(texts))
        self.assertEqual(len(batch_2_generated), len(texts))
        self.assertEqual(len(batch_4_generated), len(texts))

        self.assertEqual(batch_1_generated, batch_2_generated)
        self.assertEqual(batch_2_generated, batch_4_generated)


def pre_process(text: list[str] | str, tokenizer: AnyTokenizer) -> list[str]:
    if isinstance(text, str):
        text = [text]
    return apply_chat_template(text, tokenizer, system_prompt=None)

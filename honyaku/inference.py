from collections.abc import Callable

from vllm import LLM, CompletionOutput, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer


def load_model(model_name: str, dtype: str = "bfloat16", ngpus: int = 1) -> LLM:
    """Load model from huggingface"""
    return LLM(
        model_name,
        dtype=dtype,
        tensor_parallel_size=ngpus,
        distributed_executor_backend="ray",
        # enforce_eager=True,
    )


def generate(
    model: LLM,
    text: str | list[str],
    params: SamplingParams,
    preprocess: Callable[[str | list[str], AnyTokenizer], list[str]] | None = None,
    postprocess: Callable[[CompletionOutput], str] | None = None,
) -> list[str]:
    if isinstance(text, str):
        text = [text]

    if preprocess:
        tokenizer = model.get_tokenizer()
        prompts = preprocess(text, tokenizer)

    requests = model.generate(prompts, sampling_params=params)

    return [postprocess(req.outputs[0]) if postprocess else req.outputs[0].text for req in requests]


# TODO: refactor system prompt
def apply_chat_template(text: str | list[str], tokenizer: AnyTokenizer, system_prompt: str | None = None) -> list[str]:
    if isinstance(text, str):
        text = [text]

    if system_prompt:
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": t},
            ]
            for t in text
        ]
    else:
        messages = [[{"role": "user", "content": t}] for t in text]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore  # noqa: PGH003

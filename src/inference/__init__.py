# if vllm is not installed, use transformers
try:
    from vllm.transformers_utils.tokenizer import AnyTokenizer as VLLMAnyTokenizer
except ImportError:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    TransfomersTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

type AnyTokenizer = VLLMAnyTokenizer | TransfomersTokenizer


def apply_chat_template(text: str | list[str], tokenizer: AnyTokenizer, system_prompt: str | None = None) -> list[str]:
    if isinstance(text, str):
        text = [text]

    if system_prompt:
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            for user_prompt in text
        ]
    else:
        messages = [[{"role": "user", "content": user_prompt}] for user_prompt in text]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore  # noqa: PGH003

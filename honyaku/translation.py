import typing

from vllm.transformers_utils.tokenizer import AnyTokenizer

from honyaku.inference import apply_chat_template


def translation_prompt(input_text: str, model_name: str, src_lang: str = "English", trg_lang: str = "Italian") -> str:
    if "X-ALMA" in model_name:
        return f"Translate this from {src_lang} to {trg_lang}:\n{src_lang}: {input_text}\n{trg_lang}:"
    elif "TowerInstruct" in model_name:
        return f"Translate the following text from {src_lang} into {trg_lang}.\n{src_lang}: {input_text}.\n{trg_lang}:"
    elif "LLaMAX3" in model_name:
        instruction = f"Translate the following sentences from {src_lang} to {trg_lang}."

        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n"
            f"### Instruction:\n{instruction}\n"
            f"### Input:\n{input_text}\n### Response:"
        )

    return f"Translate this from {src_lang} to {trg_lang}:\n{src_lang}: {input_text}\n{trg_lang}:"


def pre_process(
    model_name: str,
    src_lang: str = "English",
    trg_lang: str = "Italian",
) -> typing.Callable[[list[str] | str, AnyTokenizer], list[str]]:
    def _pre_process(text: list[str] | str, tokenizer: AnyTokenizer) -> list[str]:
        if isinstance(text, str):
            text = [text]

        prompts = [translation_prompt(t, model_name, src_lang, trg_lang) for t in text]

        if "LLaMAX3" in model_name:
            return prompts

        return apply_chat_template(prompts, tokenizer)

    return _pre_process

import os

from typing import Callable

import torch

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.utils import is_flash_attn_2_available


type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def load_model(model_name_or_path: str, dtype: torch.dtype = torch.bfloat16) -> tuple[PreTrainedModel, Tokenizer]:
    """Load model and tokenizer from Hugging Face Hub"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # TODO: refactor for more generic handling
    if "Mistral" in model_name_or_path or "Ministral" in model_name_or_path:
        tokenizer.pad_token = "<pad>"  # noqa: S105
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        model.config.pad_token_id = tokenizer.pad_token_id

    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.inference_mode()
def generate(  # noqa: PLR0913
    model: PreTrainedModel | Callable[[PreTrainedModel], PreTrainedModel],
    tokenizer: Tokenizer,
    text: str | list[str],
    config: GenerationConfig,
    preprocess: Callable[[str | list[str], Tokenizer], list[str]] | None = None,
    postprocess: Callable[[str], str] | None = None,
    return_prompt: bool = False,
    batch_size: int = 1,
) -> str | list[str]:
    """Generate text using an Hugging Face model

    This function generates text with the given model and tokenizer. The input text might be preprocessed
    using the `preprocess` function, and the output text might be postprocessed using the `postprocess` function.

    Parameters
    ----------
    model : PreTrainedModel
        An Hugging Face model
    tokenizer : Tokenizer
        An Hugging Face tokenizer
    text : str | list[str]
        The input text or list of input texts
    config : GenerationConfig
        The generation configuration for the model
    preprocess : Callable[[str  |  list[str], Tokenizer], list[str]] | None, optional
        Preprocessing function for the input text, by default None
    postprocess : Callable[[str], str] | None, optional
        Postprocessing function for the output text, by default None
    return_prompt : bool, optional
        Whether to return the prompt in the output, by default False
    batch_size : int, optional
        The batch size for processing, by default 1

    Returns
    -------
    list[str]
        The generated text or list of generated texts
    """
    if isinstance(text, str):
        text = [text]

    outputs = []

    for i in tqdm(range(0, len(text), batch_size), desc="Processing batches", unit="batch"):
        batch = text[i : i + batch_size]

        if preprocess is not None:
            batch = preprocess(batch, tokenizer)

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, generation_config=config)  # type: ignore  # noqa: PGH003

        if return_prompt:
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        outputs.extend(batch_outputs)

    if postprocess is not None:
        outputs = [postprocess(o) for o in outputs]

    return outputs if len(outputs) > 1 else outputs[0]

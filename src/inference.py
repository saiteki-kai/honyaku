import os
import typing

from typing import Callable

import torch

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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


@torch.inference_mode()
def generate(  # noqa: PLR0913
    model: PreTrainedModel | Callable[[PreTrainedModel], PreTrainedModel],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt: str | list[str],
    config: GenerationConfig,
    include_prompt: bool = False,
    skip_special_tokens: bool = True,
) -> str | list[str]:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = typing.cast(torch.Tensor, input_ids)

    generated_ids = model.generate(input_ids, generation_config=config)

    if include_prompt:
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    else:
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

    return outputs[0]

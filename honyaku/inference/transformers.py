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


type Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def load_model(model_name_or_path: str, dtype: str = "bfloat16") -> tuple[PreTrainedModel, Tokenizer]:
    """Load model and tokenizer from Hugging Face Hub"""
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer


@torch.inference_mode()
def generate(  # noqa: PLR0913
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    text: str | list[str],
    config: GenerationConfig,
    preprocess: Callable[[str | list[str], Tokenizer], list[str]] | None = None,
    postprocess: Callable[[str], str] | None = None,
    return_prompt: bool = False,
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

    Returns
    -------
    list[str]
        The generated text or list of generated texts
    """
    if isinstance(text, str):
        text = [text]

    if preprocess is not None:
        text = preprocess(text, tokenizer)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)

    generated_ids = model.generate(**input_ids, generation_config=config)

    if return_prompt:
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    else:
        input_lengths = inputs.attention_mask.sum(dim=1)

        outputs = []
        for i in range(len(generated_ids)):
            response_ids = generated_ids[i, input_lengths[i] :]
            output = tokenizer.decode(response_ids, skip_special_tokens=True)
            outputs.append(output)

    if postprocess is not None:
        outputs = [postprocess(o) for o in outputs]

    return outputs if len(outputs) > 1 else outputs[0]

"""Model loading utilities for Hugging Face pipelines."""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_text_generation_pipeline(
    model_name: str,
    use_8bit: bool = True,
    allow_cpu_offload: bool = False,
):
    """Load a Hugging Face text-generation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    quantization_config = None
    if use_8bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        if allow_cpu_offload:
            model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return pipe
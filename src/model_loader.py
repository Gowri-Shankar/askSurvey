"""Model loading utilities for Hugging Face pipelines."""

import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# transformers 4.40+ has native Phi-3 support — trust_remote_code downloads a
# custom modeling_phi3.py that has rope_scaling key bugs with newer transformers.
# Use the built-in implementation on 4.40+; fall back to remote code on older versions.
_transformers_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_NEED_TRUST_REMOTE_CODE = _transformers_version < (4, 40)


def load_text_generation_pipeline(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    allow_cpu_offload: bool = False,
):
    """Load a Hugging Face text-generation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    quantization_config = None
    if device == "cuda":
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=_NEED_TRUST_REMOTE_CODE
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": _NEED_TRUST_REMOTE_CODE,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    if device == "cuda":
        if quantization_config:
            # Force all quantized layers onto the single GPU — auto-dispatch
            # would try to split between CPU/GPU which bitsandbytes doesn't support.
            model_kwargs["device_map"] = "cuda:0"
        elif allow_cpu_offload:
            model_kwargs["device_map"] = "auto"
            model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
        else:
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    return pipe
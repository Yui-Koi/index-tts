from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class InferenceConfig:
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 1.0
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_generate_length: int = 600


@dataclass(frozen=True)
class DecoderConfig:
    sampling_rate: int = 24000
    max_text_tokens_per_sentence: int = 100
    sentences_bucket_max_size: int = 4
    silent_token: int = 52
    max_consecutive_silence: int = 30
    stop_mel_token: int = 8193


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    use_fp16: bool
    use_cuda_kernel: bool
    dtype: torch.dtype

    @classmethod
    def auto_detect(
        cls,
        device: Optional[str] = None,
        use_cuda_kernel: Optional[bool] = None,
    ) -> 'DeviceConfig':
        if device:
            use_fp16 = device != "cpu"
            return cls(
                device=device,
                use_fp16=use_fp16,
                use_cuda_kernel=bool(use_cuda_kernel) and device.startswith("cuda"),
                dtype=torch.float16 if use_fp16 else torch.float32,
            )

        if torch.cuda.is_available():
            return cls(
                device="cuda:0",
                use_fp16=True,
                use_cuda_kernel=use_cuda_kernel is None or use_cuda_kernel,
                dtype=torch.float16,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return cls(
                device="mps",
                use_fp16=False,
                use_cuda_kernel=False,
                dtype=torch.float32,
            )

        print(">> Running on CPU. This may be slow.")
        return cls(
            device="cpu",
            use_fp16=False,
            use_cuda_kernel=False,
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class TTSConfig:
    inference: InferenceConfig
    decoder: DecoderConfig
    device: DeviceConfig

@dataclass
class TTSModelContainer:
    cfg: dict
    device: torch.device
    dtype: torch.dtype
    gpt: torch.nn.Module
    bigvgan: torch.nn.Module
    tokenizer: object
    audio_processor: object


def validate_inference(cfg: InferenceConfig) -> None:
    if not (0.0 <= cfg.top_p <= 1.0):
        raise ValueError(f"top_p must be in [0,1], got {cfg.top_p}")
    if cfg.temperature <= 0:
        raise ValueError(f"temperature must be >0, got {cfg.temperature}")
    if cfg.top_k < 0:
        raise ValueError(f"top_k must be >=0, got {cfg.top_k}")
    if cfg.num_beams < 1:
        raise ValueError(f"num_beams must be >=1, got {cfg.num_beams}")
    if cfg.repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >=1, got {cfg.repetition_penalty}")

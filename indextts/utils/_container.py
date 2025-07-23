from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from transformers.integrations import is_deepspeed_available

from .config import TTSConfig, TTSModelContainer

def _load_state(model: torch.nn.Module, path: Path, map_key: str | None = None) -> None:
    state = torch.load(path, map_location="cpu")
    state = state[map_key] if map_key and map_key in state else state
    model.load_state_dict(state, strict=True)

def _load_gpt(
    cfg: TTSConfig,
    model_dir: Path,
    hydra_cfg: DictConfig,
) -> torch.nn.Module:
    from indextts.gpt.model import UnifiedVoice

    model = UnifiedVoice(**hydra_cfg.gpt)
    _load_state(model, model_dir / hydra_cfg.gpt_checkpoint, map_key="model")
    
    use_fp16 = cfg.device.use_fp16
    if use_fp16:
        model = model.half()

    use_deepspeed = use_fp16 and is_deepspeed_available()
    if not use_deepspeed:
        print(">> DeepSpeed unavailable; falling back to native fp16 inference.")
        print("   See https://www.deepspeed.ai/tutorials/advanced-install/")

    model.post_init_gpt2_config(
        use_deepspeed=use_deepspeed,
        kv_cache=True,
        half=use_fp16,
    )
    return model


def _load_bigvgan(
    cfg: TTSConfig,
    model_dir: Path,
    hydra_cfg: DictConfig,
) -> torch.nn.Module:
    from indextts.BigVGAN.models import BigVGAN as Generator
    from indextts.BigVGAN.alias_free_activation.cuda import load as cuda_load

    use_cuda = cfg.device.use_cuda_kernel
    if use_cuda:
        try:
            cuda_load.load()
        except Exception as e:
            print(
                ">> Failed to load BigVGAN CUDA kernel; falling back to torch.",
                e,
                file=sys.stderr,
            )
            use_cuda = False

    model = Generator(hydra_cfg.bigvgan, use_cuda_kernel=use_cuda)
    _load_state(model, model_dir / hydra_cfg.gpt_checkpoint, map_key="generator")
    
    model.to(cfg.device.device).eval().remove_weight_norm()
    return model


def _load_tokenizer(
    cfg: TTSConfig,
    model_dir: Path,
    hydra_cfg: DictConfig,
):
    from indextts.utils.front import TextNormalizer, TextTokenizer

    normalizer = TextNormalizer()
    normalizer.load()
    bpe_path = model_dir / hydra_cfg.dataset["bpe_model"]
    return TextTokenizer(str(bpe_path), normalizer)


def _load_audio_processor(
    cfg: TTSConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    from indextts.utils.feature_extractors import MelSpectrogramFeatures

    return MelSpectrogramFeatures(
        sample_rate=cfg.decoder.sampling_rate,
        device=device,
        dtype=dtype,
    )


class _ContainerRegistry:
    _lock = threading.Lock()
    _cache: Dict[Tuple[str, torch.dtype], TTSModelContainer] = {}

    @classmethod
    def get(
        cls,
        cfg: TTSConfig,
        model_dir: Path,
        hydra_cfg: Optional[DictConfig],
    ) -> TTSModelContainer:
        key = (cfg.device.device, cfg.device.dtype)
        if key in cls._cache:
            return cls._cache[key]

        with cls._lock:
            if key in cls._cache:
                return cls._cache[key]

            if hydra_cfg is None:
                raise RuntimeError(
                    "First call must provide hydra_cfg (DictConfig) to load models."
                )

            cls._cache[key] = cls._build(cfg, model_dir, hydra_cfg)
            return cls._cache[key]

    @staticmethod
    def _build(
        cfg: TTSConfig,
        model_dir: Path,
        hydra_cfg: DictConfig,
    ) -> TTSModelContainer:
        device = torch.device(cfg.device.device)
        dtype = cfg.device.dtype

        return TTSModelContainer(
            cfg=hydra_cfg,
            device=device,
            dtype=dtype,
            gpt=_load_gpt(cfg, model_dir, hydra_cfg),
            bigvgan=_load_bigvgan(cfg, model_dir, hydra_cfg),
            tokenizer=_load_tokenizer(cfg, model_dir, hydra_cfg),
            audio_processor=_load_audio_processor(cfg, device, dtype),
        )


def get_container(
    cfg: TTSConfig,
    model_dir: Path,
    hydra_cfg: Optional[DictConfig] = None,
) -> TTSModelContainer:
    # returns singleton container for specific device/dtype.
    return _ContainerRegistry.get(cfg, Path(model_dir), hydra_cfg)

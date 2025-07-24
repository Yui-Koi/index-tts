# indextts/inference.py
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace, asdict
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.utils.config import TTSConfig, InferenceConfig, DecoderConfig
from indextts.utils._container import get_container
from indextts.utils.front import TextNormalizer, TextTokenizer



def remove_excessive_silence(
    codes: torch.Tensor,
    silent_token: int,
    stop_token: int,
    max_consecutive: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    processed, lengths = [], []
    for code in codes:
        stop_idx = (code == stop_token).nonzero(as_tuple=False)
        code = code[: stop_idx[0].item()] if len(stop_idx) else code
        if (code == silent_token).sum() > max_consecutive:
            out, silence_count = [], 0
            for tok in code:
                if tok != silent_token:
                    out.append(tok)
                    silence_count = 0
                elif silence_count < max_consecutive:
                    out.append(tok)
                    silence_count += 1
            code = torch.tensor(out, device=codes.device, dtype=codes.dtype)
        processed.append(code)
        lengths.append(len(code))
    padded = pad_sequence(processed, batch_first=True, padding_value=stop_token)
    return padded, torch.tensor(lengths, dtype=torch.long, device=codes.device)


def create_sentence_buckets(
    sentences: List[str], bucket_size: int
) -> List[List[Tuple[int, List[str]]]]:
    indexed = [(i, s) for i, s in enumerate(sentences) if s.strip()]
    indexed.sort(key=itemgetter(1))
    buckets, current = [], []
    for item in indexed:
        if len(current) >= bucket_size or (
            current and len(item[1]) >= len(current[-1][1]) * 1.5
        ):
            buckets.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        buckets.append(current)
    return buckets


class IndexTTS:
    def __init__(
        self,
        cfg_path: str = "checkpoints/config.yaml",
        model_dir: str = "checkpoints",
        device: Optional[str] = None,
        use_cuda_kernel: Optional[bool] = None,
        tts_config: Optional[TTSConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
        device_config: Optional[DeviceConfig] = None,
    ) -> None:
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = Path(model_dir)

        base_cfg = tts_config or TTSConfig(
            inference=inference_config or InferenceConfig(),
            decoder=decoder_config or DecoderConfig(),
            device=device_config or DeviceConfig.auto_detect(device, use_cuda_kernel),
        )

        self._container = get_container(base_cfg, self.model_dir, self.cfg)
        self._base_cfg = base_cfg
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    @property
    def device(self) -> str:
        return str(self._container.device)

    @cached_property
    def stop_mel_token(self) -> int:
        return self.cfg.gpt.stop_mel_token

    def set_progress_callback(
        self, callback: Optional[Callable[[float, str], None]]
    ) -> "IndexTTS":
        self._progress_callback = callback
        return self

    def _update_progress(self, value: float, desc: str) -> None:
        if self._progress_callback:
            self._progress_callback(value, desc)

    def _clear_gpu_cache(self) -> None:
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        elif "mps" in self.device:
            torch.mps.empty_cache()

   def _resolve_cfg(
        self,
        *,
        inference: Optional[dict[str, Any]] = None,
        decoder: Optional[dict[str, Any]] = None,
    ) -> TTSConfig:
        new_inf = (
            replace(self._base_cfg.inference, **inference)
            if inference
            else self._base_cfg.inference
        )
        new_dec = (
            replace(self._base_cfg.decoder, **decoder)
            if decoder
            else self._base_cfg.decoder
        )
        return replace(self._base_cfg, inference=new_inf, decoder=new_dec)

    def infer_fast(
        self,
        audio_prompt: str,
        text: str,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[int, np.ndarray]]:
        overrides = {"num_beams": 1, "do_sample": False}
        overrides.update(kwargs.pop("inference", {}) or {})
        cfg = self._resolve_cfg(inference=overrides, decoder=kwargs.pop("decoder", None))
        return self._run_pipeline(audio_prompt, text, cfg, fast=True, output_path=output_path, **kwargs)

    def infer(
        self,
        audio_prompt: str,
        text: str,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[int, np.ndarray]]:
        cfg = self._resolve_cfg(
            inference=kwargs.pop("inference", None),
            decoder=kwargs.pop("decoder", None),
        )
        return self._run_pipeline(audio_prompt, text, cfg, fast=False, output_path=output_path, **kwargs)

    def _run_pipeline(
        self,
        audio_prompt: str,
        text: str,
        cfg: TTSConfig,
        fast: bool,
        output_path: Optional[str],
        **unused: Any,
    ) -> Union[str, Tuple[int, np.ndarray]]:
        self._update_progress(0.0, "Starting inference...")
        conditioning = self._audio_processor.get_mel_features(audio_prompt, self.device)

        sentences = self._process_text(text, cfg.decoder.max_text_tokens_per_sentence)
        buckets = create_sentence_buckets(sentences, cfg.decoder.sentences_bucket_max_size)

        all_latents: dict[int, torch.Tensor] = {}
        total_buckets = len(buckets)

        with tqdm(total=total_buckets, disable=not fast) as pbar:
            for i, bucket in enumerate(buckets):
                self._update_progress(
                    0.1 + 0.7 * (i / total_buckets), f"Bucket {i+1}/{total_buckets}"
                )
                token_ids = [
                    torch.tensor(
                        self._container.tokenizer.convert_tokens_to_ids(sent[1]),
                        dtype=torch.long,
                        device=self._container.device,
                    ).unsqueeze(0)
                    for _, sent in bucket
                ]
                padded_tokens = pad_sequence(
                    [t.squeeze(0) for t in token_ids],
                    batch_first=True,
                    padding_value=self.cfg.gpt.stop_text_token,
                )

                codes = self._generate_codes(conditioning, padded_tokens, cfg.inference)
                codes, code_lengths = remove_excessive_silence(
                    codes,
                    cfg.decoder.silent_token,
                    cfg.decoder.stop_mel_token,
                    cfg.decoder.max_consecutive_silence,
                )

                for j, (original_idx, _) in enumerate(bucket):
                    latent = self._generate_latents(
                        conditioning,
                        token_ids[j],
                        codes[j].unsqueeze(0),
                        code_lengths[j].unsqueeze(0),
                    )
                    all_latents[original_idx] = latent
                pbar.update(1)

        ordered = [all_latents[k] for k in sorted(all_latents)]
        self._update_progress(0.9, "Decoding audio...")
        wav = self._decode_audio(torch.cat(ordered, dim=2), conditioning)
        self._clear_gpu_cache()
        self._update_progress(1.0, "Done")

        return self._finalize_audio(wav, output_path, cfg)

    def _process_text(self, text: str, max_tokens: int) -> list[list[str]]:
        tokens = self._container.tokenizer.tokenize(text)
        return self._container.tokenizer.split_sentences(tokens, max_tokens)

    @torch.no_grad()
    def _generate_codes(
        self, conditioning: torch.Tensor, text_tokens: torch.Tensor, cfg_inf: InferenceConfig
    ) -> torch.Tensor:
        model = self._container.gpt
        with torch.autocast(
            self.device.split(":")[0],
            enabled=self._container.dtype is torch.float16,
            dtype=self._container.dtype,
        ):
            return model.inference_speech(
                conditioning,
                text_tokens,
                cond_mel_lengths=torch.tensor(
                    [conditioning.shape[-1]], device=self._container.device
                ),
                **asdict(cfg_inf),
            )

    @torch.no_grad()
    def _generate_latents(
        self,
        conditioning: torch.Tensor,
        text_tokens: torch.Tensor,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
    ) -> torch.Tensor:
        model = self._container.gpt
        with torch.autocast(
            self.device.split(":")[0],
            enabled=self._container.dtype is torch.float16,
            dtype=self._container.dtype,
        ):
            return model(
                conditioning,
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=self._container.device),
                codes,
                code_lengths * model.mel_length_compression,
                cond_mel_lengths=torch.tensor(
                    [conditioning.shape[-1]], device=self._container.device
                ),
                return_latent=True,
                clip_inputs=False,
            )

    @torch.no_grad()
    def _decode_audio(
        self, latents: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        vocoder = self._container.bigvgan
        with torch.autocast(
            self.device.split(":")[0],
            enabled=self._container.dtype is torch.float16,
            dtype=self._container.dtype,
        ):
            wav, _ = vocoder(latents, conditioning.transpose(1, 2))
            return torch.clamp(32767 * wav.squeeze(1), -32767.0, 32767.0)

    def _finalize_audio(
        self, wav: torch.Tensor, output_path: Optional[str], cfg: TTSConfig
    ) -> Union[str, tuple[int, np.ndarray]]:
        wav_int16 = wav.cpu().type(torch.int16)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, wav_int16, cfg.decoder.sampling_rate)
            return output_path
        return (cfg.decoder.sampling_rate, wav_int16.numpy().T)

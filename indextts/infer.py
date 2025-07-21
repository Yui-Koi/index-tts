import time
import warnings
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 1.0
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 600
    # Max tokens per sentence for splitting. Lower for faster batching, higher for closer non-fast quality.
    max_text_tokens_per_sentence: int = 100
    # Max sentences per bucket in fast inference. Larger for faster speed, but more memory.
    sentences_bucket_max_size: int = 4
    sampling_rate: int = 24000
    # Token ID for silence. Used for trimming excessive silence.
    silent_token: int = 52
    # The maximum number of consecutive silent tokens allowed before trimming.
    max_consecutive_silence: int = 30


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    use_fp16: bool
    use_cuda_kernel: bool
    dtype: torch.dtype
    
    @classmethod
    def auto_detect(cls, device: Optional[str] = None, use_cuda_kernel: Optional[bool] = None) -> 'DeviceConfig':
        if device:
            use_fp16 = device != "cpu"
            return cls(
                device=device,
                use_fp16=use_fp16,
                use_cuda_kernel=bool(use_cuda_kernel) and device.startswith("cuda"),
                dtype=torch.float16 if use_fp16 else torch.float32
            )
        
        if torch.cuda.is_available():
            return cls(
                device="cuda:0",
                use_fp16=True,
                use_cuda_kernel=use_cuda_kernel is None or use_cuda_kernel,
                dtype=torch.float16
            )
        elif hasattr(torch, "backends") and torch.backends.mps.is_available():
            return cls(
                device="mps",
                use_fp16=False, # FP16 is not beneficial on MPS
                use_cuda_kernel=False,
                dtype=torch.float32
            )
        else:
            print(">> Running on CPU. This may be slow.")
            return cls(
                device="cpu",
                use_fp16=False,
                use_cuda_kernel=False,
                dtype=torch.float32
            )

class CachedAudioProcessor:
    def __init__(self, mel_cache_size: int = 10, target_sr: int = 24000):
        self.mel_cache = lru_cache(maxsize=mel_cache_size)(self._get_mel_features)
        self.target_sr = target_sr

    def _get_mel_features(self, audio_path: str, device: str) -> torch.Tensor:
       # audio: [C, T_samples]
        audio, sr = torchaudio.load(audio_path)
        # audio: [1, T_samples]
        audio = audio.mean(dim=0, keepdim=True) 
        if sr != self.target_sr:
            audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)
        
        # mel_spec: [1, n_mels, T_mel_frames]
        mel_spec = MelSpectrogramFeatures()(audio)
        return mel_spec.to(device)

    def get_mel_features(self, audio_path: str, device: str) -> torch.Tensor:
        return self.mel_cache(audio_path, device)

    def clear_cache(self):
        self.mel_cache.cache_clear()


def remove_excessive_silence(
    codes: torch.Tensor,
    silent_token: int,
    stop_token: int,
    max_consecutive: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shrinks long sequences of silent tokens in the generated acoustic codes.

    Args:
        codes (torch.Tensor): The acoustic codes tensor of shape `[B, T]`.
        silent_token (int): The token ID for silence.
        stop_token (int): The token ID for stopping generation.
        max_consecutive (int): The maximum number of consecutive silent tokens to allow.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The processed codes, padded to the new max length `[B, T_new]`.
            - The new lengths of each sequence in the batch `[B]`.
    """
    processed, lengths = [], []
    
    for code in codes:
        # Find the first stop token to determine the actual length
        stop_indices = (code == stop_token).nonzero(as_tuple=False)
        length = stop_indices[0].item() if len(stop_indices) > 0 else len(code)
        code = code[:length]
        
        # Only process if there's a chance of excessive silence
        if (code == silent_token).sum() > max_consecutive:
            indices = []
            silence_count = 0
            for token in code:
                if token != silent_token:
                    indices.append(token)
                    silence_count = 0
                elif silence_count < max_consecutive:
                    indices.append(token)
                    silence_count += 1
            code = torch.tensor(indices, device=codes.device, dtype=codes.dtype)

        processed.append(code)
        lengths.append(len(code))

    # Pad the processed codes to be of equal length for batching
    padded = pad_sequence(processed, batch_first=True, padding_value=stop_token) if processed else codes.new_empty(0)
    return padded, torch.tensor(lengths, dtype=torch.long, device=codes.device)


def create_sentence_buckets(sentences: List[str], bucket_size: int) -> List[List[Tuple[int, List[str]]]]:
    """
    Groups sentences into buckets of similar length for efficient batch processing.
    
    Args:
        sentences (List[str]): A list of sentences to bucket.
        bucket_size (int): The maximum number of sentences per bucket.

    Returns:
        List[List[Tuple[int, List[str]]]]: A list of buckets, where each bucket
        is a list of (original_index, sentence_tokens) tuples.
    """
    # Sort sentences by length to enable efficient bucketing, keeping original index
    indexed_sentences = sorted([(i, s) for i, s in enumerate(sentences)], key=lambda x: len(x[1]))
    if len(indexed_sentences) <= bucket_size:
        return [indexed_sentences]

    buckets = []
    current_bucket = []
    for sent in indexed_sentences:
        if not sent[1]: continue # Skip empty sentences
        
        # Start a new bucket if the current one is full or the length difference is too large
        is_new_bucket = (
            not current_bucket or
            len(current_bucket) >= bucket_size or
            len(sent[1]) >= len(current_bucket[-1][1]) * 1.5 # Heuristic to prevent large padding
        )
        if is_new_bucket:
            if current_bucket: buckets.append(current_bucket)
            current_bucket = [sent]
        else:
            current_bucket.append(sent)
            
    if current_bucket: buckets.append(current_bucket)
    return buckets


class IndexTTS:
    def __init__(
        self,
        cfg_path: str = "checkpoints/config.yaml",
        model_dir: str = "checkpoints",
        device: Optional[str] = None,
        use_cuda_kernel: Optional[bool] = None,
    ):
        """
        Args:
            cfg_path (str): Path to the config.yaml file.
            model_dir (str): Path to the directory containing model checkpoints.
            device (Optional[str]): Device to use (e.g., 'cuda:0', 'cpu'). If None, it will be auto-detected.
            use_cuda_kernel (Optional[bool]): Whether to use BigVGAN's custom CUDA kernel for speed.
        """
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = Path(model_dir)
        self.device_config = DeviceConfig.auto_detect(device, use_cuda_kernel)
        
        self._models_loaded = False
        self._audio_processor = CachedAudioProcessor()
        self._progress_callback: Optional[Callable] = None
        
        # Load all models and tokenizers upon initialization
        self._load_models()

    @property
    def device(self) -> str:
        return self.device_config.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.device_config.dtype
    
    @cached_property
    def stop_mel_token(self) -> int:
        return self.cfg.gpt.stop_mel_token
    
    @cached_property
    def model_version(self) -> Optional[float]:
        return getattr(self.cfg, 'version', None)
    
    def set_progress_callback(self, callback: Optional[Callable[[float, str], None]]):
        self._progress_callback = callback
        return self
        
    def _update_progress(self, value: float, desc: str):
        if self._progress_callback: 
            self._progress_callback(value, desc=desc)
    
    def _load_models(self):
        if self._models_loaded: return
        print(">> Loading models...")
        self.gpt = self._load_gpt_model()
        self.bigvgan = self._load_vocoder()
        self.normalizer = TextNormalizer(); self.normalizer.load()
        bpe_path = self.model_dir / self.cfg.dataset["bpe_model"]
        self.tokenizer = TextTokenizer(str(bpe_path), self.normalizer)
        self._models_loaded = True
        print(f">> All models loaded successfully on {self.device}")
    
    def _load_gpt_model(self) -> UnifiedVoice:
        model = UnifiedVoice(**self.cfg.gpt)
        load_checkpoint(model, str(self.model_dir / self.cfg.gpt_checkpoint))
        model = model.to(self.device).eval()
        
        if self.device_config.use_fp16: model = model.half()
        
        use_deepspeed = False
        if self.device_config.use_fp16:
            try:
                import deepspeed
                use_deepspeed = True
            except ImportError: pass
        
        # Configure model for efficient inference with KV caching
        model.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.device_config.use_fp16)
        return model
    
    def _load_vocoder(self) -> Generator:
        use_cuda = self.device_config.use_cuda_kernel
        if use_cuda:
            try:
                # Preload the custom CUDA kernel for BigVGAN's activation function
                from indextts.BigVGAN.alias_free_activation.cuda import load
                load.load()
            except Exception: 
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", file=sys.stderr)
                use_cuda = False
        
        model = Generator(self.cfg.bigvgan, use_cuda_kernel=use_cuda)
        state = torch.load(str(self.model_dir / self.cfg.bigvgan_checkpoint), map_location="cpu")
        model.load_state_dict(state["generator"])
        model.to(self.device).eval().remove_weight_norm()
        return model
    
    def _clear_gpu_cache(self):
        if "cuda" in self.device: torch.cuda.empty_cache()
        elif "mps" in self.device: torch.mps.empty_cache()
    
    @torch.no_grad()
    def _generate_codes(self, conditioning: torch.Tensor, text_tokens: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """
        Generates discrete acoustic codes from text tokens using the GPT model (autoregressive step).
        
        Args:
            conditioning (torch.Tensor): Reference audio mel-spectrogram `[1, n_mels, T_mel]`.
            text_tokens (torch.Tensor): Input text token IDs `[B, T_text]`.
            config (GenerationConfig): The generation configuration object.

        Returns:
            torch.Tensor: Generated acoustic codes `[B, T_codes]`.
        """
        model_kwargs = {
            "do_sample": config.do_sample, "top_p": config.top_p, "top_k": config.top_k,
            "temperature": config.temperature, "length_penalty": config.length_penalty,
            "num_beams": config.num_beams, "repetition_penalty": config.repetition_penalty,
            "max_generate_length": config.max_mel_tokens,
        }
        with torch.amp.autocast(self.device.split(':')[0], enabled=self.device_config.use_fp16, dtype=self.dtype):
            return self.gpt.inference_speech(
                conditioning, text_tokens,
                cond_mel_lengths=torch.tensor([conditioning.shape[-1]], device=self.device),
                **model_kwargs
            ) 

    @torch.no_grad()
    def _generate_latents(self, conditioning: torch.Tensor, text_tokens: torch.Tensor, codes: torch.Tensor, code_lengths: torch.Tensor) -> torch.Tensor:
        """
        Generates continuous latent representations from text tokens and acoustic codes.
        
        Args:
            conditioning (torch.Tensor): Reference audio mel-spectrogram `[1, n_mels, T_mel]`.
            text_tokens (torch.Tensor): Input text token IDs `[B, T_text]`.
            codes (torch.Tensor): Generated acoustic codes `[B, T_codes]`.
            code_lengths (torch.Tensor): Length of each code sequence in the batch `[B]`.

        Returns:
            torch.Tensor: The continuous latent tensor `[B, D_latent, T_latent]`.
        """
        with torch.amp.autocast(self.device.split(':')[0], enabled=self.device_config.use_fp16, dtype=self.dtype):
            return self.gpt(
                conditioning, text_tokens, torch.tensor([text_tokens.shape[-1]], device=self.device),
                codes, code_lengths * self.gpt.mel_length_compression,
                cond_mel_lengths=torch.tensor([conditioning.shape[-1]], device=self.device),
                return_latent=True, clip_inputs=False
            )
    
    @torch.no_grad()
    def _decode_audio(self, latents: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representations into an audio waveform using the BigVGAN vocoder.

        Args:
            latents (torch.Tensor): The continuous latent tensor `[B, D_latent, T_latent]`.
            conditioning (torch.Tensor): Reference audio mel-spectrogram `[1, n_mels, T_mel]`.

        Returns:
            torch.Tensor: The decoded audio waveform `[1, T_samples]`.
        """
        with torch.amp.autocast(self.device.split(':')[0], enabled=self.device_config.use_fp16, dtype=self.dtype):
            # latents are [B, D, T], conditioning is [1, D_mel, T_mel]
            # bigvgan expects conditioning as [B, T_mel, D_mel], so we transpose
            wav, _ = self.bigvgan(latents, conditioning.transpose(1, 2))
            return torch.clamp(32767 * wav.squeeze(1), -32767.0, 32767.0)
    
    def _process_text(self, text: str, max_tokens: int) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.split_sentences(tokens, max_tokens)
    
    def _tokens_to_ids(self, tokens: List[str]) -> torch.Tensor:
        # returns tensor of shape [1, T_text]
        return torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.int32, device=self.device).unsqueeze(0)
    
    def _pad_tokens(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Pads a list of token tensors to the same length to form a batch.
        Input: List of tensors, each `[1, T]`.
        Output: A single batch tensor `[B, T_max]`.
        """
        if self.model_version and self.model_version >= 1.5:
            # Squeeze to [T] before padding
            tokens = [t.squeeze(0) for t in token_list]
            return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token)
        
        max_len = max(t.size(1) for t in token_list)
        padded_tensors = []
        for tensor in token_list:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            padded_tensors.append(tensor[:, :max_len])
        return torch.cat(padded_tensors, dim=0)

    def _finalize_audio(self, wav: torch.Tensor, output_path: Optional[str], config: GenerationConfig):
        # wav: [1, T_samples], convert to int16 on CPU
        wav_int16 = wav.cpu().type(torch.int16)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, wav_int16, config.sampling_rate)
            return output_path
        # Return format for Gradio: (sample_rate, numpy_array)
        return (config.sampling_rate, wav_int16.numpy().T)

    def infer_fast(self, audio_prompt: str, text: str, output_path: Optional[str] = None, **kwargs):
        """
        Performs fast TTS inference by batching and bucketing sentences.
        Ideal for long texts, offering significant speed improvements (2-10x+).
        
        Args:
            audio_prompt (str): Path to the reference audio file.
            text (str): The text to synthesize.
            output_path (Optional[str]): Path to save the output .wav file. If None, returns audio data.
            **kwargs: Additional generation parameters to override `GenerationConfig` defaults.

        Returns:
            Union[str, Tuple[int, np.ndarray]]: The output file path or a tuple of (sample_rate, audio_data).
        """
        self._update_progress(0, "Starting fast inference...")
        config = GenerationConfig(**kwargs)
        
        conditioning = self._audio_processor.get_mel_features(audio_prompt, self.device)
        sentences = self._process_text(text, config.max_text_tokens_per_sentence)
        buckets = create_sentence_buckets(sentences, config.sentences_bucket_max_size)

        all_latents = {} # stores latents to reorder later
        with tqdm(total=len(buckets), desc="Generating Codes & Latents") as pbar:
            for i, bucket in enumerate(buckets):
                self._update_progress(0.1 + 0.6 * (i / len(buckets)), f"Processing bucket {i+1}/{len(buckets)}")
                token_ids = [self._tokens_to_ids(sent[1]) for sent in bucket]
                padded_tokens = self._pad_tokens(token_ids)
                codes = self._generate_codes(conditioning, padded_tokens, config)
                codes, code_lengths = remove_excessive_silence(codes, config.silent_token, self.stop_mel_token, config.max_consecutive_silence)
                # Generate latents for each sentence in the batch
                for j, (original_idx, _) in enumerate(bucket):
                    latent = self._generate_latents(conditioning, token_ids[j], codes[j].unsqueeze(0), code_lengths[j].unsqueeze(0))
                    all_latents[original_idx] = latent
                pbar.update(1)

        ordered_latents = [all_latents[i] for i in sorted(all_latents.keys())]
        self._update_progress(0.8, "Decoding audio...")
        # Concatenate all latents into one big tensor [1, D, T_total] for a single vocoder pass
        wav = self._decode_audio(torch.cat(ordered_latents, dim=2), conditioning) # Note: dim=2 for [B, D, T]
        
        self._clear_gpu_cache()
        self._update_progress(1.0, "Finished!")
        return self._finalize_audio(wav, output_path, config)

    def infer(self, audio_prompt: str, text: str, output_path: Optional[str] = None, **kwargs):
        """
        Performs standard TTS inference, processing one sentence at a time.
        This is a straightforward and reliable method, but slower for long texts.

        Args:
            audio_prompt (str): Path to the reference audio file.
            text (str): The text to synthesize.
            output_path (Optional[str]): Path to save the output .wav file. If None, returns audio data.
            **kwargs: Additional generation parameters to override `GenerationConfig` defaults.

        Returns:
            Union[str, Tuple[int, np.ndarray]]: The output file path or a tuple of (sample_rate, audio_data).
        """
        config = GenerationConfig(**kwargs)
        
        conditioning = self._audio_processor.get_mel_features(audio_prompt, self.device)
        sentences = self._process_text(text, config.max_text_tokens_per_sentence)
        
        wavs = []
        with tqdm(total=len(sentences), desc="Inference") as pbar:
            for i, sent in enumerate(sentences):
                self._update_progress(0.1 + 0.8 * (i / len(sentences)), f"Processing sentence {i+1}/{len(sentences)}")
                token_ids = self._tokens_to_ids(sent)
                codes = self._generate_codes(conditioning, token_ids, config)
                codes, code_lengths = remove_excessive_silence(codes, config.silent_token, self.stop_mel_token, config.max_consecutive_silence)
                latent = self._generate_latents(conditioning, token_ids, codes, code_lengths)
                wav = self._decode_audio(latent, conditioning)
                wavs.append(wav)
                pbar.update(1)
        
        self._clear_gpu_cache()
        self._update_progress(1.0, "Finished!")

        return self._finalize_audio(torch.cat(wavs, dim=1), output_path, config)

if __name__ == '__main__':
    print("IndexTTS class defined successfully.")
    print("To run inference, instantiate the class and call .infer() or .infer_fast()")
    # Example usage:
    # try:
    #     tts = IndexTTS()
    #     result = tts.infer_fast(
    #         "test_data/input.wav",
    #         "This is a test of the new and improved text to speech system.",
    #         "output.wav"
    #     )
    #     print(f"Audio saved to {result}")
    # except Exception as e:
    #     print(f"Could not run inference. Ensure model files are in ./checkpoints. Error: {e}")

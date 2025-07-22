from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TTSRequest:
    text: str
    audio_prompt: str
    output_path: str
    verbose: bool = False
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TTSConfig:
    # inference params
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

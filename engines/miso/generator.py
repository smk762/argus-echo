from dataclasses import dataclass
import os
from typing import Iterator, List, Optional, Tuple

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import MISO_TTS_8B_CONFIG, Model, ModelArgs
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
patch_bitsandbytes_import_for_unquantized_layers()


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992

    ``meta-llama/Llama-3.2-1B`` is gated (manual approval). We default to an
    ungated mirror that ships the identical Llama 3.2 tokenizer (same 128,256
    vocab). Override with MISO_TTS_TOKENIZER if you have Meta access.
    """
    tokenizer_name = os.environ.get("MISO_TTS_TOKENIZER", "unsloth/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        self._frame_size = self._model.config.audio_num_codebooks + 1

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(self._model.config.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}] {text.lstrip()}")
        text_frame = torch.zeros(len(text_tokens), self._frame_size).long()
        text_frame_mask = torch.zeros(len(text_tokens), self._frame_size).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._frame_size).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._frame_size).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, audio_num_codebooks + 1), (seq_len, audio_num_codebooks + 1)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            # End-of-speech on the first frame (or a budget under one frame): an
            # empty clip, as generate_stream yields nothing. torch.stack([]) raises.
            return torch.zeros(0, device=self.device)

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # If using Miso TTS in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, MISO_TTS_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio

    def _build_prompt(
        self, text: str, speaker: int, context: List[Segment]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        return prompt_tokens, prompt_tokens_mask

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        first_chunk_frames: int = 5,
        decode_chunk_frames: int = 12,
    ) -> Iterator[torch.Tensor]:
        """Yield 24 kHz mono float32 audio chunks (1-D tensors) as frames are
        produced, instead of decoding the whole clip up front.

        Mimi's transposed-conv decoder keeps continuity across calls via its
        streaming state, so chunk boundaries are seamless. The watermark step is
        intentionally skipped on this low-latency path (it operates on the whole
        clip and would re-introduce full-generation latency); use ``generate``
        for the watermarked, file-oriented output.
        """
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        prompt_tokens, prompt_tokens_mask = self._build_prompt(text, speaker, context)

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        frame_buf: List[torch.Tensor] = []
        target = max(1, first_chunk_frames)
        chunk = max(1, decode_chunk_frames)

        def _decode(buf: List[torch.Tensor]) -> torch.Tensor:
            codes = torch.stack(buf).permute(1, 2, 0)  # (1, K, n_frames)
            # One frame per call: moshi CUDA-graphs Mimi's streaming decoder, and a graph replays only the shape it
            # was captured with, so a first chunk, a full chunk and a short last chunk can't share it. The streaming
            # context carries the conv state across calls, so per-frame decodes concatenate gaplessly.
            frames = [self._audio_tokenizer.decode(codes[:, :, i : i + 1]) for i in range(codes.shape[-1])]
            return torch.cat(frames, dim=-1).squeeze(0).squeeze(0)

        # Streaming context keeps the decoder's conv state continuous across
        # the per-chunk decode calls so concatenated chunks are gapless.
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
                if torch.all(sample == 0):
                    break  # eos

                frame_buf.append(sample)

                curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), zeros_mask_1_1], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

                if len(frame_buf) >= target:
                    yield _decode(frame_buf)
                    frame_buf = []
                    target = chunk  # ramp up after the low-latency first chunk

            if frame_buf:
                yield _decode(frame_buf)


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state_dict[key.removeprefix("module.")] = value
    if not state_dict:
        raise ValueError("Checkpoint did not contain any tensor state_dict entries")
    return state_dict


def _rebuild_rope_caches(model: torch.nn.Module, device: torch.device) -> None:
    """torchtune's RoPE keeps its theta/cache as non-persistent buffers, so they
    are not part of the checkpoint. After a meta-device init + ``assign=True``
    load they are still on the meta device; rebuild them on the real device.
    The caches are computed in float32 regardless of model dtype (see
    ``build_rope_cache``), so this does not affect numerical precision."""
    for module in model.modules():
        rope_init = getattr(module, "rope_init", None)
        if callable(rope_init):
            with device:
                rope_init()


def _load_model(
    model_path_or_repo_id: str,
    config: ModelArgs,
    device: str,
    dtype: torch.dtype,
) -> Model:
    if os.path.isfile(model_path_or_repo_id):
        model_file = model_path_or_repo_id
    elif os.path.isdir(model_path_or_repo_id):
        model_file = os.path.join(model_path_or_repo_id, "model.safetensors")
    else:
        model_file = hf_hub_download(repo_id=model_path_or_repo_id, filename="model.safetensors")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Could not resolve model checkpoint: {model_path_or_repo_id}")

    target_device = torch.device(device)

    # Build the ~8B-param graph on the meta device so we never allocate the full
    # fp32 model (~33GB) in host RAM. Weights are streamed straight to the target
    # device in the checkpoint precision, keeping host RAM usage negligible.
    with torch.device("meta"):
        model = Model(config)

    if model_file.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("Install safetensors to load .safetensors checkpoint files") from exc

        # safetensors can load each tensor directly onto CUDA, avoiding a host
        # copy of the fp32 weights. Fall back to CPU staging for non-CUDA runs.
        load_device = str(target_device) if target_device.type == "cuda" else "cpu"
        state_dict = {}
        with safe_open(model_file, framework="pt", device=load_device) as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name).to(dtype)
    else:
        # weights_only: a .pt/.bin checkpoint is a pickle, and only its tensors
        # are wanted. Never let one run code in the server process.
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
        raw_state_dict = _state_dict_from_checkpoint(checkpoint)
        state_dict = {name: tensor.to(dtype) for name, tensor in raw_state_dict.items()}

    model.load_state_dict(state_dict, assign=True, strict=True)
    del state_dict

    _rebuild_rope_caches(model, target_device)

    leftover = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    ]
    if leftover:
        raise RuntimeError(
            f"Tensors left on meta device after load (first few): {leftover[:8]}"
        )

    model.to(device=target_device)
    model.eval()
    return model


def load_miso_8b(
    device: str = "cuda",
    model_path_or_repo_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Generator:
    source = model_path_or_repo_id or os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    model = _load_model(source, MISO_TTS_8B_CONFIG, device=device, dtype=dtype)
    return Generator(model)

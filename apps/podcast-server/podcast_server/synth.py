"""Model-side rendering: load VibeVoice-1.5B and turn a plan into audio.

Imported only by the worker process and the in-process CLI — never by the
API process, which must stay torch-free.

Inference is plain Hugging Face transformers (`VibeVoiceForConditionalGeneration`,
merged in v5.17). Nothing here post-processes the audio: what generate()
returns is what the caller gets.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
SAMPLES_PER_TOKEN = 3200          # acoustic tokenizer: one latent = 7.5 Hz at 24 kHz


def load_voice(path: str | Path) -> np.ndarray:
    """Read a 24 kHz mono 16-bit voice prompt as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: voice prompts must be 24 kHz mono 16-bit PCM")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class Rendered:
    audio: np.ndarray                   # float32 mono @ SAMPLE_RATE
    gen_secs: float
    peak_vram_gib: float                # torch max_memory_reserved during generate()
    audio_tokens: int
    finished: bool                      # model emitted EOS (False: hit max_new_tokens)
    # [start, end) sample ranges of each audio_bos…audio_eos span the model
    # emitted. VibeVoice-1.5B renders a whole dialogue as ONE span, so this
    # carries no per-turn boundaries (see README "Turn timings").
    segments: list[tuple[int, int]] = field(default_factory=list)


class Renderer:
    """Owns the loaded model + processor. One instance per process."""

    def __init__(self, model_id: str, revision: str, device: str = "cuda",
                 dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForTextToWaveform, AutoProcessor

        self._torch = torch
        t0 = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForTextToWaveform.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), device_map=device,
        ).eval()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_secs = time.monotonic() - t0
        self.resident_vram_gib = (torch.cuda.memory_reserved() / 2**30
                                  if device.startswith("cuda") else 0.0)
        cfg = self.model.config
        self._audio_bos = cfg.audio_bos_token_id
        self._audio_eos = cfg.audio_eos_token_id
        self._audio_token = cfg.audio_token_id
        self._eos = cfg.eos_token_id

    def build_inputs(self, turns: list[tuple[int, str]], voices: list[np.ndarray]):
        """Chat-template prompt with one voice prompt per role, in role order.

        The voice audio rides on each speaker's FIRST turn; the template
        emits the `Voice input` placeholders in that order, and the arrays
        are passed to the processor in the same order.
        """
        conversation, seen = [], set()
        for role, text in turns:
            content: list[dict] = [{"type": "text", "text": text}]
            if role not in seen:
                seen.add(role)
                content.append({"type": "audio", "path": f"voice-{role}.wav"})
            conversation.append({"role": str(role), "content": content})
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, audio=voices, sampling_rate=SAMPLE_RATE,
                                return_tensors="pt")
        return prompt, inputs.to(self.model.device, self.model.dtype)

    def render(self, turns: list[tuple[int, str]], voices: list[np.ndarray], *,
               seed: int, cfg_scale: float, diffusion_steps: int,
               max_new_tokens: int) -> Rendered:
        torch = self._torch
        from transformers import set_seed

        _, inputs = self.build_inputs(turns, voices)
        prompt_len = inputs["input_ids"].shape[1]
        cuda = self.model.device.type == "cuda"
        set_seed(seed)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                guidance_scale=cfg_scale,
                num_diffusion_steps=diffusion_steps,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        clip = out.audio[0]
        audio = (clip.float().cpu().numpy().reshape(-1) if clip is not None
                 else np.zeros(0, dtype=np.float32))
        gen_secs = time.monotonic() - t0
        peak = torch.cuda.max_memory_reserved() / 2**30 if cuda else 0.0

        generated = out.sequences[0, prompt_len:].tolist()
        segments, audio_tokens, start, open_span = [], 0, 0, True   # prompt ends with audio_bos
        for tok in generated:
            if tok == self._audio_token:
                audio_tokens += 1
            elif tok == self._audio_eos and open_span:
                segments.append((start * SAMPLES_PER_TOKEN, audio_tokens * SAMPLES_PER_TOKEN))
                open_span = False
            elif tok == self._audio_bos:
                start, open_span = audio_tokens, True
            elif tok == self._eos:
                break
        if open_span and audio_tokens > start:
            segments.append((start * SAMPLES_PER_TOKEN, audio_tokens * SAMPLES_PER_TOKEN))
        return Rendered(
            audio=audio,
            gen_secs=gen_secs,
            peak_vram_gib=peak,
            audio_tokens=audio_tokens,
            finished=self._eos in generated,
            segments=segments,
        )

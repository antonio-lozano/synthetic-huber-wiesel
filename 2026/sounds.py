import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoundConfig:
    sample_rate: int = 44100
    master_volume: float = 0.20
    max_rate_hz: float = 40.0
    n_channels: int = 24


class SpikeRateAudioEngine:
    """
    Real-time audio driven by firing rate and spikes.

    - Spikes trigger short click transients.
    - Firing rate controls granular melodic texture density and pitch region.
    """

    def __init__(self, cfg: SoundConfig = SoundConfig(), seed: int = 2026) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.enabled = False
        self._pygame = None
        self._tones = []
        self._click = None
        self._smooth_rate = 0.0
        self._last_spike_sound_t = 0.0

        try:
            import pygame

            self._pygame = pygame
            pygame.mixer.init(
                frequency=self.cfg.sample_rate,
                size=-16,
                channels=2,
                buffer=1024,
            )
            pygame.mixer.set_num_channels(self.cfg.n_channels)
            self._click = self._make_click()
            self._tones = self._make_tones()
            self.enabled = True
        except Exception as exc:
            print(f"[audio] Disabled: {exc}")

    def _make_sound(self, wave: np.ndarray):
        wave_i16 = np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16)
        stereo = np.column_stack([wave_i16, wave_i16])
        return self._pygame.mixer.Sound(buffer=stereo.tobytes())

    def _make_click(self):
        sr = self.cfg.sample_rate
        dur = 0.025
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        env = np.exp(-70.0 * t)
        wave = (np.sin(2.0 * np.pi * 2400.0 * t) * env).astype(np.float32)
        snd = self._make_sound(0.35 * wave)
        snd.set_volume(float(self.cfg.master_volume))
        return snd

    def _make_tones(self):
        # C-major pentatonic-like set, adapted from latent_explorer's idea.
        scale_freqs = [
            261.63,
            293.66,
            329.63,
            392.00,
            440.00,
            523.25,
            587.33,
            659.25,
            783.99,
            880.00,
        ]
        sr = self.cfg.sample_rate
        dur = 0.18
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        env = np.exp(-16.0 * t).astype(np.float32)
        sounds = []
        for f in scale_freqs:
            wave = np.sin(2.0 * np.pi * f * t)
            wave += 0.12 * np.sin(2.0 * np.pi * 2.0 * f * t)
            wave = (wave * env).astype(np.float32)
            snd = self._make_sound(0.23 * wave)
            snd.set_volume(float(self.cfg.master_volume))
            sounds.append(snd)
        return sounds

    def update(self, rate_hz: float, spike: int, dt_s: float) -> None:
        if not self.enabled:
            return

        r = float(np.clip(rate_hz / max(1e-6, self.cfg.max_rate_hz), 0.0, 1.0))
        self._smooth_rate += 0.15 * (r - self._smooth_rate)

        now = time.monotonic()
        if spike == 1 and (now - self._last_spike_sound_t) > 0.01:
            self._last_spike_sound_t = now
            vol = 0.55 + 0.45 * self._smooth_rate
            self._click.set_volume(float(vol * self.cfg.master_volume))
            ch = self._pygame.mixer.find_channel(True)
            if ch is not None:
                ch.play(self._click)

        # Rate-driven granular tones.
        p = (0.02 + 0.35 * self._smooth_rate) * max(dt_s * 30.0, 0.3)
        if self.rng.random() < p:
            hi = max(1, int(round((len(self._tones) - 1) * (0.3 + 0.7 * self._smooth_rate))))
            idx = int(self.rng.integers(0, hi + 1))
            tone = self._tones[idx]
            tone.set_volume(float((0.2 + 0.8 * self._smooth_rate) * self.cfg.master_volume))
            ch = self._pygame.mixer.find_channel(True)
            if ch is not None:
                ch.play(tone)

    def reset(self) -> None:
        self._smooth_rate = 0.0

    def stop(self) -> None:
        if not self.enabled:
            return
        try:
            self._pygame.mixer.quit()
        except Exception:
            pass
        self.enabled = False

"""Conservative automatic mixing and mastering for generated tracks.

This module deliberately uses NumPy only. Track dynamics are detected on the summed
track signal, then the same gain envelope is applied to each voice in that track. The
master envelope is likewise shared by every voice so exported stems still sum to mix.
"""

from __future__ import annotations

import re

import numpy as np

from .filters import Filter, apply as apply_filters
from .synth import SR, sum_tracks

ROLE_LEVEL_DB = {
    "vocal": 2.0,
    "lead": 1.0,
    "harmony": -2.0,
    "bass": -1.0,
    "percussion": -4.0,
    "other": -3.0,
}

COMPRESSOR = {
    "vocal": (-17.0, 2.0, 0.012, 0.10),
    "lead": (-16.0, 2.0, 0.012, 0.09),
    "harmony": (-18.0, 2.0, 0.020, 0.12),
    "bass": (-18.0, 2.5, 0.025, 0.12),
    "percussion": (-13.0, 3.0, 0.005, 0.08),
    "other": (-17.0, 2.0, 0.015, 0.10),
}

REVERB_WET = {
    "vocal": 0.10,
    "lead": 0.07,
    "harmony": 0.12,
    "bass": 0.02,
    "percussion": 0.04,
    "other": 0.07,
}

_ROLE_WORDS = {
    "percussion": ("drum", "kick", "snare", "hat", "perc", "ドラム", "キック", "スネア", "打楽器"),
    "vocal": ("vocal", "voice", "歌", "ボーカル", "ボイス"),
    "harmony": ("harmony", "chorus", "backing", "ハモリ", "コーラス", "伴唱"),
    "bass": ("bass", "sub", "ベース", "低音"),
    "lead": ("lead", "melody", "main", "主旋律", "メロディ", "リード"),
}


def _track_key(item) -> tuple[str, int]:
    return item.track_key if hasattr(item, "track_key") else (item.origin, item.track_index)


def _median_pitch(voice) -> float | None:
    pitches = [float(unit.target_f0_hz) for unit in voice.units if unit.target_f0_hz > 0]
    if not pitches:
        return None
    notes = 69 + 12 * np.log2(np.asarray(pitches, dtype=np.float64) / 440.0)
    return float(np.median(notes))


def infer_roles(rendered) -> dict[tuple[str, int], str]:
    """Infer useful mix roles from explicit names, source type, and target pitch range."""
    tracks = {}
    for item in rendered:
        tracks.setdefault(_track_key(item), item.voice)

    roles: dict[tuple[str, int], str] = {}
    unknown = []
    for key, voice in tracks.items():
        if voice.lyrics:
            roles[key] = "vocal"
            continue
        if voice.percussion:
            roles[key] = "percussion"
            continue
        name = re.sub(r"[_\-]+", " ", voice.track_name.lower())
        name_words = set(re.findall(r"[a-z]+", name))
        role = next((candidate for candidate, words in _ROLE_WORDS.items()
                     if any((word in name_words) if word.isascii() else (word in name) for word in words)), None)
        if role is None:
            unknown.append((key, voice, _median_pitch(voice)))
        else:
            roles[key] = role

    # A single non-percussion MIDI track is usually the melody. For multi-track
    # arrangements, use a clear pitch-range separation to identify bass and lead.
    if len(unknown) == 1 and not any(role == "lead" for role in roles.values()):
        roles[unknown[0][0]] = "lead"
    elif len(unknown) > 1:
        pitched = sorted((entry for entry in unknown if entry[2] is not None), key=lambda e: e[2])
        if len(pitched) >= 2 and pitched[-1][2] - pitched[0][2] >= 12:
            low, high = pitched[0], pitched[-1]
            if low[2] < 52 and "bass" not in roles.values():
                roles[low[0]] = "bass"
            if "lead" not in roles.values():
                roles[high[0]] = "lead"

    for key, _, _ in unknown:
        roles.setdefault(key, "other")
    return roles


def _active_loudness_db(audio: np.ndarray, sample_rate: int = SR, block_sec: float = 0.1) -> float:
    block = max(1, int(sample_rate * block_sec))
    count = len(audio) // block
    if count == 0:
        return -120.0
    chunks = audio[:count * block].astype(np.float64).reshape(count, block)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    active = rms[rms > max(1e-7, float(rms.max(initial=0.0)) * 10 ** (-50 / 20))]
    if not active.size:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(active * active))))


def balance_gains(rendered, roles: dict[tuple[str, int], str]) -> dict[tuple[str, int], float]:
    """Return bounded dB corrections that put tracks near a role-aware level target."""
    grouped: dict[tuple[str, int], list[np.ndarray]] = {}
    for item in rendered:
        grouped.setdefault(_track_key(item), []).append(item.audio)
    levels = {key: _active_loudness_db(sum_tracks(audio)) for key, audio in grouped.items()}
    sounding = [db for db in levels.values() if db > -100.0]
    if not sounding:
        return {key: 0.0 for key in levels}
    reference = float(np.median(sounding))
    return {
        key: float(np.clip(reference + ROLE_LEVEL_DB[roles.get(key, "other")] - db, -8.0, 5.0))
        if db > -100.0 else 0.0
        for key, db in levels.items()
    }


def _spectral_ratios(audio: np.ndarray, sample_rate: int = SR) -> tuple[float, float, float]:
    """Estimate low-mid, presence, and high-frequency energy from active audio frames."""
    frame, hop = 4096, 2048
    if len(audio) < frame:
        return 0.0, 0.0, 0.0
    total_frames = 1 + (len(audio) - frame) // hop
    picks = np.unique(np.linspace(0, total_frames - 1, min(48, total_frames), dtype=int))
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1 / sample_rate)
    total = low_mid = presence = high = 0.0
    for index in picks:
        chunk = audio[index * hop:index * hop + frame].astype(np.float64)
        if np.sqrt(np.mean(chunk * chunk)) < 1e-5:
            continue
        power = np.abs(np.fft.rfft(chunk * window)) ** 2
        total += float(power[(freqs >= 30) & (freqs <= min(18000, sample_rate / 2))].sum())
        low_mid += float(power[(freqs >= 150) & (freqs < 400)].sum())
        presence += float(power[(freqs >= 2000) & (freqs < 5000)].sum())
        high += float(power[(freqs >= 8000) & (freqs <= sample_rate / 2)].sum())
    if total <= 0:
        return 0.0, 0.0, 0.0
    return low_mid / total, presence / total, high / total


def eq_for_role(audio: np.ndarray, role: str, sample_rate: int = SR) -> list[Filter]:
    """Build a restrained EQ curve, with small corrections from measured band energy."""
    low_mid, presence, high = _spectral_ratios(audio, sample_rate)
    if role in ("vocal", "lead"):
        specs = ["hp:70"]
    elif role == "harmony":
        specs = ["hp:85"]
    elif role == "bass":
        specs = ["hp:28", "lp:10000"]
    elif role == "percussion":
        specs = ["hp:32"]
    else:
        specs = ["hp:38"]
    if role != "bass" and low_mid > 0.27:
        specs.append("peak:260:-1.5:0.8")
    if role in ("vocal", "lead") and 0 < presence < 0.055:
        specs.append("peak:3500:+0.8:0.8")
    if high > 0.18:
        specs.append("highshelf:9000:-1.0")
    # Reuse the established filter parser so validation and response shapes stay consistent.
    from .filters import parse

    return parse(",".join(specs))


def _gain_envelope(levels: np.ndarray, threshold_db: float, ratio: float,
                   attack_sec: float, release_sec: float, block_sec: float) -> np.ndarray:
    if not levels.size:
        return np.zeros(0, dtype=np.float32)
    desired = np.maximum(0.0, levels - threshold_db) * (1.0 - 1.0 / ratio)
    attack = np.exp(-block_sec / max(attack_sec, 1e-4))
    release = np.exp(-block_sec / max(release_sec, 1e-4))
    smoothed = np.empty_like(desired)
    previous = 0.0
    for i, target in enumerate(desired):
        coefficient = attack if target > previous else release
        previous = target + coefficient * (previous - target)
        smoothed[i] = previous
    return smoothed.astype(np.float32)


def compressor_envelope(audio: np.ndarray, role: str, sample_rate: int = SR) -> np.ndarray:
    """Create a track-bus gain curve from block RMS with role-specific gentle settings."""
    threshold, ratio, attack, release = COMPRESSOR.get(role, COMPRESSOR["other"])
    block = max(1, int(sample_rate * 0.01))
    count = (len(audio) + block - 1) // block
    if count == 0:
        return np.ones(len(audio), dtype=np.float32)
    padded = np.pad(audio.astype(np.float64), (0, count * block - len(audio)))
    rms = np.sqrt(np.mean(padded.reshape(count, block) ** 2, axis=1))
    levels = 20 * np.log10(rms + 1e-12)
    reduction = _gain_envelope(levels, threshold, ratio, attack, release, block / sample_rate)
    return _expand_gain(reduction, len(audio), block)


def _expand_gain(reduction_db: np.ndarray, sample_count: int, block: int) -> np.ndarray:
    if not sample_count:
        return np.zeros(0, dtype=np.float32)
    positions = (np.arange(len(reduction_db), dtype=np.float64) + 0.5) * block
    sample_positions = np.arange(sample_count, dtype=np.float64)
    expanded = np.interp(sample_positions, positions, reduction_db,
                          left=float(reduction_db[0]), right=float(reduction_db[-1]))
    return np.power(10.0, -expanded / 20.0).astype(np.float32)


def limiter_envelope(audio: np.ndarray, sample_rate: int = SR, ceiling_db: float = -1.2,
                     release_sec: float = 0.06) -> np.ndarray:
    """Offline peak limiter curve; blockwise look-ahead lets each block set its own ceiling."""
    block = max(1, int(sample_rate * 0.005))
    count = (len(audio) + block - 1) // block
    if count == 0:
        return np.ones(len(audio), dtype=np.float32)
    padded = np.pad(audio.astype(np.float64), (0, count * block - len(audio)))
    peaks = np.max(np.abs(padded.reshape(count, block)), axis=1)
    levels = 20 * np.log10(peaks + 1e-12)
    desired_gain_db = np.minimum(0.0, ceiling_db - levels)
    release = np.exp(-(block / sample_rate) / max(release_sec, 1e-4))
    smoothed = np.empty_like(desired_gain_db)
    previous = 0.0
    for i, target in enumerate(desired_gain_db):
        previous = target if target < previous else target + release * (previous - target)
        smoothed[i] = previous
    # _expand_gain expects attenuation as positive dB, while limiter values are gain dB.
    return _expand_gain(-smoothed, len(audio), block)


def apply_envelope(audio: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return (audio * envelope[:len(audio)]).astype(np.float32)


def _comb(audio: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    out = audio.astype(np.float64).copy()
    for start in range(delay, len(out), delay):
        end = min(len(out), start + delay)
        out[start:end] += feedback * out[start - delay:end - delay]
    return out


def add_room_reverb(audio: np.ndarray, wet: float, sample_rate: int = SR,
                    decay_sec: float = 0.55) -> np.ndarray:
    """Add a quiet, mono multi-comb room return without external DSP dependencies."""
    if wet <= 0 or not audio.size:
        return audio
    dry = audio.astype(np.float64)
    delays_ms = (29.7, 37.1, 41.1, 43.7, 47.3, 53.1)
    return_signal = np.zeros_like(dry)
    for delay_ms in delays_ms:
        delay = max(1, round(sample_rate * delay_ms / 1000))
        feedback = 10 ** (-3 * delay / sample_rate / decay_sec)
        return_signal += _comb(dry, delay, feedback) - dry
    return (dry + wet * return_signal / len(delays_ms)).astype(np.float32)


def apply_track_eq(rendered, roles: dict[tuple[str, int], str], sample_rate: int = SR) -> None:
    """Apply restrained, role-aware EQ to each track while retaining stem additivity."""
    grouped = {}
    for item in rendered:
        grouped.setdefault(_track_key(item), []).append(item)
    for key, members in grouped.items():
        role = roles.get(key, "other")
        track_audio = sum_tracks([item.audio for item in members])
        eq = eq_for_role(track_audio, role, sample_rate)
        if eq:
            for item in members:
                item.audio = apply_filters(item.audio, eq, sample_rate)


def apply_track_dynamics(rendered, roles: dict[tuple[str, int], str], sample_rate: int = SR) -> None:
    """Apply track-bus compression and small reverb while retaining stem additivity."""
    grouped = {}
    for item in rendered:
        grouped.setdefault(_track_key(item), []).append(item)
    for key, members in grouped.items():
        role = roles.get(key, "other")
        track_audio = sum_tracks([item.audio for item in members])
        envelope = compressor_envelope(track_audio, role, sample_rate)
        for item in members:
            item.audio = apply_envelope(item.audio, envelope)
            item.audio = add_room_reverb(item.audio, REVERB_WET[role], sample_rate)


def master_compressor_envelope(audio: np.ndarray, sample_rate: int = SR) -> np.ndarray:
    block = max(1, int(sample_rate * 0.01))
    count = (len(audio) + block - 1) // block
    if count == 0:
        return np.ones(len(audio), dtype=np.float32)
    padded = np.pad(audio.astype(np.float64), (0, count * block - len(audio)))
    rms = np.sqrt(np.mean(padded.reshape(count, block) ** 2, axis=1))
    levels = 20 * np.log10(rms + 1e-12)
    reduction = _gain_envelope(levels, -11.0, 1.5, 0.025, 0.15, block / sample_rate)
    return _expand_gain(reduction, len(audio), block)


def output_gain(audio: np.ndarray, sample_rate: int = SR,
                target_active_db: float = -16.0, peak_db: float = -1.0) -> tuple[float, float]:
    """Aim for a useful active RMS while keeping the sample peak under the output ceiling."""
    if not audio.size:
        return 1.0, -120.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return 1.0, -120.0
    active_db = _active_loudness_db(audio, sample_rate)
    loudness_gain = 10 ** ((target_active_db - active_db) / 20)
    peak_gain = 10 ** (peak_db / 20) / peak
    return min(loudness_gain, peak_gain), active_db

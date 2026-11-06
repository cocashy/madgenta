"""Vowel-first unit selection for MIDI melody synth mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .db import Corpus
from .phonemes import CONSONANTS, is_voiced_sustained, normalize
from .target import TargetUnit

ATTACK_MAX_SEC = 0.06
ATTACK_MAX_NOTE_RATIO = 0.4
CORE_DURATION_SAFETY_SEC = 0.04


@dataclass(frozen=True)
class MelodySynthCandidate:
    """A sustained vowel core with an optional directly preceding consonant."""

    vowel_idx: int
    onset_idx: int | None
    vowel_phoneme: str
    confidence: float


def build_candidates(corpus: Corpus) -> list[MelodySynthCandidate]:
    if corpus.cand_phoneme is None or corpus.cand_conf is None:
        raise SystemExit("melody synth mode needs phoneme analysis: run `madgen build-corpus --phonemes wav2vec2`")
    if corpus.core_start is None or corpus.core_end is None:
        raise SystemExit("melody synth mode needs updated phoneme analysis: rerun `madgen build-corpus --phonemes wav2vec2`")

    candidates = []
    for i in range(len(corpus.phoneme)):
        if (not np.isfinite(corpus.core_start[i])
                or not np.isfinite(corpus.core_end[i]) or corpus.core_end[i] <= corpus.core_start[i]
                or not np.isfinite(corpus.f0[i])):
            continue
        vowel_match = next(
            ((normalize(str(ph)), float(conf))
             for ph, conf in zip(corpus.cand_phoneme[i], corpus.cand_conf[i], strict=True)
             if ph and is_voiced_sustained(normalize(str(ph)))),
            None,
        )
        if vowel_match is None:
            continue

        onset = None
        previous = i - 1
        if (previous >= 0 and corpus.next_idx[previous] == i
                and corpus.source_id[previous] == corpus.source_id[i]
                and normalize(str(corpus.phoneme[previous])) in CONSONANTS):
            onset = previous
        candidates.append(MelodySynthCandidate(vowel_idx=i, onset_idx=onset,
                                               vowel_phoneme=vowel_match[0],
                                               confidence=vowel_match[1]))

    if not candidates:
        raise SystemExit(
            "phoneme corpus has no usable sustained vowels; rerun `madgen build-corpus --phonemes wav2vec2` "
            "with clearer voice material"
        )
    return candidates


def _attack_duration(candidate: MelodySynthCandidate, corpus: Corpus, note_dur: float) -> float:
    if candidate.onset_idx is None:
        return 0.0
    onset_len = corpus.end[candidate.onset_idx] - corpus.start[candidate.onset_idx]
    return max(0.0, min(onset_len, ATTACK_MAX_SEC, note_dur * ATTACK_MAX_NOTE_RATIO))


def select_units(units: list[TargetUnit], corpus: Corpus,
                 candidates: list[MelodySynthCandidate], top_k: int = 50
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Select vowel candidates; return candidate indices and whether each needs stretching."""
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    m = len(candidates)
    if m == 0:
        raise SystemExit("melody synth mode found no usable vowel candidates")
    k = min(top_k, m)
    vowel = np.asarray([c.vowel_idx for c in candidates], dtype=np.int64)
    onset = np.asarray([-1 if c.onset_idx is None else c.onset_idx for c in candidates], dtype=np.int64)
    assert corpus.core_start is not None and corpus.core_end is not None
    assert corpus.cand_conf is not None
    core_dur = corpus.core_end[vowel] - corpus.core_start[vowel]
    onset_len = np.zeros(m, dtype=np.float64)
    has_onset = onset >= 0
    onset_len[has_onset] = corpus.end[onset[has_onset]] - corpus.start[onset[has_onset]]
    seg_f0 = corpus.f0[vowel]
    seg_semitones = 12 * np.log2(seg_f0 / 440.0)
    loud_ref = float(np.percentile(corpus.rms_db[vowel], 90))
    confidence = np.clip(np.asarray([c.confidence for c in candidates]), 0.0, 1.0)

    cand = np.empty((n, k), dtype=np.int64)
    local = np.empty((n, k), dtype=np.float64)
    for i, unit in enumerate(units):
        attack = np.minimum(onset_len, min(ATTACK_MAX_SEC, unit.duration_sec * ATTACK_MAX_NOTE_RATIO))
        needed = np.maximum(0.0, unit.duration_sec - attack)
        required = needed + CORE_DURATION_SAFETY_SEC
        eligible = core_dur >= required

        target_semitone = 12 * np.log2(unit.target_f0_hz / 440.0)
        diff = np.abs(target_semitone - seg_semitones)
        pitch_cost = 0.15 * np.minimum(diff, 1.0) + np.maximum(diff - 1.0, 0.0)
        quality_cost = 1.5 * (1.0 - confidence)
        loudness_cost = 0.3 * np.maximum(0.0, loud_ref - corpus.rms_db[vowel]) / 10.0
        shortfall = np.maximum(0.0, required - core_dur)
        duration_cost = 100.0 * shortfall / np.maximum(required, 0.05)
        costs = pitch_cost + quality_cost + loudness_cost + duration_cost

        # Prefer native-length material whenever any candidate can fill the note. Stretching is
        # reserved for notes for which no sufficiently long vowel exists in the corpus.
        choices = np.flatnonzero(eligible) if eligible.any() else np.arange(m)
        take_n = min(k, len(choices))
        choice_cost = costs[choices]
        if take_n < len(choices):
            chosen = choices[np.argpartition(choice_cost, take_n - 1)[:take_n]]
        else:
            chosen = choices
        chosen = chosen[np.argsort(costs[chosen])]
        if len(chosen) < k:
            chosen = np.resize(chosen, k)
        cand[i] = chosen
        local[i] = costs[chosen]

    score = local[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        gap = units[i].start_sec - (units[i - 1].start_sec + units[i - 1].duration_sec)
        if gap > 0.05:
            trans = np.zeros((k, k))
        else:
            prev_ids, cur_ids = cand[i - 1], cand[i]
            prev_vowel, cur_vowel = vowel[prev_ids], vowel[cur_ids]
            cur_start = np.where(onset[cur_ids] >= 0, onset[cur_ids], cur_vowel)
            follows = corpus.next_idx[prev_vowel][:, None] == cur_start[None, :]
            same = ((prev_vowel[:, None] == cur_vowel[None, :])
                    & (onset[prev_ids][:, None] == onset[cur_ids][None, :]))
            trans = np.where(follows, 0.0, np.where(same, 0.2, 0.4))
        total = score[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        score = total[back[i], np.arange(k)] + local[i]

    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(score))
    for i in range(n - 1, -1, -1):
        path[i] = cand[i, j]
        j = back[i, j]

    stretch = np.zeros(n, dtype=bool)
    for i, (unit, ci) in enumerate(zip(units, path, strict=True)):
        c = candidates[ci]
        needed = max(0.0, unit.duration_sec - _attack_duration(c, corpus, unit.duration_sec))
        stretch[i] = core_dur[ci] < needed + CORE_DURATION_SAFETY_SEC
    return path, stretch

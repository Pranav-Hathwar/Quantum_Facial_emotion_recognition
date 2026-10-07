"""The seven FER2013 emotion classes (Chapter 12, Sec. IV).

The order is the FER2013 integer coding used in fer2013.csv:
0=Angry 1=Disgust 2=Fear 3=Happy 4=Sad 5=Surprise 6=Neutral.
Neutral is a separate, seventh class. Do not reorder: trained weights depend on it.
"""
from __future__ import annotations

EMOTIONS: tuple[str, ...] = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
NUM_CLASSES: int = len(EMOTIONS)

INDEX_TO_EMOTION: dict[int, str] = dict(enumerate(EMOTIONS))
EMOTION_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(EMOTIONS)}


def label_to_index(name: str) -> int:
    """'Happy' / 'happy' -> 3. Raises ValueError for unknown names."""
    key = name.strip().lower()
    if key not in EMOTION_TO_INDEX:
        raise ValueError(f"Unknown emotion '{name}'. Expected one of {EMOTIONS}.")
    return EMOTION_TO_INDEX[key]


def index_to_label(index: int) -> str:
    if index not in INDEX_TO_EMOTION:
        raise ValueError(f"Emotion index {index} out of range 0..{NUM_CLASSES - 1}.")
    return INDEX_TO_EMOTION[index]


def display_name(name: str) -> str:
    return name.strip().capitalize()

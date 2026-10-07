import pytest
from ml.labels import EMOTIONS, NUM_CLASSES, label_to_index, index_to_label


def test_exactly_seven_classes_in_fer2013_order():
    assert NUM_CLASSES == 7
    assert EMOTIONS == ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def test_neutral_is_its_own_class():
    assert "neutral" in EMOTIONS and label_to_index("Neutral") == 6


@pytest.mark.parametrize("i,name", list(enumerate(EMOTIONS)))
def test_round_trip(i, name):
    assert label_to_index(name) == i and index_to_label(i) == name


def test_unknown_label():
    with pytest.raises(ValueError):
        label_to_index("contempt")
    with pytest.raises(ValueError):
        index_to_label(7)

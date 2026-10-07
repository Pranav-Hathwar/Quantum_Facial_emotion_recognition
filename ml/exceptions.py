"""Domain errors so the API can map them to clear messages instead of crashing."""


class QuantumVisionError(Exception):
    """Base class for expected, user-facing errors."""


class InvalidImageError(QuantumVisionError):
    """The bytes/path could not be decoded as a supported image."""


class NoFaceDetectedError(QuantumVisionError):
    def __init__(self, message: str = "No face detected. Please upload an image containing a clearly visible face."):
        super().__init__(message)


class DatasetNotFoundError(QuantumVisionError):
    """FER2013 is missing or not in the expected layout."""

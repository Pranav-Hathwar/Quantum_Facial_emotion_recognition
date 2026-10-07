from __future__ import annotations

import numpy as np
from fastapi import HTTPException, UploadFile

from ml.exceptions import InvalidImageError
from ml.preprocessing.image_preprocessor import SUPPORTED_EXTENSIONS, decode_image


async def read_image_upload(file: UploadFile, max_mb: int) -> tuple[np.ndarray, bytes]:
    name = (file.filename or "").lower()
    if name and not name.endswith(SUPPORTED_EXTENSIONS) and not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Unsupported file type. Please upload a JPG, JPEG, PNG or WEBP image.")
    data = await file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {max_mb} MB.")
    try:
        return decode_image(data), data
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc

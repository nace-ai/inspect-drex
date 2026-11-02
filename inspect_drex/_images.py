"""Turn the images Inspect code holds into the entries Drex reads in `images`."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from typing import Any

import nace_sdk

from inspect_drex.errors import DrexImageError

_REMOTE = ("http://", "https://", "gs://")


def _is_base64(value: str) -> bool:
    if len(value) % 4:
        return False
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def as_data_url(image: str) -> str:
    """A data URL for base64 image bytes, typed from their first bytes."""
    if image.startswith("data:"):
        return image
    head = base64.b64decode(image[:24] + "=" * (-len(image[:24]) % 4), validate=False)
    mime = None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif head.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        mime = "image/webp"
    return f"data:{mime or 'application/octet-stream'};base64,{image}"


def to_drex_image(item: Any) -> str | dict[str, Any]:
    """Return one `images` entry for Drex.

    Accepts base64, a data URL, bytes, a path, an Inspect `ContentImage`, an
    `UploadedFile`/`workspace_file` reference. Remote URLs are refused: Drex never
    fetches them, so load the image into the sample (a data URL) or upload it.

    Raises:
        DrexImageError: The item isn't an image Drex can read.

    """
    image = getattr(item, "image", None)
    if isinstance(image, str) and not isinstance(item, Mapping):  # inspect ContentImage
        item = image
    if isinstance(item, str):
        if item.startswith("data:"):
            return item
        if item.startswith(_REMOTE):
            raise DrexImageError(
                f"Drex doesn't fetch image URLs ({item[:60]}…). Put the image "
                "into the sample as a data URL or a local file, or upload it with `nace_sdk` "
                "and pass the uploaded file."
            )
        if item and os.path.exists(item):
            return _from_sdk(item)
        if item and _is_base64(item):
            return item
        raise DrexImageError(f"{item[:60]!r} is not base64, a data URL, or an existing file path.")
    if isinstance(item, bytes | bytearray | os.PathLike):
        return _from_sdk(item)
    if isinstance(item, nace_sdk.UploadedFile | nace_sdk.WorkspaceFileSource):
        return _from_sdk(item)
    if isinstance(item, Mapping) and item.get("type") == "workspace_file":
        return dict(item)
    raise DrexImageError(f"Unsupported image type: {type(item).__name__}.")


def _from_sdk(item: Any) -> str | dict[str, Any]:
    try:
        return nace_sdk.image_input(item)
    except nace_sdk.NaceError as error:
        raise DrexImageError(str(error)) from error

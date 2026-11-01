"""Platform paths used by the desktop diagnostic application."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


def _windows_downloads() -> Optional[Path]:
    """Resolve the redirected Windows Downloads known folder."""
    if os.name != "nt":
        return None
    # FOLDERID_Downloads = {374DE290-123F-4565-9164-39C4925E467B}
    class Guid(ctypes.Structure):
        _fields_ = [
            ("data1", ctypes.c_ulong),
            ("data2", ctypes.c_ushort),
            ("data3", ctypes.c_ushort),
            ("data4", ctypes.c_ubyte * 8),
        ]

    folder_id = Guid(
        0x374DE290,
        0x123F,
        0x4565,
        (ctypes.c_ubyte * 8)(0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B),
    )
    result = ctypes.c_wchar_p()
    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    status = shell32.SHGetKnownFolderPath(
        ctypes.byref(folder_id), 0, None, ctypes.byref(result)
    )
    if status != 0 or not result.value:
        return None
    try:
        return Path(result.value)
    finally:
        ole32.CoTaskMemFree(result)


def diagnostic_root() -> Path:
    """Return the local root used for downloads and generated reports."""
    downloads = _windows_downloads() or (Path.home() / "Downloads")
    return downloads / "PPOTrial-CI"


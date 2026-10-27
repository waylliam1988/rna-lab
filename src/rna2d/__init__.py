"""Minimal, independently testable RNA secondary structure components."""
import os
from pathlib import Path

# Optional app-local signed VC runtime on Windows. No system PATH modification.
_native_handles = []
_native_dir = Path(__file__).resolve().parents[2] / '.runtime/vc'
if os.name == 'nt' and _native_dir.is_dir():
    _native_handles.append(os.add_dll_directory(str(_native_dir)))

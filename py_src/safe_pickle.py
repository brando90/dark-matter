"""
Exact allow-list unpickler for the third-party Porian et al. results file (data/data_resolving_discrepancies/*.pickle.xz).

A pickle can execute arbitrary code on load, and this file comes from a public GitHub repository. The loader below
resolves only the thirteen (module, name) globals that this one file asks for and raises on anything else. The list was
found by a dry run that recorded every global the file requests; there is no prefix rule and no wildcard. This is meant
for that one trusted file. It is not a general sandbox for untrusted pickles.

    from py_src.safe_pickle import load_xz_pickle
    df = load_xz_pickle(path, sha256="<hex digest of the xz file>")    # the digest check is optional

If a re-exported file needs another global, the loader raises "BLOCKED global module.name". Add the pair only after
looking at what it is (a dry run subclass that records the requests while delegating to this class shows the full set).

Run the loader with `python -I` (isolated mode) so nothing in the working directory is importable.
"""
from __future__ import annotations

import hashlib
import lzma
import pickle

from pandas.compat.pickle_compat import Unpickler as _PandasUnpickler

# The globals the Porian pickle requests, as written in the file (older numpy and pandas module names included).
ALLOWED_GLOBALS = frozenset({
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.numeric", "Int64Index"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.series", "Series"),
})


class SafeUnpickler(_PandasUnpickler):
    """Unpickler whose only way to reach a global is the exact pair list above.

    find_class is the single path by which pickle opcodes (GLOBAL, STACK_GLOBAL, INST and the extension codes)
    resolve a name, so refusing there refuses a module object, a builtin, a dunder name or a dotted attribute path.
    Persistent ids and out-of-band buffers are refused by the base class when no handler is given.
    """

    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"BLOCKED global {module}.{name}")
        return super().find_class(module, name)


def load_xz_pickle(path, sha256: str | None = None):
    """Load an xz-compressed pickle through the exact allow-list; optionally check the file's SHA-256 first."""
    if sha256 is not None:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256.lower():
            raise ValueError(f"{path}: SHA-256 {digest.hexdigest()} does not match the expected {sha256.lower()}")
    with lzma.open(path, "rb") as f:
        return SafeUnpickler(f).load()

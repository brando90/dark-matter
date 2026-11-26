"""
Tests for py_src/safe_pickle.py. None of the probes runs attacker code: each one is a tiny pickle that only asks the
loader to resolve a global, which the exact allow-list must refuse. The positive test loads the real Porian file and
checks the frame against a digest recorded with the loader that was used for experiment E4.

    python3 -I test_safe_pickle.py        (run from any directory; prints PASS or the first failure, exit code 1 on failure)
"""
import hashlib
import io
import pickle
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from py_src import safe_pickle as sp  # noqa: E402

PICKLE = REPO_ROOT / "data" / "data_resolving_discrepancies" / "experiment_results.pickle.xz"
PICKLE_SHA256 = "eae02848c22e72f90d1a08fb4d3f9782941563fd70636a33ce30692ec709bc34"
# SHA-256 of df.to_csv() for the frame that the earlier, prefix-based loader returned (975 rows, 29 columns).
FRAME_SHA256 = "c3a46da24f24002d3a6edcf72e06bdc3644ce1d089d3388a82f09b04ac2124f4"

FAILS = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    if not ok:
        FAILS.append(label)


def only_global_pickle(module: str, name: str, protocol: int) -> bytes:
    """A pickle whose only action is to resolve module.name (protocol 0 GLOBAL, or protocol 4 STACK_GLOBAL)."""
    if protocol == 0:
        return f"c{module}\n{name}\n.".encode()
    m, n = module.encode(), name.encode()
    return b"\x80\x04" + b"\x8c" + bytes([len(m)]) + m + b"\x94" + b"\x8c" + bytes([len(n)]) + n + b"\x94" + b"\x93."


def refused(module, name, protocol):
    try:
        sp.SafeUnpickler(io.BytesIO(only_global_pickle(module, name, protocol))).load()
    except pickle.UnpicklingError as e:
        return "BLOCKED" in str(e)
    return False


PROBES = [
    ("builtins", "eval"), ("builtins", "exec"), ("builtins", "getattr"), ("builtins", "__import__"), ("builtins", "open"),
    ("os", "system"), ("subprocess", "Popen"), ("posix", "system"),
    ("pandas.io.pickle", "read_pickle"), ("pandas.core.computation.eval", "eval"),
    # the bypass chain found in QA: dunder names and module objects reachable as attributes of pandas.core.* modules
    ("pandas.core.dtypes.common", "__builtins__"), ("pandas.core.frame", "com"), ("pandas.core.config_init", "os"),
    ("pandas.core.common", "builtins"), ("pandas.core.generic", "pickle"), ("pandas.core.common", "partial"),
    ("pandas.core.common", "pipe"),
    # a dotted attribute path (resolved by the protocol 4 STACK_GLOBAL opcode) and a wrong module for an allowed name
    ("pandas.core.config_init", "os.system"), ("pandas.core.generic", "DataFrame"),
]
for mod, nm in PROBES:
    for proto in (0, 4):
        check(f"refuses {mod}.{nm} (protocol {proto})", refused(mod, nm, proto))

# every allowed pair really is accepted by find_class (so the list is not stale against the installed pandas and numpy)
for mod, nm in sorted(sp.ALLOWED_GLOBALS):
    try:
        sp.SafeUnpickler(io.BytesIO(b"")).find_class(mod, nm)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print("   ", type(e).__name__, e)
    check(f"allowed pair resolves: {mod}.{nm}", ok)

# the real file: digest check passes, frame is the one E4 used, a wrong digest is refused
df = sp.load_xz_pickle(PICKLE, sha256=PICKLE_SHA256)
check("real Porian file loads with the pinned SHA-256", df.shape == (975, 29))
check("loaded frame equals the frame from the earlier loader", hashlib.sha256(df.to_csv().encode()).hexdigest() == FRAME_SHA256)
try:
    sp.load_xz_pickle(PICKLE, sha256="0" * 64)
    check("a wrong SHA-256 is refused", False)
except ValueError:
    check("a wrong SHA-256 is refused", True)

# a freshly pickled frame references globals that are not on the list, so it must be refused rather than loaded
import lzma  # noqa: E402

import pandas as pd  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "fresh.pickle.xz"
    with lzma.open(p, "wb") as f:
        pickle.dump(pd.DataFrame({"a": [1, 2]}), f)
    try:
        sp.load_xz_pickle(p)
        check("a freshly pickled frame (other globals) is refused", False)
    except pickle.UnpicklingError as e:
        check("a freshly pickled frame (other globals) is refused", "BLOCKED" in str(e))

print("PASS" if not FAILS else f"FAILED {len(FAILS)}: {FAILS}")
sys.exit(1 if FAILS else 0)

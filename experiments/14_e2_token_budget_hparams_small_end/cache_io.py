"""E2 resumable NumPy caches and observation-level Gaussian innovations (2026-11-26)."""
import hashlib
import json
from pathlib import Path

import numpy as np

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


VERSION = 'e2-amendment-2026-11-26-v1'


def encode(obj, arrays):
    if isinstance(obj, np.ndarray):
        key = f'array_{len(arrays)}'
        arrays[key] = obj
        return {'array': key}
    if isinstance(obj, dict):
        return {'dict': [[encode(k, arrays), encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return {'list': [encode(v, arrays) for v in obj]}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def decode(obj, arrays):
    if isinstance(obj, dict):
        if 'array' in obj:
            return arrays[obj['array']].copy()
        if 'dict' in obj:
            return {decode(k, arrays): decode(v, arrays) for k, v in obj['dict']}
        return [decode(v, arrays) for v in obj['list']]
    return obj


def digest(payload):
    arrays = {}
    obj = encode(payload, arrays)
    h = hashlib.sha256(json.dumps([VERSION, obj], sort_keys=True).encode())
    for key, value in arrays.items():
        a = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = json.dumps(encode(obj, arrays))
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, metadata=np.array(meta), **arrays)
    tmp.replace(path)


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return decode(json.loads(str(z['metadata'])), z)


def cached(path, payload, rng, compute):
    key = digest({'payload': payload, 'rng': rng.bit_generator.state})
    path = Path(path)
    if path.exists():
        item = read(path)
        if item['key'] == key:
            rng.bit_generator.state = item['rng_after']
            print(f'cached {path.name}', flush=True)
            return item['result']
    result = compute()
    save(path, {'key': key, 'version': VERSION, 'rng_after': rng.bit_generator.state, 'result': result})
    print(f'saved {path.name}', flush=True)
    return result


class NoiseBank:
    """One standardized innovation per observed number; stable across processing order/count convention."""
    def __init__(self, namespace, B):
        self.namespace, self.B, self.vectors = namespace, B, {}

    def get(self, recipe, width, tokens):
        key = f'{self.namespace}|{recipe}|{float(width):.8g}|{float(tokens):.12g}'
        if key not in self.vectors:
            seed = np.frombuffer(hashlib.sha256(key.encode()).digest(), dtype='<u4')
            self.vectors[key] = np.random.default_rng(seed).normal(size=self.B)
        return self.vectors[key]

    def matrix(self, recipe, points, B=None):
        B = self.B if B is None else B
        return np.column_stack([self.get(recipe, w, d)[:B] for w, d in zip(points.width, points.D)])

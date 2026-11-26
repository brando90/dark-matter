"""Resume the fixed amended E2 controls independently of the real-data fits (2026-11-26)."""
import verify as V
import cache_io as C
import e2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    rows = []
    for fast in (False, True):
        for rep in range(V.REPS):
            path = V.RES / 'controls' / f'result_{int(fast)}_{rep:02}.npz'
            rng = np.random.default_rng([e2.SEED, 99, int(fast), rep])
            row = C.cached(path, V.control_payload(fast, rep), rng, lambda: V.control(fast, rep))
            rows.append(row)
            pd.DataFrame(rows).to_csv(V.RES/'verify_controls.csv', index=False)
            print(row, flush=True)
    print('Controls complete; run verify.py to evaluate the unchanged gates.', flush=True)

"""E2 checks (2026-11-26): software integrity and unchanged preregistered scientific control gates.

python verify.py                  # all checks; nonzero if a scientific control gate fails
python verify.py --software-only  # integrity/reproduction only; not a scientific gate pass
Original failed suite: results/original_2026-11-26/verify_original.txt.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import e2
import e2lib as L
import cache_io as CACHE

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
TRUTH = dict(E=2.0, A=3.0, alpha=0.6, B=1.0, beta=0.2)
FAST = dict(size=0.3, alpha=1.4)
NOISE, REPS, BN, BJ = 0.003, 20, 300, 100


def truth(N, D, fast):
    y = TRUTH['E'] + TRUTH['A']*(N/1e6)**-TRUTH['alpha'] + TRUTH['B']*(D/1e9)**-TRUTH['beta']
    if fast:
        y += FAST['size'] * (N / 5173248)**-FAST['alpha']
    return y


def control(fast, seed):
    sd = {w: NOISE for (ds, w) in L.seed_sd() if ds == 'rw'}
    rng = np.random.default_rng([e2.SEED, 99, int(fast), seed])
    lad = L.ladder('rw', 'TS', 20).copy()
    lad['y'] = truth(lad.N_tied.values, 20*lad.N_tied.values, fast) + rng.normal(0, NOISE, len(lad))
    J = e2.joint_points_tc('rw').copy()
    J['y'] = truth(J.N_tied.values, J.D.values, fast) + rng.normal(0, NOISE, len(J))
    # Reuse observed TC20 values; preserve RNG consumption for all other observations.
    by_observation = {(r.width, r.D): r.y for r in lad.itertuples()}
    for ix, r in J.iterrows():
        if (r.width, r.D) in by_observation:
            J.loc[ix, 'y'] = by_observation[(r.width, r.D)]
    bank = CACHE.NoiseBank(f'{e2.SEED}|control|{int(fast)}|{seed}', BN)
    eps = {float(r.width): bank.get('TS', r.width, r.D) for r in lad.itertuples() if r.width in L.SMALL}
    ref = lad.N_tied >= L.REF_MIN
    X = e2.excess_boot(lad, 'tied', sd, rng, BN, eps, bank.matrix('TS', lad[ref]))
    small = lad[lad.width.isin(list(L.SMALL))]
    partial = RES / 'controls' / f'partial_{int(fast)}_{seed:02}.npz'
    XJ = e2.joint_boot(J, small, 'tied', sd, rng, BJ, eps, bank.matrix('TS', J, BJ), partial)
    w = 96.0
    c = X['obs'][w] - XJ['obs'][w]
    lo, hi = e2.ci(X['boot'][w][:BJ] - XJ['boot'][w])
    vJ = e2.verdict(c, lo, hi, X['obs'][w], 2*sd[w])
    mlo, mhi = e2.ci(XJ['boot'][w])
    vM = e2.ret_verdict(XJ['obs'][w], mlo, mhi, 2*sd[w])
    return dict(fast=int(fast), rep=seed, X=X['obs'][w], XJ=XJ['obs'][w], J=c,
                J_lo=lo, J_hi=hi, M_lo=mlo, M_hi=mhi, vJ=vJ, vM=vM,
                synthetic_sd=NOISE, B_N=BN, B_J=BJ, joint_sd=XJ['sd_fit'])


def control_payload(fast, rep):
    cols = ['width', 'N_nonemb', 'N_tied', 'D']
    return dict(fast=int(fast), rep=rep, truth=TRUTH, FAST=FAST, NOISE=NOISE,
                B_N=BN, B_J=BJ, version=CACHE.VERSION,
                ladder_design=L.ladder('rw', 'TS', 20)[cols].to_numpy(),
                joint_design=e2.joint_points_tc('rw')[cols].to_numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--software-only', action='store_true')
    args = ap.parse_args()
    out, software = [], []

    def report(name, ok, msg, scientific=False):
        if not scientific:
            software.append(bool(ok))
        out.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        print(out[-1], flush=True)

    cur = pd.read_csv(os.path.join(L.DM2, 'results', 'curves.csv'))
    X = pd.read_csv(RES / 'e2_excess_tied.csv')
    msgs, ok = [], True
    for ds, ref in (('rw', .347), ('owt2', .371)):
        x = X[(X.dataset == ds) & (X.cond == 'TC20') & (X['size'] == '5M')].X.iloc[0]
        ok &= abs(x-ref) <= .005
        msgs.append(f'{ds} X_5M(TC20)={x:+.6f}, reference={ref:+.3f}')
        for c, curve in (('TC20', 'tuned-short-const'), ('BC20', 'base-short-chinchilla')):
            lad = L.ladder(ds, *e2.COND[c])
            refy = cur[(cur.curve == f'porian/{ds}/{curve}') & (cur.slice == 'D=20N')].dropna(subset=['y'])
            m = lad.merge(refy[['width', 'y']], on='width', suffixes=('', '_dm2'))
            err = float(np.max(np.abs(m.y-m.y_dm2)))
            ok &= err <= 1e-9 and len(m) == len(lad) == len(refy)
            msgs.append(f'{ds} {c} |dy|max={err:.1e}, {len(m)} sizes')
    report('software: reproduction', ok, '; '.join(msgs))

    errors, files, complete = [], [], True
    for count in ['tied', 'untied']:
        x = pd.read_csv(RES / f'e2_excess_{count}.csv')
        f = pd.read_csv(RES / f'e2_factors_{count}.csv')
        j = pd.read_csv(RES / f'e2_joint_{count}.csv')
        sw = pd.read_csv(RES / f'e2_sweep_{count}.csv')
        for row in f.itertuples():
            sel = x[(x.dataset == row.dataset) & (x['size'] == row.size)]
            get = lambda c: sel[sel.cond == c].X.iloc[0]
            if row.factor.startswith('T (TC'):
                expected = get('TCfin') - get('TC20')
            elif row.factor.startswith('T (BC'):
                expected = get('BCfin') - get('BC20')
            elif row.factor.startswith('T2'):
                expected = get('BKfin') - get('BC20')
            elif row.factor.startswith('J ('):
                tag = 'TC' if '(TC:' in row.factor else 'BC'
                jr = j[(j.dataset == row.dataset) & (j.joint == tag)].iloc[0]
                expected = get(tag+'20') - jr[f'XJ_{row.size}']
            elif row.factor.startswith('R ('):
                expected = get('BC20') - get('TC20')
            elif row.factor.startswith('H ('):
                expected = sw[sw.width == row.width].H.iloc[0]
            elif row.factor.startswith('G ('):
                expected = sw[sw.width == row.width].G.iloc[0]
            else:
                expected = get('BC20') - get('TC20') - sw[sw.width == row.width].H.iloc[0]
            errors.append(abs(row.C-expected))
        for ds in ['rw', 'owt2']:
            directory = RES / 'bootstrap' / count / ds
            for cond in e2.COND:
                path = directory / f'{cond}.npz'
                obj = CACHE.read(path)['result']
                if obj is not None:
                    complete &= obj['B'] == e2.B_N and all(len(v) == e2.B_N and np.isfinite(v).all() for v in obj['boot'].values())
                    files.append(dict(count=count, dataset=ds, kind=cond, draws=obj['B']))
            for tag in ['TC', 'BC']:
                path = directory / f'joint_{tag}.npz'
                obj = CACHE.read(path)['result']
                part = CACHE.read(directory / f'partial_joint_{tag}.npz')
                complete &= obj['B'] == e2.B_J and part['completed'] == e2.B_J
                complete &= all(len(v) == e2.B_J and np.isfinite(v).all() for v in obj['boot'].values())
                files.append(dict(count=count, dataset=ds, kind='joint_'+tag, draws=obj['B']))
            if ds == 'rw':
                obj = CACHE.read(directory / 'sweep.npz')['result']
                complete &= all(len(v) == e2.B_N for v in obj['Gb'].values())
                files.append(dict(count=count, dataset=ds, kind='sweep', draws=e2.B_N))
    report('software: factor arithmetic', max(errors) < 1e-12, f'{len(errors)} identities; max error={max(errors):.3g}')
    pd.DataFrame(files).to_csv(RES / 'bootstrap_manifest.csv', index=False)
    report('software: persisted draws', complete, f'{len(files)} completed caches; N/sweep B={e2.B_N}, joint B={e2.B_J}')
    m = pd.read_csv(RES / 'e2_matched_tied.csv')
    missing = m[(m.dataset == 'rw') & (m['size'] == '9M')].iloc[0]
    report('software: unavailable sweep correction', pd.isna(missing.M) and 'UNDETERMINED' in missing.verdict,
           'rw 9M corrected M unavailable, never G=0')
    bank = CACHE.NoiseBank('integrity', 17)
    ladder = L.ladder('rw', 'TS', 20)
    ref = ladder[ladder.N_tied >= L.REF_MIN]
    joint = e2.joint_points_tc('rw')
    a, b = bank.matrix('TS', ref), bank.matrix('TS', joint)
    overlaps = [(i, j) for i, r in enumerate(ref.itertuples()) for j, t in enumerate(joint.itertuples())
                if r.width == t.width and r.D == t.D]
    paired = all(np.array_equal(a[:, i], b[:, j]) for i, j in overlaps)
    report('software: observation pairing', paired and len(overlaps) == len(ref),
           f'{len(overlaps)} TC20 reference observations reuse their joint-design innovations exactly')
    diagnostic = []
    for fast in (False, True):
        yl = truth(ladder.N_tied.values, ladder.D.values, fast)
        yj = truth(joint.N_tied.values, joint.D.values, fast)
        mask = ladder.N_tied.values >= L.REF_MIN
        f1 = L.fit1(ladder.N_tied.values[mask]/1e6, yl[mask])
        fj = L.F.fitJ1(joint.N_tied.values/1e6, joint.D.values/1e9, yj)
        x5 = float(yl[0] - L.pred1(f1, ladder.N_tied.values[0]/1e6))
        xj5 = float(yl[0] - L.F.predJ(fj, ladder.N_tied.values[:1]/1e6, ladder.D.values[:1]/1e9)[0])
        diagnostic.append(dict(planted_fast_term=int(fast), X5=x5, XJ5=xj5, J5=x5-xj5, J_fraction=(x5-xj5)/x5))
    pd.DataFrame(diagnostic).to_csv(RES/'deterministic_diagnostic.csv', index=False)
    report('software: noiseless joint-law recovery', abs(diagnostic[0]['XJ5']) < 1e-4,
           f"no-fast XJ5={diagnostic[0]['XJ5']:.9f}; planted J fraction={diagnostic[1]['J_fraction']:.6f}; does not replace statistical gates")
    scientific_ok = None
    if not args.software_only:
        rows = []
        for fast in (False, True):
            for rep in range(REPS):
                path = RES / 'controls' / f'result_{int(fast)}_{rep:02}.npz'
                rng = np.random.default_rng([e2.SEED, 99, int(fast), rep])
                row = CACHE.cached(path, control_payload(fast, rep), rng, lambda: control(fast, rep))
                rows.append(row)
                pd.DataFrame(rows).to_csv(RES / 'verify_controls.csv', index=False)
        c = pd.DataFrame(rows)
        p, n = c[c.fast == 0], c[c.fast == 1]
        pj, pm = int((p.vJ == 'EXPLAINS').sum()), int((p.vM == 'RETURNS').sum())
        nj, nm = int((n.vJ != 'EXPLAINS').sum()), int((n.vM == 'STAYS OFF').sum())
        positive, null = pj >= .8*REPS and pm >= .8*REPS, nj >= .8*REPS and nm >= .8*REPS
        report('scientific gate: positive control', positive,
               f'J EXPLAINS {pj}/{REPS}; M RETURNS {pm}/{REPS}; both require >=16/20', True)
        report('scientific gate: planted-fast-term control', null,
               f'J not EXPLAINS {nj}/{REPS}; M STAYS OFF {nm}/{REPS}; both require >=16/20', True)
        scientific_ok = bool(positive and null)
    status = dict(software_pass=bool(all(software)), scientific_gate_pass=scientific_ok,
                  original_scientific_gate_pass=False, date='2026-11-26',
                  meaning='Software-only checks do not validate a scientific verdict.')
    if args.software_only:
        suffix = '_software'
    else:
        suffix = ''
    (RES / f'verify{suffix}.json').write_text(json.dumps(status, indent=2)+'\n')
    out.append('SOFTWARE '+('PASS' if status['software_pass'] else 'FAIL'))
    out.append('SCIENTIFIC GATE '+('NOT RUN' if scientific_ok is None else 'PASS' if scientific_ok else 'FAILED; verdicts conditional/descriptive'))
    print('\n'.join(out[-2:]), flush=True)
    (RES / f'verify{suffix}.txt').write_text('\n'.join(out)+'\n')
    return 0 if all(software) and scientific_ok is not False else 1


if __name__ == '__main__':
    raise SystemExit(main())

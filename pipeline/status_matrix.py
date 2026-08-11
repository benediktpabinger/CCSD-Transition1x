"""The complete status of every candidate structure of every reaction.

One row per candidate, one block per reaction, and for each the state of all
three stages. The point of the exercise is the third state in every cell:

    bestanden        the test ran and the structure passed
    DURCHGEFALLEN    the test ran and the structure failed
    NICHT GEPRUEFT   nobody ever ran it
    laeuft           a job for it is in the queue right now

Collapsing the last two into the second is what produced months of tables in
which "no model candidate clears all three stages" read as a result when it
meant that no model candidate had ever been tested. Every cell below says which
of the four it is, and where the number came from.

Candidates per reaction:
    ours        our broken-symmetry saddle, optimised from the RKS reference
    UMA-S/M     the model predictions
    eSEN
    NEB-TS      ORCA's saddle optimisation after a broken-symmetry NEB, which
                starts at the relaxed endpoints and never sees the reference
    CI          the climbing image only, where no saddle optimisation followed.
                A band point, not a saddle -- listed but not judged.
    tsopt-*     where a transition-state optimisation started from a model
                geometry ended up

Thresholds, stated because they are choices:
    stage 1   gradient < 0.05 eV/A stationary, < 0.15 near, above that not
              stationary. Our confirmed saddles measure 0.006 to 0.011 in ORCA.
    stage 2   exactly one mode below -20 cm-1 after projecting out translation
              and rotation
    stage 3   at least 0.10 of the mode on the four reactive atoms, and at
              least 0.05 rate of change on one reactive bond. The bond lengths
              are printed rather than thresholded: a bond already at its normal
              value means the reaction is over at that point, and calibrating a
              cutoff for that on the two cases it decides would be circular.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
HA_MEV = 27211.386
BOHR = 0.529177210903
CM = 5140.4871
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}
FRAC_MIN, RATE_MIN = 0.10, 0.05
OUT = f'{H}/status_matrix.md'


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    if len(A) != len(B):
        return float('nan')
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def read_orca_hess(path):
    lines = open(path).read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.strip() == '$hessian')
    n = int(lines[i + 1].split()[0])
    Hm = np.zeros((n, n))
    k, cols = i + 2, []
    while True:
        t = lines[k].split()
        k += 1
        if not t:
            continue
        if all(x.lstrip('-').isdigit() for x in t) and len(t) <= 8:
            cols = [int(x) for x in t]
            continue
        r = int(t[0])
        for c, v in zip(cols, t[1:]):
            Hm[r, c] = float(v)
        if r == n - 1 and cols and cols[-1] == n - 1:
            break
    return Hm


def trans_rot(msqrt, xyz_bohr):
    nat = len(msqrt)
    w2 = msqrt ** 2
    c = xyz_bohr - (xyz_bohr * w2[:, None]).sum(0) / w2.sum()
    B = []
    for k in range(3):
        v = np.zeros((nat, 3)); v[:, k] = msqrt
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        B.append((np.cross(np.tile(e, (nat, 1)), c) * msqrt[:, None]).ravel())
    U, s, _ = np.linalg.svd(np.array(B).T, full_matrices=False)
    return U[:, s > 1e-8]


def analyse(hess, sym, xyz, pairs):
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    msqrt = np.sqrt(m)
    w = np.repeat(1.0 / msqrt, 3)
    Hm = hess * w[:, None] * w[None, :]
    P = trans_rot(msqrt, xyz / BOHR)
    Q = np.eye(len(Hm)) - P @ P.T
    ev, vec = np.linalg.eigh(Q @ Hm @ Q)
    fr = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
    n_imag = int((fr < -20).sum())
    k = int(np.argmin(ev))
    q = vec[:, k].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    bonds = [(nm, abs(float(np.dot(q[a] - q[b],
                                   (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])))),
              float(np.linalg.norm(xyz[a] - xyz[b]))) for a, b, nm in pairs]
    return {'n_imag': n_imag, 'imag': float(fr[k]),
            'frac': float((q[idx] ** 2).sum()), 'bonds': bonds,
            'maxrate': max(b[1] for b in bonds)}


# ---------------------------------------------------------------- ORCA readers
S2RE = re.compile(r'<S\*\*2>\s*:\s*([-\d.]+)')


def orca_dir(label):
    for d in (f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}'):
        if os.path.isdir(d):
            return d
    return None


def orca_state(label):
    """What ORCA has produced for this structure so far."""
    d = orca_dir(label)
    if not d:
        return None
    r = {'dir': d, 's2': None, 'e': None, 'grad': None,
         'hess': None, 'state': 'laeuft'}
    p = f'{d}/bs_sp.out'
    if os.path.exists(p):
        t = open(p, errors='replace').read()
        m = S2RE.findall(t)
        if m:
            r['s2'] = float(m[-1])
        e = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)', t)
        if e:
            r['e'] = float(e[-1])
    p = f'{d}/engrad.out'
    if os.path.exists(p):
        t = open(p, errors='replace').read()
        i = t.find('CARTESIAN GRADIENT')
        if i > 0:
            mx = 0.0
            for line in t[i:].split('\n')[3:]:
                f = line.split()
                if len(f) < 6:
                    break
                for v in f[3:6]:
                    try:
                        mx = max(mx, abs(float(v)))
                    except ValueError:
                        pass
            if mx:
                r['grad'] = mx * 51.42208
    hp = f'{d}/numfreq.hess'
    if os.path.exists(hp):
        r['hess'] = hp
        r['state'] = 'fertig'
    return r


# ---------------------------------------------------------------- inputs
res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
grp = {}
for r in TOP: grp[r] = 'high'
for r in MID: grp.setdefault(r, 'mid')
for r in LOW: grp.setdefault(r, 'low')
nf = {x['rxn']: x['nfod'] for x in res}


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def ours(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                return {'geom': f, 'e': j['e_uks_final'],
                        's2': j.get('s2_final'),
                        'origin': d.replace('bs_tsopt_', ''),
                        'status': j.get('status')}
    return None


def pyscf_freq(rx, label=None):
    if label is None:
        for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
            p = f'{H}/{fd}/{rx}/hessian.npy'
            if os.path.exists(p):
                return p
        return None
    p = f'{H}/freq_at_model/{label}/hessian.npy'
    return p if os.path.exists(p) else None


def our_gradient(rx):
    """The residual force at our own saddle, recorded by the frequency run.

    It sits in the bs_freq result as max_grad_ha_bohr and was never carried into
    any table, which is why our own structures showed an empty stage 1 while
    every model geometry had one.
    """
    for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
        p = f'{H}/{fd}/{rx}/result.json'
        if os.path.exists(p):
            v = json.load(open(p)).get('max_grad_ha_bohr')
            if v is not None:
                return float(v) * 51.42208
    return None


def stab(rx):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return {}
    return {g['source']: g for g in json.load(open(p))['geometries']}


def stage1_word(g):
    if g is None:
        return 'unbekannt'
    if g < 0.05:
        return 'stationaer'
    if g < 0.15:
        return 'nahe'
    return 'NICHT STATIONAER'


def fmt(v, n=3):
    return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f'{v:.{n}f}'


# ---------------------------------------------------------------- build
MR, SIMPLE = [], []
for rx in grp:
    s = stab(rx)
    ref = s.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        continue
    (MR if ref['ext_stable'] is False else SIMPLE).append(rx)
MR.sort(key=lambda r: -nf[r])
SIMPLE.sort(key=lambda r: -nf[r])

lines = []
W = lines.append
W('# Statusmatrix — Stufe 1, 2 und 3 für jede Struktur jeder Reaktion')
W('')
W('Erzeugt von `pipeline/status_matrix.py` aus den gespeicherten Ergebnissen.')
W('')
W('## Wie die Felder zu lesen sind')
W('')
W('| Eintrag | Bedeutung |')
W('|---|---|')
W('| `bestanden` | die Prüfung lief und die Struktur besteht |')
W('| `DURCHGEFALLEN` | die Prüfung lief und die Struktur besteht nicht |')
W('| `NICHT GEPRUEFT` | niemand hat sie je gerechnet |')
W('| `laeuft` | eine Rechnung dafür steht gerade in der Warteschlange |')
W('| `—` | die Struktur existiert nicht |')
W('')
W('**Der Unterschied zwischen `DURCHGEFALLEN` und `NICHT GEPRUEFT` ist der')
W('Grund für dieses Dokument.** Frühere Tabellen haben beides gleich')
W('dargestellt, wodurch ungeprüfte Kandidaten wie widerlegte aussahen.')
W('')
W('## Schwellen')
W('')
W('| Stufe | Kriterium | Schwelle |')
W('|---|---|---|')
W('| 1 | ist der Punkt stationär? | Gradient < 0.05 eV/Å `stationaer`, < 0.15 `nahe`, darüber `NICHT STATIONAER`. Unsere bestätigten Sattelpunkte liegen in ORCA bei 0.006–0.011 |')
W('| 2 | Sattelpunkt erster Ordnung? | genau eine Mode unter −20 cm⁻¹, nach Herausprojizieren von Translation und Rotation |')
W('| 3 | gehört er zu dieser Reaktion? | Modenanteil ≥ 0.10 auf den vier reaktiven Atomen **und** Bindungsrate ≥ 0.05 |')
W('')
W('Die **Bindungslängen** werden ausgegeben und nicht mit einer Schwelle')
W('versehen: eine reaktive Bindung, die bereits ihren normalen Wert hat, zeigt')
W('an, dass die Reaktion dort abgeschlossen ist — aber eine Schwelle dafür an')
W('den zwei Fällen zu kalibrieren, die sie entscheiden soll, wäre zirkulär.')
W('')

for title, RXS in (('Multireferenz — die 19', MR),
                   ('Einfach — die 26 (Kontrollgruppe)', SIMPLE)):
    W('---')
    W('')
    W(f'# {title}')
    W('')
    for rx in RXS:
        s = stab(rx)
        pairs = reactive(rx)
        o = ours(rx)
        ref = s.get('RKS-ref') or {}
        de_bs = ((ref.get('bs') or {}).get('de_meV'))
        W(f'## {rx}   N_FOD {nf[rx]:.3f}'
          + (f'   ΔE_BS an der Referenz {de_bs:.1f} meV' if de_bs is not None else ''))
        W('')
        if pairs:
            W(f'reaktive Bindungen: ' + ', '.join(nm for _, _, nm in pairs))
            W('')

        cands = []
        # ours
        if o:
            cands.append(('ours', o['geom'], o['e'], o['s2'], our_gradient(rx),
                          f"PySCF, {o['origin']}"
                          + (f", {o['status']}" if o.get('status') else '')))
        else:
            cands.append(('ours', None, None, None, None,
                          'kein konvergierter Sattelpunkt'))
        # models
        for m, dn in MODELS.items():
            g = f'{H}/{dn}/{rx}/transition_state.xyz'
            gg = s.get(m) or {}
            if gg.get('ext_stable') is None:
                e = gr = s2 = None
                surf = '?'
            elif gg['ext_stable']:
                e = gg.get('e_rks')
                gr = (gg.get('rks_grad') or {}).get('max_evang')
                s2, surf = 0.0, 'RKS'
            else:
                b = gg.get('bs') or {}
                e = b.get('e_uks')
                gr = (b.get('bs_grad') or {}).get('max_evang')
                s2, surf = b.get('s2'), 'BS'
            cands.append((m, g if os.path.exists(g) else None, e, s2, gr, surf))
        # NEB
        nt = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz'))
        ci = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-CI_converged.xyz'))
        if nt:
            cands.append(('NEB-TS', nt[0], None, None, None, 'ORCA BS-NEB'))
        elif ci:
            cands.append(('NEB-CI', ci[0], None, None, None,
                          'nur Bandpunkt, kein Sattelpunkt'))
        # tsopt from model
        for d in sorted(glob.glob(f'{H}/tsopt_from_model/{rx}_*/')):
            tag = os.path.basename(os.path.dirname(d))
            xs = [f for f in sorted(glob.glob(f'{d}/*.xyz'))
                  if any(p in os.path.basename(f).lower()
                         for p in ('ts', 'final', 'opt'))]
            if xs:
                j = {}
                if os.path.exists(f'{d}/result.json'):
                    j = json.load(open(f'{d}/result.json'))
                cands.append((f'tsopt {tag.split("_", 1)[1]}', xs[0],
                              j.get('e_uks_final'), j.get('s2_final'), None,
                              'TS-Opt ab Modellgeometrie'))

        W('| Kandidat | Fläche/⟨S²⟩ | **Stufe 1** Gradient | ΔE meV | **Stufe 2** ν_imag | **Stufe 3** Anteil / Raten | Bindungen Å | Quelle |')
        W('|---|---|---|---|---|---|---|---|')

        # common energy zero: the lowest energy among candidates
        es = [c[2] for c in cands if c[2] is not None]
        e0 = min(es) if es else None

        for name, geom, e, s2, gr, src in cands:
            label = (f'{rx}_{name}' if name in MODELS else
                     f'ours_{rx}' if name == 'ours' else
                     f'nebts_{rx}' if name == 'NEB-TS' else
                     f'tsopt_{rx}_{name.split()[-1]}' if name.startswith('tsopt') else
                     None)
            oc = orca_state(label) if label else None
            if oc:
                if oc['grad'] is not None:
                    gr = oc['grad']
                if oc['s2'] is not None:
                    s2 = oc['s2']
                if e is None and oc['e'] is not None:
                    e = oc['e']
            # stage 2 and 3
            hp = None
            code = ''
            if oc and oc['hess']:
                hp, code = oc['hess'], 'ORCA'
            elif name == 'ours':
                p = pyscf_freq(rx)
                if p:
                    hp, code = p, 'PySCF'
            elif name in MODELS:
                p = pyscf_freq(rx, f'{rx}_{name}')
                if p:
                    hp, code = p, 'PySCF'
            st2 = st3 = 'NICHT GEPRUEFT'
            bl = '—'
            if geom is None:
                st2 = st3 = '—'
            elif hp:
                try:
                    sym, xyz = read_xyz(geom)
                    hess = read_orca_hess(hp) if code == 'ORCA' else np.load(hp)
                    a = analyse(hess, sym, xyz, pairs) if pairs else None
                    if a:
                        st2 = (f'**bestanden** 1 @ {a["imag"]:.0f}'
                               if a['n_imag'] == 1
                               else f'**DURCHGEFALLEN** {a["n_imag"]}'
                               + (f' @ {a["imag"]:.0f}' if a['n_imag'] else ''))
                        ok3 = a['frac'] >= FRAC_MIN and a['maxrate'] >= RATE_MIN
                        st3 = (f'{"**bestanden**" if ok3 else "**DURCHGEFALLEN**"} '
                               f'{a["frac"]:.2f} / '
                               + ' '.join(f'{b[1]:.3f}' for b in a['bonds']))
                        bl = ' '.join(f'{b[2]:.2f}' for b in a['bonds'])
                        code += ''
                except Exception as exc:
                    st2 = st3 = f'Fehler {type(exc).__name__}'
            elif oc and oc['state'] == 'laeuft':
                st2 = st3 = 'laeuft'
            if geom is not None and bl == '—' and pairs:
                try:
                    _, xyz = read_xyz(geom)
                    bl = ' '.join(f'{np.linalg.norm(xyz[a] - xyz[b]):.2f}'
                                  for a, b, _ in pairs)
                except Exception:
                    pass
            de = f'{(e - e0) * HA_MEV:+.0f}' if (e is not None and e0 is not None) else '—'
            s1 = f'{fmt(gr)} {stage1_word(gr)}' if gr is not None else \
                 ('laeuft' if oc and oc['state'] == 'laeuft' else 'NICHT GEPRUEFT')
            W(f'| {name} | {src} / {fmt(s2, 3)} | {s1} | {de} | {st2} | {st3} '
              f'| {bl} | {code or src} |')
        W('')

open(OUT, 'w').write('\n'.join(lines) + '\n')
print(f'{len(lines)} lines -> {OUT}')
print(f'{len(MR)} multireference, {len(SIMPLE)} single-reference reactions')

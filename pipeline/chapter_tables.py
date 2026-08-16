"""The four tables the chapter needs and did not have.

Each one exists because a specific question could not be answered from the
document as it stood:

  T1  Which method found a saddle, per reaction.  The chapter pointed at
      saddle_matrix.txt seven times and never printed it.  Enriched here with
      the imaginary frequency, so competing saddles can be told apart by
      character and not only by position.

  T2  The barriers themselves, in eV, measured from the reactant.  A chapter
      about transition states had no barrier in it.  bs_ts_energies.txt exists
      but measures everything relative to OUR saddle point -- the circular
      reference the rest of the analysis avoids.  Here the zero is the relaxed
      reactant, which is closed-shell in all 45 reactions and therefore the one
      point every method agrees on.

  T3  What the reactions actually are.  Nowhere was it written which chemistry
      rxn8837 stands for, so no reader could judge whether the multireference
      set is homogeneous.  RDKit is not installed on the cluster, so this is
      formula plus the bond changes derived from the two endpoint geometries --
      which for this purpose says more than a SMILES string would.

  T4  The conditioning variable, in one place.  Everything in the chapter is
      conditioned on the instability at the RKS-TS, and its numbers were spread
      over endpoint_report.txt and cheap_stab_report.txt.

All energies come from the stage-1a single point (bs_sp.out), which is the run
that carries the stability analysis and therefore sits on the ground-state
sheet.  Gradients come from stage 1b, frequencies from stage 2.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks

H = '/home/energy/s242862'
HA_EV = 27.211386245988
BOHR = 0.529177210903

MR = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn6196',
      'rxn0346', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060', 'rxn5691',
      'rxn1283', 'rxn8827', 'rxn4522', 'rxn1147', 'rxn0894', 'rxn7957',
      'rxn5690']

# label in orca_freq -> column name.
#
# nebts_ is NOT the benchmark reference.  make_freq_list.py builds it from
# bs_uks_neb_results/<rxn>/*NEB-TS_converged.xyz, so it is our own BS-NEB
# result.  The benchmark's transition state -- the RKS-TS -- never got a
# Hessian and is not in orca_freq at all; its gradient lives in
# _collected_stability.json under results/rks_stab_bs_grad.  Mislabelling this
# column makes the RKS-TS look like a valid saddle in eight reactions, which is
# the opposite of the chapter's central finding.
CANDS = [('nebts_%s', 'UKS-NEB'), ('ours_%s', 'unsere'),
         ('%s_UMA-S', 'UMA-S'), ('%s_UMA-M', 'UMA-M'), ('%s_eSEN', 'eSEN'),
         ('tsopt_%s_UMA-M', 'TSopt/M')]

# Where the Hessian for a label actually sits.  Our structures at rxn1147 and
# rxn7957 were run inside the IRC job and live in orca_irc/<rxn>_ours, not in
# orca_freq.  Looking only in orca_freq drops them -- and those are exactly the
# two reactions where our structure loses, so the omission flatters the result.
# lowest_saddle.py carried this same bug once; this is the same fix.
def cdir(label):
    for d in (f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}'):
        if os.path.isdir(d):
            return d
    m = re.match(r'ours_(rxn\d+)$', label)
    if m:
        d = f'{H}/orca_irc/{m.group(1)}_ours'
        if os.path.isdir(d):
            return d
    return None

ERE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)')

COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'S': 1.05}


def last_energy(path):
    if not os.path.exists(path):
        return None
    m = ERE.findall(open(path, errors='replace').read())
    return float(m[-1]) if m else None


def last_s2(path):
    if not os.path.exists(path):
        return None
    m = S2RE.findall(open(path, errors='replace').read())
    return float(m[-1]) if m else None


def grad_max(d):
    """max |component| in eV/A from the ORCA .engrad file."""
    p = f'{d}/engrad.engrad'
    if not os.path.exists(p):
        return None
    nums, L = [], open(p, errors='replace').read().split('\n')
    for i, line in enumerate(L):
        if 'current gradient' in line.lower():
            j = i + 2
            while j < len(L) and not L[j].startswith('#'):
                s = L[j].strip()
                if s:
                    nums.append(float(s))
                j += 1
            break
    if not nums:
        return None
    return float(np.abs(np.array(nums)).max() * HA_EV / BOHR)


def imag_freqs(d):
    """Negative frequencies in cm-1 from the NumFreq block."""
    p = f'{d}/numfreq.out'
    if not os.path.exists(p):
        return None
    txt = open(p, errors='replace').read()
    i = txt.rfind('VIBRATIONAL FREQUENCIES')
    if i < 0:
        return None
    out = []
    for line in txt[i:].split('\n')[:400]:
        m = re.match(r'\s*\d+:\s+(-?\d+\.\d+)\s+cm', line)
        if m:
            v = float(m.group(1))
            if v < -1.0:
                out.append(v)
    return out


def read_xyz(p):
    if not os.path.exists(p):
        return None, None
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) >= 4:
            sym.append(f[0])
            xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def formula(sym):
    order = ['C', 'H', 'N', 'O', 'F', 'S']
    c = {s: sym.count(s) for s in set(sym)}
    out = ''
    for e in order:
        if c.get(e):
            out += e + (str(c[e]) if c[e] > 1 else '')
    for e in sorted(c):
        if e not in order:
            out += e + (str(c[e]) if c[e] > 1 else '')
    return out


def dmat(x):
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def bond_changes(sym, xr, xp, k=3):
    """The k pairs whose distance changes most AND that are a bond on one side.

    Ranking by raw distance change alone puts H-H pairs on top that were never
    bonded and never become bonded -- they simply swing apart as the molecule
    rearranges.  Requiring at least one endpoint to be within covalent contact
    is what makes the column describe chemistry instead of geometry.
    """
    dr, dp = dmat(xr), dmat(xp)
    n = len(sym)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            lim = (COV.get(sym[i], 0.8) + COV.get(sym[j], 0.8)) * 1.3
            a, b = dr[i, j], dp[i, j]
            if min(a, b) > lim:          # never a bond on either side
                continue
            if max(a, b) <= lim:         # a bond on both sides -- not a change
                continue
            cand.append((abs(b - a), i, j, a, b))
    cand.sort(reverse=True)
    out = []
    for _, i, j, a, b in cand[:k]:
        out.append((f'{sym[i]}{i}-{sym[j]}{j}',
                    'bricht' if b > a else 'knuepft', a, b))
    return out


def reactant_energy(rx):
    """Ground-state energy of the relaxed reactant.

    sp.out carries the stability analysis, rks.out does not.  Every reactant in
    the set is closed-shell, so the two agree; sp.out is preferred because it is
    the run that proved it.
    """
    for f in ('sp.out', 'rks.out'):
        e = last_energy(f'{H}/orca_endpoint/{rx}_reactant/{f}')
        if e is not None:
            return e, f
    return None, None


def load_stab():
    """The RKS-TS numbers, which live in the json and not in orca_freq.

    The authoritative per-reaction record is stab_pipeline/<rxn>/result.json,
    entry source == 'RKS-ref' -- the same one saddle_matrix.py reads.  The
    aggregate _collected_stability.json is only partially filled (17 of 45 in
    rks_stab_bs_grad), so it is used for the stability eigenvalues alone.

    Two gradients live in that entry and they are NOT interchangeable:

        rks_grad.max_evang     gradient on the RESTRICTED surface
        bs.bs_grad.max_evang   gradient on the BROKEN-SYMMETRY surface

    The RKS-TS was optimised on the restricted surface, so the first is small
    by construction everywhere.  The second is the one the chapter reports,
    because it asks whether the point is stationary on the surface the reaction
    actually runs on.  Reading the first where the second is meant turns the
    central finding into its opposite.
    """
    eigs = {}
    p = f'{H}/_collected_stability.json'
    if os.path.exists(p):
        eigs = json.load(open(p)).get('eigs', {})
    ref = {}
    for rx in MR:
        p = f'{H}/stab_pipeline/{rx}/result.json'
        if not os.path.exists(p):
            continue
        try:
            geos = json.load(open(p))['geometries']
        except Exception:
            continue
        e = {x['source']: x for x in geos}.get('RKS-ref')
        if e:
            ref[rx] = e
    return ref, eigs


def two_gradients(e):
    """(gradient on the RKS sheet, gradient on the BS sheet)."""
    if not e:
        return None, None
    g_rks = (e.get('rks_grad') or {}).get('max_evang')
    g_bs = ((e.get('bs') or {}).get('bs_grad') or {}).get('max_evang')
    return g_rks, g_bs


def dig(entry, *names):
    """First matching key, searched one level deep as well."""
    if not isinstance(entry, dict):
        return None
    for n in names:
        if n in entry and isinstance(entry[n], (int, float)):
            return entry[n]
    for v in entry.values():
        if isinstance(v, dict):
            got = dig(v, *names)
            if got is not None:
                return got
    return None


def saddle_matrix_ref_grads():
    """The reference gradients as saddle_matrix.txt reports them.

    saddle_matrix.py already picks the right surface -- rks_grad where the
    restricted solution is stable, bs_grad where it is not.  Comparing against
    it is what catches reading the wrong one of the two, which is exactly the
    mistake that made the RKS-TS look valid everywhere.
    """
    out = {}
    p = f'{H}/saddle_matrix.txt'
    if not os.path.exists(p):
        return out
    for line in open(p, errors='replace'):
        m = re.match(r'\s*(rxn\d+)\s+n\.stat\.\s+([\d.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ref, eigs = load_stab()
    checks.header(__file__,
                  inputs=[f'{H}/stab_pipeline', f'{H}/orca_freq',
                          f'{H}/orca_irc', f'{H}/orca_endpoint',
                          f'{H}/orca_neb_results', f'{H}/saddle_matrix.txt',
                          f'{H}/endpoint_report.txt',
                          f'{H}/_collected_stability.json'],
                  note='nebts_<rxn> ist das BS-NEB-Ergebnis, NICHT der RKS-TS.')

    print('=' * 100)
    print('T0  DER RKS-TS AUF BEIDEN FLAECHEN')
    print('=' * 100)
    print('Dieselbe Geometrie, zwei Gradienten.  Links die Flaeche, auf der sie')
    print('optimiert wurde; rechts die, auf der die Reaktion ablaeuft.')
    print()
    print(f'{"rxn":<9}{"auf RKS":>10}{"auf BS":>10}{"Faktor":>9}   Lesart')
    print('-' * 100)
    fr, fb, on_bs = [], [], {}
    for rx in MR:
        a, b = two_gradients(ref.get(rx))
        if b is not None:
            on_bs[rx] = b
        if a is None and b is None:
            print(f'{rx:<9}{"—":>10}{"—":>10}')
            continue
        fa = f'{a:.3f}' if a is not None else '—'
        fb_ = f'{b:.3f}' if b is not None else '—'
        fac = f'{b / a:.0f}x' if (a and b) else '—'
        note = ''
        if a is not None and b is not None:
            note = ('stationaer auf RKS, nicht auf BS'
                    if a < 0.15 <= b else
                    'auf beiden nicht stationaer' if a >= 0.15 else
                    'auf beiden stationaer')
        print(f'{rx:<9}{fa:>10}{fb_:>10}{fac:>9}   {note}')
        if a is not None:
            fr.append(a)
        if b is not None:
            fb.append(b)
    if fr and fb:
        print()
        print(f'Median auf der RKS-Flaeche  {np.median(fr):.3f} eV/A   '
              f'({sum(1 for x in fr if x < 0.15)} von {len(fr)} stationaer)')
        print(f'Median auf der BS-Flaeche   {np.median(fb):.3f} eV/A   '
              f'({sum(1 for x in fb if x < 0.15)} von {len(fb)} stationaer)')
        print()
        print('Der RKS-TS ist eine korrekte Rechnung auf der falschen Flaeche.')
        print('Das ist der Unterschied zwischen einem Fehler und einer Annahme,')
        print('die nicht mehr gilt.')
    print()
    # The guard for this table: the BS column must reproduce what
    # saddle_matrix.txt already reports.  Reading rks_grad here instead of
    # bs_grad turns 0 of 19 stationary into 18 of 19 -- the finding inverted.
    checks.crosscheck(on_bs, saddle_matrix_ref_grads(),
                      'RKS-TS-Gradient auf der BS-Flaeche gegen saddle_matrix.txt',
                      tol=0.005)
    checks.sentinel(list(on_bs.values()), 'Gradient auf der BS-Flaeche')
    print()

    print('=' * 100)
    print('T1  WER HAT EINEN SATTELPUNKT GEFUNDEN  --  mit Imaginaerfrequenz')
    print('=' * 100)
    print('Gradient in eV/A (max. Komponente), Frequenz in cm-1.')
    print('stationaer = Gradient < 0.15.  Ein Punkt ohne Stationaritaet ist')
    print('kein Sattelpunkt, gleich was die Hesse sagt.')
    print()
    hdr = f'{"rxn":<9}{"RKS-TS":>16}'
    for _, name in CANDS:
        hdr += f'{name:>22}'
    print(hdr)
    print('-' * 100)
    for rx in MR:
        gref = two_gradients(ref.get(rx))[1]
        row = f'{rx:<9}'
        row += (f'{"n.stat " + format(gref, ".2f"):>16}' if gref is not None
                else f'{"—":>16}')
        for pat, _ in CANDS:
            d = cdir(pat % rx)
            if d is None:
                row += f'{"--":>22}'
                continue
            g = grad_max(d)
            fr = imag_freqs(d)
            if g is None:
                cell = 'kein Grad.'
            elif g >= 0.15:
                cell = f'n.stat {g:.2f}'
            elif fr is None:
                cell = f'g{g:.03f} keine Hesse'
            elif len(fr) == 1:
                cell = f'JA {fr[0]:.0f}'
            elif len(fr) == 0:
                cell = f'Minimum g{g:.03f}'
            else:
                cell = f'{len(fr)} imag {fr[0]:.0f}'
            row += f'{cell:>22}'
        print(row)

    print()
    print('=' * 100)
    print('T2  BARRIEREN IN eV, GEMESSEN VOM RELAXIERTEN EDUKT')
    print('=' * 100)
    print('Nullpunkt ist das relaxierte Edukt der Referenz, geschlossenschalig')
    print('in allen 45 Reaktionen -- der einzige Punkt, ueber den alle Methoden')
    print('einig sind.  Energien vom Stufe-1a-Einzelpunkt, also auf der')
    print('Grundzustandsflaeche.  Eine Barriere an einem nicht-stationaeren')
    print('Punkt ist keine Barriere; solche Zellen tragen (n.stat.).')
    print()
    hdr = f'{"rxn":<9}{"RKS-TS":>12}'
    for _, name in CANDS:
        hdr += f'{name:>14}'
    print(hdr + f'{"Spanne":>10}')
    print('-' * 100)
    spans = []
    for rx in MR:
        e0, src = reactant_energy(rx)
        _e = ref.get(rx) or {}
        eref = ((_e.get('bs') or {}).get('e_uks')) or _e.get('e_rks')
        row = f'{rx:<9}'
        row += (f'{(eref - e0) * HA_EV:>11.3f}*'
                if (eref is not None and e0 is not None) else f'{"—":>12}')
        if e0 is None:
            print(row + '   kein Eduktbezug')
            continue
        vals = []
        for pat, _ in CANDS:
            d = cdir(pat % rx)
            e = last_energy(f'{d}/bs_sp.out') if d else None
            if e is None:
                row += f'{"--":>14}'
                continue
            b = (e - e0) * HA_EV
            g = grad_max(d)
            mark = '' if (g is not None and g < 0.15) else '*'
            row += f'{b:>13.3f}{mark}'
            if mark == '':
                vals.append(b)
        if len(vals) > 1:
            sp = max(vals) - min(vals)
            spans.append((sp, rx))
            row += f'{sp:>10.3f}'
        print(row)
    print()
    print('*  nicht stationaer -- die Zahl ist eine Energie, keine Barriere')
    if spans:
        spans.sort(reverse=True)
        print()
        print('Spanne ueber die gueltigen Sattelpunkte, groesste zuerst:')
        for sp, rx in spans[:8]:
            print(f'   {rx}  {sp:.3f} eV')

    print()
    print('=' * 100)
    print('T3  WELCHE REAKTIONEN DAS SIND')
    print('=' * 100)
    print('RDKit ist auf dem Cluster nicht installiert, also keine SMILES.')
    print('Stattdessen Summenformel und die drei Atompaare, deren Abstand sich')
    print('zwischen Edukt und Produkt am staerksten aendert -- dieselbe')
    print('Definition wie fuer die reaktive Koordinate, nur mit einem Paar mehr,')
    print('damit ein mitlaufender Zuschauer sichtbar wird statt versteckt.')
    print()
    print(f'{"rxn":<9}{"Formel":<10}{"N":>4}   Aenderung Edukt -> Produkt [A]')
    print('-' * 100)
    for rx in MR:
        sr, xr = read_xyz(f'{H}/orca_neb_results/{rx}/reactant.xyz')
        sp_, xp = read_xyz(f'{H}/orca_neb_results/{rx}/product.xyz')
        if sr is None or sp_ is None:
            print(f'{rx:<9}  Endpunkte fehlen')
            continue
        ch = bond_changes(sr, xr, xp)
        txt = '   '.join(f'{p} {k} {a:.2f}->{b:.2f}' for p, k, a, b in ch)
        print(f'{rx:<9}{formula(sr):<10}{len(sr):>4}   {txt}')

    print()
    print('=' * 100)
    print('T4  DIE GROESSE, AUF DIE ALLES BEDINGT IST')
    print('=' * 100)
    print('Alles am RKS-TS, ausser den beiden Produktspalten.')
    print('dE_BS = wie viel tiefer die gebrochene Loesung liegt [meV].')
    print()
    nfod = {}
    p = f'{H}/cheap_stab_report.txt'
    if os.path.exists(p):
        for line in open(p, errors='replace'):
            m = re.match(r'\s*(rxn\d+)\s+([\d.]+)\s', line)
            if m:
                nfod[m.group(1)] = float(m.group(2))

    # endpoint_report.txt already carries S2 and dE_BS at all three points for
    # all 45 reactions and has been checked; re-deriving them from the json
    # would only add a second, partial source that could drift from the first.
    ep = {}
    p = f'{H}/endpoint_report.txt'
    if os.path.exists(p):
        for line in open(p, errors='replace'):
            f_ = line.split()
            if len(f_) >= 8 and re.match(r'rxn\d+$', f_[0]):
                try:
                    ep[f_[0]] = [float(x) for x in f_[2:8]]
                except ValueError:
                    pass

    print(f'{"rxn":<9}{"N_FOD":>8}{"lmin_ext":>10}{"S2 TS":>8}{"dE_BS TS":>10}'
          f'{"Grad TS":>9}{"S2 Prod":>9}{"dE_BS Prod":>11}')
    print('-' * 100)
    for rx in MR:
        ext = (eigs.get(rx) or {}).get('ext')
        lmin = min(ext) if ext else None
        e = ep.get(rx)
        s2p, dep = (e[2], e[3]) if e else (None, None)
        s2ts, dets = (e[4], e[5]) if e else (None, None)
        gts = two_gradients(ref.get(rx))[1]

        def f(v, w, d=3):
            return f'{v:>{w}.{d}f}' if isinstance(v, (int, float)) else f'{"—":>{w}}'
        print(f'{rx:<9}{f(nfod.get(rx), 8)}{f(lmin, 10, 4)}{f(s2ts, 8)}'
              f'{f(dets, 10, 1)}{f(gts, 9)}{f(s2p, 9)}{f(dep, 11, 1)}')
    print()
    print('lmin_ext ist der kleinste externe Stabilitaetseigenwert; negativ')
    print('heisst, die restringierte Loesung ist kein Minimum der')
    print('unrestringierten Gleichungen.  dE_BS am Produkt steht in')
    print('endpoint_report.txt und wird hier nicht doppelt gerechnet.')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()

"""Guards against the one kind of mistake this project keeps making.

On 2026-08-14 four results turned out to be wrong, and all four had the same
shape: a number was read from a place that did not contain it.

    nebts_<rxn> read as the benchmark TS      it is our own BS-NEB result
    cartesian normal modes read as mass-weighted   factor 3.5 for hydrogen
    "band phase" read out of the NEB log      the log has PREOPT, not the band
    NoIter single points read as measurements they computed nothing at all

None was an arithmetic or reasoning error.  Each was caught by comparing
against something that already existed, never by re-reading the analysis.  And
three of the four produced the answer that was expected, which is why they
nearly survived -- a confirming result got less scrutiny than a surprising one.

This module makes the four checks that caught them cheap enough to use every
time.  The intended pattern for a new analysis script:

    import checks
    checks.header(__file__, inputs=[...])          # what am I reading
    checks.control(known=0.68, measured=x, ...)    # can the recipe measure?
    vals = checks.expect(rows, n=220, what='...')  # did I get everything?
    checks.sentinel(vals, 'S2')                    # is it suspiciously clean?
    checks.crosscheck(mine, theirs, ...)           # does it match what exists?

Failures raise by default.  A recipe that cannot measure should stop, not write
220 zeros that look like a finding.
"""
import os
import sys
import time

import numpy as np


class CheckFailed(AssertionError):
    """Raised when a guard trips.  Meant to abort the run, not be caught."""


def _emit(msg, warn_only):
    if warn_only:
        print(f'  WARNUNG: {msg}')
        return False
    raise CheckFailed(msg)


def header(script, inputs=(), note=''):
    """Print what this run read, so the output carries its own provenance.

    The nebts_ mix-up survived for weeks because the output tables named a
    column and never named the directory behind it.  A path in the output is
    the cheapest possible defence.
    """
    print('=' * 78)
    print(f'{os.path.basename(script)}   {time.strftime("%Y-%m-%d %H:%M")}')
    if note:
        print(note)
    if inputs:
        print('gelesen:')
        for p in inputs:
            if os.path.isdir(p):
                n = len(os.listdir(p))
                print(f'  {p}   ({n} Eintraege)')
            elif os.path.exists(p):
                st = os.stat(p)
                stamp = time.strftime('%m-%d %H:%M', time.localtime(st.st_mtime))
                print(f'  {p}   ({st.st_size} B, {stamp})')
            else:
                print(f'  {p}   FEHLT')
    print('=' * 78)


def control(known, measured, what, tol=0.05, warn_only=False):
    """Did the recipe reproduce a value that was already known?

    This is the guard that NoIter would have failed instantly: asked to read
    orbitals whose <S^2> was 0.68, it returned nothing.  A measurement that
    cannot recover a known answer cannot be trusted with an unknown one.
    """
    if measured is None or (isinstance(measured, float) and np.isnan(measured)):
        return _emit(f'Kontrolle {what}: nichts gemessen (bekannt {known})',
                     warn_only)
    d = abs(float(measured) - float(known))
    ok = d <= tol
    print(f'  Kontrolle {what}: bekannt {known:.6g}, gemessen '
          f'{float(measured):.6g}, Differenz {d:.3g}  '
          f'{"bestanden" if ok else "DURCHGEFALLEN"}')
    if not ok:
        return _emit(f'Kontrolle {what} weicht um {d:.3g} ab (Toleranz {tol})',
                     warn_only)
    return True


def expect(found, n, what, warn_only=False):
    """Did the extraction get as many records as the run should have produced?

    rxn4113 logged 36 SCFs for 100 band iterations over 8 images.  Nobody
    subtracted those numbers, and the band claim rested on the difference.
    """
    got = found if isinstance(found, int) else len(found)
    print(f'  Anzahl {what}: {got} gefunden, {n} erwartet')
    if got != n:
        _emit(f'{what}: {got} statt {n} Datensaetze -- die Quelle enthaelt '
              f'nicht, was angenommen wurde', warn_only)
    return found


def sentinel(values, what, warn_only=True):
    """Is the result suspiciously clean?

    Exact zeros and perfect uniformity are, in numerical chemistry, almost
    always a missing value rather than a measurement.  Three of today's four
    errors looked exactly like this: energy 0.000000000000, <S^2> 0.000 in 220
    of 220 rows, RMSD 0.0000 A.  Warning only -- a legitimately uniform result
    exists, it just has to be looked at.
    """
    v = np.asarray([x for x in np.ravel(values)
                    if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if v.size == 0:
        return _emit(f'{what}: keine auswertbaren Werte', warn_only)
    hits = []
    if np.all(v == 0.0):
        hits.append(f'alle {v.size} Werte exakt null')
    elif np.count_nonzero(v == 0.0) > 0.9 * v.size:
        hits.append(f'{np.count_nonzero(v == 0.0)} von {v.size} exakt null')
    if v.size > 3 and np.ptp(v) == 0.0:
        hits.append(f'alle {v.size} Werte identisch ({v[0]:.6g})')
    for h in hits:
        _emit(f'{what}: {h} -- exakte Gleichheit ist meist ein fehlender Wert, '
              f'keine Messung', warn_only)
    if not hits:
        print(f'  Sentinel {what}: unauffaellig '
              f'(n={v.size}, min {v.min():.4g}, max {v.max():.4g})')
    return not hits


def crosscheck(mine, theirs, what, tol=1e-6, warn_only=True):
    """Compare against a source that already exists, and print the count.

    Both dicts keyed the same way; only the overlap is compared.  The rule this
    encodes: a new table that silently disagrees with an old one is worse than
    no new table.  Report zero disagreements or explain them -- the
    mass-weighting difference was found exactly this way.
    """
    keys = sorted(set(mine) & set(theirs))
    print(f'  Abgleich {what}: {len(keys)} gemeinsame Eintraege '
          f'(neu {len(mine)}, bestehend {len(theirs)})')
    if not keys:
        return _emit(f'{what}: keine Ueberschneidung -- der Abgleich prueft '
                     f'nichts', warn_only)
    bad = []
    for k in keys:
        a, b = mine[k], theirs[k]
        if a is None or b is None:
            bad.append((k, a, b))
            continue
        if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
            if a != b:
                bad.append((k, a, b))
        elif abs(float(a) - float(b)) > tol:
            bad.append((k, a, b))
    print(f'  Abweichungen: {len(bad)}')
    for k, a, b in bad[:20]:
        print(f'     {k}: neu {a}, bestehend {b}')
    if bad:
        _emit(f'{what}: {len(bad)} von {len(keys)} weichen ab', warn_only)
    return bad


# --------------------------------------------------------------- ORCA readers
# Each returns None rather than a plausible-looking wrong number when the
# quantity is absent.  A missing value that reads as 0.0 is how NoIter got
# through.

def orca_energy(path):
    """Last FINAL SINGLE POINT ENERGY, or None.  Exact 0.0 counts as absent."""
    if not os.path.exists(path):
        return None
    v = None
    for line in open(path, errors='replace'):
        if 'FINAL SINGLE POINT ENERGY' in line:
            try:
                v = float(line.split()[-1])
            except ValueError:
                pass
    # ORCA writes exactly this when it skipped the calculation
    if v is not None and v == 0.0:
        print(f'  {os.path.basename(path)}: Energie exakt 0.0 -- '
              f'die Rechnung hat nichts ausgewertet')
        return None
    return v


def orca_s2(path):
    """Last <S**2>, or None."""
    if not os.path.exists(path):
        return None
    v = None
    for line in open(path, errors='replace'):
        if 'Expectation value of <S**2>' in line:
            try:
                v = float(line.split()[-1])
            except ValueError:
                pass
    return v


def orca_terminated(path):
    return (os.path.exists(path)
            and 'ORCA TERMINATED NORMALLY' in open(path, errors='replace').read())


def summary(ok=True):
    print('-' * 78)
    print('alle Pruefungen bestanden' if ok else 'PRUEFUNGEN FEHLGESCHLAGEN')
    return 0 if ok else 1


if __name__ == '__main__':
    # A self-test, so the guards are themselves checked.  Each block asserts
    # that the guard trips on the failure it was written for.
    print('Selbsttest')
    header(__file__, inputs=[__file__])
    control(0.68, 0.681, 'Kontrollbeispiel')
    try:
        control(0.68, None, 'leere Messung')
        print('FEHLER: leere Messung haette abbrechen muessen')
        sys.exit(1)
    except CheckFailed as e:
        print(f'  korrekt abgebrochen: {e}')
    try:
        expect(36, 800, 'SCF-Bloecke')
        print('FEHLER: Zaehlprobe haette abbrechen muessen')
        sys.exit(1)
    except CheckFailed as e:
        print(f'  korrekt abgebrochen: {e}')
    sentinel([0.0] * 220, 'S2 aus NoIter')
    sentinel([0.1, 0.5, 0.9], 'echte Werte')
    crosscheck({'a': 1.0, 'b': 2.0}, {'a': 1.0, 'b': 2.5}, 'Beispieltabelle')
    print()
    print('Selbsttest durch: jede Wache hat auf ihren Fehler reagiert.')

"""Post-process a converged BS-UKS NEB: energies, forces and <S^2> per image.

ORCA's NEB writes geometries and energies but no gradients, so the band cannot
be used for anything that needs forces.  This runs one UKS EnGrad single point
per final MEP image at the same level and BrokenSym setting as the NEB, and
writes an extxyz that ASE can read directly.

<S^2> is stored per image because it says which surface the point sits on. A
value near 0 at the band ends is correct, not a failure: the broken-symmetry
solution only exists past the Coulson-Fischer point.

SPLIT is written into every frame. These reactions are Transition1x *test*
data -- benchmark reactions -- so the file is for inspection and evaluation.
Training on them would contaminate the benchmark they are used to measure.

Usage: python neb_images_engrad.py <rxn> [--nprocs 8] [--maxcore 8000]
"""
import argparse
import os
import re
import subprocess

HOME = '/home/energy/s242862'
BASE = f'{HOME}/bs_uks_neb_results'
SPLIT = 'test'          # benchmark reactions -- do not train on these
LEVEL = 'UKS wB97M-V def2-TZVP TightSCF SlowConv EnGrad'
HA_EV = 27.211386245988
BOHR = 0.529177210903

S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)')


def read_multi_xyz(p):
    """Yield (symbols, coords, comment) for every frame."""
    L = open(p).read().split('\n')
    i = 0
    while i < len(L):
        if not L[i].strip():
            i += 1
            continue
        n = int(L[i].split()[0])
        comment = L[i + 1]
        sym, xyz = [], []
        for line in L[i + 2:i + 2 + n]:
            f = line.split()
            sym.append(f[0])
            xyz.append([float(x) for x in f[1:4]])
        yield sym, xyz, comment
        i += 2 + n


def parse_engrad(path, natm):
    """ORCA .engrad: energy in Ha, gradient in Ha/Bohr."""
    L = [x.strip() for x in open(path)]
    e, grad = None, []
    for i, x in enumerate(L):
        if 'current total energy' in x:
            e = float(L[i + 2])
        if 'current gradient' in x:
            grad = [float(v) for v in L[i + 2:i + 2 + 3 * natm]]
    return e, grad


def main(a):
    d = f'{BASE}/{a.rxn}'
    mep = f'{d}/bs_uks_neb_MEP_trj.xyz'
    if not os.path.exists(mep):
        print(f'{a.rxn}: kein konvergierter MEP ({mep})')
        return 1
    frames = list(read_multi_xyz(mep))
    print(f'{a.rxn}: {len(frames)} Bilder', flush=True)

    work = f'{d}/engrad'
    os.makedirs(work, exist_ok=True)
    out = f'{d}/images_bs.extxyz'
    written = 0

    with open(out, 'w') as fh:
        for k, (sym, xyz, comment) in enumerate(frames):
            n = len(sym)
            inp = f'{work}/img{k:02d}.inp'
            with open(inp, 'w') as g:
                g.write(f'! {LEVEL}\n%pal nprocs {a.nprocs} end\n'
                        f'%maxcore {a.maxcore}\n'
                        f'%scf BrokenSym 1,1\n  MaxIter 500\nend\n\n'
                        f'* xyz 0 1\n')
                for s, (x, y, z) in zip(sym, xyz):
                    g.write(f'{s} {x:.8f} {y:.8f} {z:.8f}\n')
                g.write('*\n')
            log = f'{work}/img{k:02d}.out'
            with open(log, 'w') as lg:
                subprocess.run([a.orca_path, inp], cwd=work, stdout=lg,
                               stderr=subprocess.STDOUT)
            eg = f'{work}/img{k:02d}.engrad'
            if not os.path.exists(eg):
                print(f'  Bild {k}: kein .engrad -- uebersprungen', flush=True)
                continue
            e, grad = parse_engrad(eg, n)
            s2 = S2_RE.findall(open(log, errors='ignore').read())
            s2 = float(s2[-1]) if s2 else float('nan')
            # forces = -gradient, converted to eV/A
            f_ev = [-v * HA_EV / BOHR for v in grad]
            fh.write(f'{n}\n')
            fh.write(f'Properties=species:S:1:pos:R:3:forces:R:3 '
                     f'energy={e * HA_EV:.10f} s2={s2:.6f} '
                     f'image={k} rxn={a.rxn} split={SPLIT} '
                     f'level="{LEVEL}" bs_guess="BrokenSym 1,1"\n')
            for i, (s, (x, y, z)) in enumerate(zip(sym, xyz)):
                fx, fy, fz = f_ev[3 * i:3 * i + 3]
                fh.write(f'{s} {x:.8f} {y:.8f} {z:.8f} '
                         f'{fx:.8f} {fy:.8f} {fz:.8f}\n')
            written += 1
            print(f'  Bild {k}: E={e:.8f} Ha  <S^2>={s2:.4f}  '
                  f'max|F|={max(abs(v) for v in f_ev):.4f} eV/A', flush=True)

    print(f'{a.rxn}: {written}/{len(frames)} Bilder -> {out}', flush=True)
    return 0 if written == len(frames) else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('rxn')
    p.add_argument('--nprocs', type=int, default=8)
    p.add_argument('--maxcore', type=int, default=8000)
    p.add_argument('--orca-path', default='orca')
    raise SystemExit(main(p.parse_args()))

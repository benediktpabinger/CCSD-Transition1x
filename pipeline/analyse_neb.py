"""
Comprehensive analysis of ORCA NEB results vs Transition1x reference.

1. Failure analysis  — classify remaining unconverged reactions
2. Barrier comparison — ORCA (wB97M-V/def2-TZVP) vs T1x (wB97x/6-31G(d))
3. TS geometry comparison — RMSD between ORCA and T1x TS geometries

Usage:
    python analyse_neb.py \
        --h5file  ~/data/Transition1x.h5 \
        --test-dir ~/orca_neb_results \
        --val-dir  ~/orca_neb_val_results \
        --test-list ~/ccsd_dataset/test_reactions.txt \
        --val-list  ~/ccsd_dataset/val_reactions.txt \
        --output    ~/neb_analysis
"""
import argparse, json, os
import numpy as np
import h5py
from ase.io import read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def last_fmax(neb_log):
    """Return last force value from neb.log, or None."""
    if not os.path.exists(neb_log) or os.path.getsize(neb_log) == 0:
        return None
    val = None
    with open(neb_log) as f:
        for line in f:
            if 'NEBOptimizer' in line:
                try:
                    val = float(line.split()[-1])
                except ValueError:
                    pass
    return val

def neb_steps(neb_log):
    if not os.path.exists(neb_log):
        return 0
    n = 0
    with open(neb_log) as f:
        for line in f:
            if 'NEBOptimizer' in line:
                n += 1
    return n

def orca_barrier(rxn_dir):
    """E_TS - E_reactant in eV from xyz comment lines."""
    ts_f = os.path.join(rxn_dir, 'transition_state.xyz')
    r_f  = os.path.join(rxn_dir, 'reactant.xyz')
    if not os.path.exists(ts_f) or not os.path.exists(r_f):
        return None
    def read_energy(path):
        with open(path) as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        for tok in lines[1].split():
            if tok.startswith('energy='):
                try:
                    return float(tok.split('=')[1])
                except ValueError:
                    pass
        # try ASE
        try:
            a = read(path)
            return a.get_potential_energy()
        except Exception:
            return None
    e_ts = read_energy(ts_f)
    e_r  = read_energy(r_f)
    if e_ts is None or e_r is None:
        return None
    return e_ts - e_r

def t1x_barrier(h5file, split, rxn_id):
    """E_TS - E_reactant from T1x H5 optimised subgroups."""
    with h5py.File(h5file, 'r') as f:
        grp = f[split]
        for formula in grp:
            if rxn_id in grp[formula]:
                rxn = grp[formula][rxn_id]
                try:
                    e_ts = float(rxn['transition_state']['wB97x_6-31G(d).energy'][0])
                    e_r  = float(rxn['reactant']['wB97x_6-31G(d).energy'][0])
                    return e_ts - e_r
                except KeyError:
                    return None
    return None

def t1x_ts_positions(h5file, split, rxn_id):
    """Return (positions, atomic_numbers) for T1x TS."""
    with h5py.File(h5file, 'r') as f:
        grp = f[split]
        for formula in grp:
            if rxn_id in grp[formula]:
                rxn = grp[formula][rxn_id]
                try:
                    pos = rxn['transition_state']['positions'][:]
                    zs  = rxn['transition_state']['atomic_numbers'][:]
                    return pos, zs
                except KeyError:
                    return None, None
    return None, None

def kabsch_rmsd(P, Q):
    """RMSD after optimal rotation (Kabsch algorithm). P, Q: (N,3) arrays."""
    P = np.array(P, dtype=float).reshape(-1, 3)
    Q = np.array(Q, dtype=float).reshape(-1, 3)
    if len(P) != len(Q) or len(P) == 0:
        return None
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    H  = P.T @ Q                       # (3,3)
    U, S, Vt = np.linalg.svd(H)       # U(3,3), Vt(3,3)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1.0, 1.0, d])
    R  = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_rot - Q)**2, axis=1))))

def ts_rmsd(rxn_dir, h5file, split, rxn_id):
    """RMSD between ORCA NEB TS and T1x TS (same atom ordering assumed)."""
    ts_f = os.path.join(rxn_dir, 'transition_state.xyz')
    if not os.path.exists(ts_f):
        return None
    orca_atoms = read(ts_f)
    t1x_pos, t1x_zs = t1x_ts_positions(h5file, split, rxn_id)
    if t1x_pos is None:
        return None
    if len(orca_atoms) != len(t1x_zs):
        return None  # different atom count — shouldn't happen
    return kabsch_rmsd(orca_atoms.get_positions(), t1x_pos)


# ── collection ────────────────────────────────────────────────────────────────

def collect_split(neb_dir, rxn_list, h5file, split):
    results = []
    with open(rxn_list) as f:
        rxns = [l.strip() for l in f if l.strip()]

    for rxn in rxns:
        d = os.path.join(neb_dir, rxn)
        converged = os.path.exists(os.path.join(d, 'converged'))
        fmax      = last_fmax(os.path.join(d, 'neb.log'))
        steps     = neb_steps(os.path.join(d, 'neb.log'))

        b_orca = orca_barrier(d) if converged else None
        b_t1x  = t1x_barrier(h5file, split, rxn)
        rmsd   = ts_rmsd(d, h5file, split, rxn) if converged else None

        results.append({
            'rxn': rxn, 'split': split,
            'converged': converged,
            'steps': steps,
            'last_fmax': fmax,
            'b_orca': b_orca,
            'b_t1x': b_t1x,
            'ts_rmsd': rmsd,
        })

    return results


# ── analysis & plots ──────────────────────────────────────────────────────────

def classify_failure(r):
    fmax = r['last_fmax']
    if fmax is None:
        return 'crash'
    if fmax < 0.06:
        return 'plateau_near_thresh'
    if fmax < 0.15:
        return 'slow_converge'
    return 'far_from_converge'


def run(args):
    os.makedirs(args.output, exist_ok=True)

    print("Collecting test split...")
    test = collect_split(args.test_dir, args.test_list, args.h5file, 'test')
    print("Collecting val split...")
    val  = collect_split(args.val_dir,  args.val_list,  args.h5file, 'val')
    all_r = test + val

    # ── 1. Failure analysis ────────────────────────────────────────────────
    print("\n=== 1. FAILURE ANALYSIS ===")
    for split_name, split_data in [('test', test), ('val', val)]:
        failed = [r for r in split_data if not r['converged']]
        total  = len(split_data)
        print(f"\n{split_name}: {total - len(failed)}/{total} converged ({len(failed)} failed)")
        cats = {}
        for r in failed:
            c = classify_failure(r)
            cats[c] = cats.get(c, [])
            cats[c].append(r)
        for cat, rxns in sorted(cats.items()):
            fmaxes = [r['last_fmax'] for r in rxns if r['last_fmax']]
            t1x_bs = [r['b_t1x'] for r in rxns if r['b_t1x'] is not None]
            print(f"  {cat}: {len(rxns)} reactions")
            if fmaxes:
                print(f"    fmax range: {min(fmaxes):.3f} – {max(fmaxes):.3f} eV/A")
            if t1x_bs:
                print(f"    T1x barrier: {np.mean(t1x_bs)*1000:.0f} meV mean  "
                      f"({min(t1x_bs)*1000:.0f}–{max(t1x_bs)*1000:.0f} meV range)")
            for r in rxns:
                print(f"    {r['rxn']}: fmax={r['last_fmax']}, steps={r['steps']}, "
                      f"T1x_barrier={r['b_t1x']*1000 if r['b_t1x'] else 'N/A':.0f} meV")

    # ── 2. Barrier comparison ──────────────────────────────────────────────
    print("\n=== 2. BARRIER COMPARISON ===")
    conv = [r for r in all_r if r['converged'] and r['b_orca'] is not None and r['b_t1x'] is not None]
    b_orca = np.array([r['b_orca'] for r in conv])
    b_t1x  = np.array([r['b_t1x']  for r in conv])
    delta  = b_orca - b_t1x

    mae    = np.mean(np.abs(delta))
    rmse   = np.sqrt(np.mean(delta**2))
    bias   = np.mean(delta)
    r2     = 1 - np.sum(delta**2) / np.sum((b_orca - b_orca.mean())**2)
    pearson = np.corrcoef(b_t1x, b_orca)[0, 1]

    print(f"N converged with both barriers: {len(conv)}")
    print(f"MAE:   {mae*1000:.1f} meV  ({mae*23.06:.2f} kcal/mol)")
    print(f"RMSE:  {rmse*1000:.1f} meV  ({rmse*23.06:.2f} kcal/mol)")
    print(f"Bias (ORCA - T1x): {bias*1000:.1f} meV  ({bias*23.06:.2f} kcal/mol)")
    print(f"Pearson r: {pearson:.4f}")
    print(f"R²:        {r2:.4f}")
    print(f"T1x range:  {b_t1x.min():.2f} – {b_t1x.max():.2f} eV")
    print(f"ORCA range: {b_orca.min():.2f} – {b_orca.max():.2f} eV")

    # Outliers (|delta| > 0.5 eV)
    outliers = [(conv[i]['rxn'], delta[i], b_t1x[i], b_orca[i])
                for i in np.where(np.abs(delta) > 0.5)[0]]
    if outliers:
        print(f"\nOutliers (|delta| > 0.5 eV): {len(outliers)}")
        for rxn, d, bt, bo in sorted(outliers, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {rxn}: T1x={bt:.3f} eV, ORCA={bo:.3f} eV, delta={d*1000:.0f} meV")

    # ── 3. TS geometry comparison ──────────────────────────────────────────
    print("\n=== 3. TS GEOMETRY COMPARISON ===")
    rmsds = np.array([r['ts_rmsd'] for r in conv if r['ts_rmsd'] is not None])
    rmsd_rxns = [r for r in conv if r['ts_rmsd'] is not None]
    print(f"N with RMSD computed: {len(rmsds)}")
    if len(rmsds) == 0:
        print("No RMSD data available.")
    else:
        print(f"RMSD mean:   {rmsds.mean():.3f} A")
        print(f"RMSD median: {np.median(rmsds):.3f} A")
        print(f"RMSD max:    {rmsds.max():.3f} A  ({rmsd_rxns[np.argmax(rmsds)]['rxn']})")
        print(f"RMSD > 0.5 A: {(rmsds > 0.5).sum()} reactions")
        print(f"RMSD > 1.0 A: {(rmsds > 1.0).sum()} reactions")
        large_rmsd = [(rmsd_rxns[i]['rxn'], rmsds[i]) for i in np.argsort(rmsds)[-10:]][::-1]
        print("\nTop 10 largest RMSD:")
        for rxn, r in large_rmsd:
            print(f"  {rxn}: {r:.3f} A")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Parity plot
    ax = axes[0]
    lim = [min(b_t1x.min(), b_orca.min()) - 0.1,
           max(b_t1x.max(), b_orca.max()) + 0.1]
    sc = ax.scatter(b_t1x, b_orca, c=rmsds[:len(b_t1x)] if len(rmsds)==len(b_t1x) else 'steelblue',
                    cmap='RdYlGn_r', alpha=0.6, s=12, vmin=0, vmax=1.0)
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('wB97x/6-31G(d) barrier [eV]')
    ax.set_ylabel('wB97M-V/def2-TZVP barrier [eV]')
    ax.set_title(f'Barrier parity  N={len(conv)}\nMAE={mae*1000:.0f} meV, r={pearson:.4f}')
    try:
        plt.colorbar(sc, ax=ax, label='TS RMSD [Å]')
    except Exception:
        pass

    # Difference histogram
    ax = axes[1]
    ax.hist(delta * 1000, bins=60, color='steelblue', edgecolor='white', lw=0.3)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.axvline(bias * 1000, color='red', lw=1.5, label=f'bias={bias*1000:.0f} meV')
    ax.set_xlabel('Δ barrier [meV]  (ORCA − T1x)')
    ax.set_ylabel('Count')
    ax.set_title('Barrier difference distribution')
    ax.legend()

    # RMSD histogram
    ax = axes[2]
    ax.hist(rmsds, bins=50, color='darkorange', edgecolor='white', lw=0.3)
    ax.axvline(np.median(rmsds), color='k', lw=1.5, ls='--',
               label=f'median={np.median(rmsds):.3f} Å')
    ax.set_xlabel('TS RMSD (ORCA vs T1x) [Å]')
    ax.set_ylabel('Count')
    ax.set_title('TS geometry deviation')
    ax.legend()

    fig.tight_layout()
    out_png = os.path.join(args.output, 'neb_analysis.png')
    fig.savefig(out_png, dpi=150)
    print(f"\nPlot saved to {out_png}")

    # Save raw data
    out_json = os.path.join(args.output, 'neb_analysis.json')
    with open(out_json, 'w') as f:
        json.dump([{k: (float(v) if isinstance(v, (np.floating, float)) else
                        bool(v) if isinstance(v, (np.bool_, bool)) else v)
                    for k, v in r.items()} for r in all_r], f, indent=2)
    print(f"Raw data saved to {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',    default='/home/energy/s242862/data/Transition1x.h5')
    parser.add_argument('--test-dir',  default='/home/energy/s242862/orca_neb_results')
    parser.add_argument('--val-dir',   default='/home/energy/s242862/orca_neb_val_results')
    parser.add_argument('--test-list', default='/home/energy/s242862/ccsd_dataset/test_reactions.txt')
    parser.add_argument('--val-list',  default='/home/energy/s242862/ccsd_dataset/val_reactions.txt')
    parser.add_argument('--output',    default='/home/energy/s242862/neb_analysis')
    args = parser.parse_args()
    run(args)

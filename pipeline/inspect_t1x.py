import h5py, numpy as np

f = h5py.File("/home/energy/s242862/data/Transition1x.h5", "r")
grp = f["test"]["C2H3N3O2"]["rxn10005"]
e = grp["wB97x_6-31G(d).energy"][:]

print("Keys:", list(grp.keys()))

# Inspect the subgroups
for key in ["reactant", "product", "transition_state"]:
    obj = grp[key]
    if hasattr(obj, 'keys'):
        print(f"{key} is a group with keys: {list(obj.keys())}")
        for k in obj.keys():
            print(f"  {key}/{k}: shape={obj[k].shape}, dtype={obj[k].dtype}, value={obj[k][()]}")
    else:
        print(f"{key}: shape={obj.shape}, dtype={obj.dtype}, value={obj[()]}")

# Also check full energy profile shape and what the path looks like
print(f"\nEnergy array shape: {e.shape}")
print(f"E[0]={e[0]:.4f}, E[-1]={e[-1]:.4f}, min={e.min():.4f}, max={e.max():.4f}")
print(f"Relative energies (first 10): {np.round(e[:10]-e[0], 4)}")
print(f"Relative energies (last 5): {np.round(e[-5:]-e[0], 4)}")

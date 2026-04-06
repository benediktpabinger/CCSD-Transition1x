import torch
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsLoader

_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

# Load full Lightning task (includes model + optimizer state)
task = spk.task.AtomisticTask.load_from_checkpoint(
    "/home/energy/s242862/painn_results_v2/checkpoints/best.ckpt",
    map_location="cpu",
)
model = task.model
model.eval()

dataset = ASEAtomsData("/home/energy/s242862/data/transition1x_train.db",
    load_properties=["energy"],
    transforms=[
        trn.ASENeighborList(cutoff=5.0),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ])
# Use configs from the END of the dataset (less likely to be in training set)
n = len(dataset)
loader = AtomsLoader(torch.utils.data.Subset(dataset, list(range(n-5, n))), batch_size=5)

with torch.enable_grad():
    for batch in loader:
        pred = model(batch)
        for i in range(5):
            e_pred = pred["energy"][i].item()
            e_dft  = batch["energy"][i].item()
            diff_meV = abs(e_pred - e_dft) * 1000
            print("config %d: DFT=%.4f eV  PaiNN=%.4f eV  diff=%.1f meV" % (i, e_dft, e_pred, diff_meV))

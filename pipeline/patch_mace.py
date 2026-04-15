"""
Apply custom patches to the installed MACE package on the cluster.

Patches added:
  1. --max_samples_per_epoch  : limit training samples per epoch (RandomSampler, no replacement)
  2. --max_valid_samples      : limit validation samples per epoch (RandomSampler, no replacement)

Usage:
    python patch_mace.py

Run after every fresh MACE install / upgrade on the cluster.
"""

import importlib, pathlib, sys, re

def find_mace_file(relative):
    import mace
    base = pathlib.Path(mace.__file__).parent
    p = base / relative
    assert p.exists(), f"Not found: {p}"
    return p


# ── 1. arg_parser.py ─────────────────────────────────────────────────────────

ARG_PARSER_PATCH = '''    parser.add_argument(
        "--max_samples_per_epoch",
        help="Limit training samples per epoch for fast epoch cycling over large datasets.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_valid_samples",
        help="Limit validation samples per epoch for fast validation on large datasets.",
        type=int,
        default=None,
    )
'''

ARG_PARSER_ANCHOR = '        "--patience",'


def patch_arg_parser():
    path = find_mace_file("tools/arg_parser.py")
    text = path.read_text()

    if "--max_valid_samples" in text:
        print("arg_parser.py already fully patched.")
        return

    if "--max_samples_per_epoch" not in text:
        # Insert both args before --patience
        text = text.replace(
            '    parser.add_argument(\n' + ARG_PARSER_ANCHOR,
            ARG_PARSER_PATCH + '    parser.add_argument(\n' + ARG_PARSER_ANCHOR,
        )
    else:
        # Only max_samples_per_epoch present; add max_valid_samples after it
        insert_after = '''\
    )
    parser.add_argument(
        "--patience",'''
        replacement = '''\
    )
    parser.add_argument(
        "--max_valid_samples",
        help="Limit validation samples per epoch for fast validation on large datasets.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--patience",'''
        # Replace only first occurrence (right after max_samples block)
        text = text.replace(insert_after, replacement, 1)

    path.write_text(text)
    print(f"Patched: {path}")

    # Remove stale pyc
    for pyc in path.parent.rglob("arg_parser*.pyc"):
        pyc.unlink()
        print(f"Removed: {pyc}")


# ── 2. run_train.py ──────────────────────────────────────────────────────────

RUN_TRAIN_TRAIN_PATCH = '''\
    if getattr(args, 'max_samples_per_epoch', None) is not None and train_sampler is None:
        _epoch_sampler = torch.utils.data.RandomSampler(
            train_set,
            num_samples=min(args.max_samples_per_epoch, len(train_set)),
            replacement=False,
        )
        train_loader = torch_geometric.dataloader.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler=_epoch_sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
        )
    else:
'''

RUN_TRAIN_VALID_PATCH = '''\
    # --- max_valid_samples patch ---
    if getattr(args, 'max_valid_samples', None) is not None:
        for head_name in list(valid_loaders.keys()):
            vset = valid_sets[head_name] if isinstance(valid_sets, dict) else valid_sets
            if vset is not None and len(vset) > 0:
                _vsampler = torch.utils.data.RandomSampler(
                    vset,
                    num_samples=min(args.max_valid_samples, len(vset)),
                    replacement=False,
                )
                valid_loaders[head_name] = torch_geometric.dataloader.DataLoader(
                    dataset=vset,
                    batch_size=args.valid_batch_size,
                    sampler=_vsampler,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=args.pin_memory,
                    num_workers=args.num_workers,
                )
    # --- end patch ---
'''

# Anchor: the closing line of the original valid_loaders for-loop.
# The patch must go AFTER this loop, not before it — otherwise the loop
# overwrites the subsampled loaders we just set.
VALID_LOADER_ANCHOR = (
    "            generator=torch.Generator().manual_seed(args.seed),\n"
    "        )\n"
    "\n"
    "    loss_fn"
)


def patch_run_train():
    path = find_mace_file("cli/run_train.py")
    text = path.read_text()

    changed = False

    if "max_samples_per_epoch" not in text:
        # Insert train sampler patch before the else: branch of existing train_loader block
        anchor = "    else:\n        train_loader = torch_geometric.dataloader.DataLoader(\n            dataset=train_set,"
        assert anchor in text, "Cannot find train_loader anchor in run_train.py"
        text = text.replace(anchor, RUN_TRAIN_TRAIN_PATCH + "        train_loader = torch_geometric.dataloader.DataLoader(\n            dataset=train_set,")
        changed = True
        print("Applied max_samples_per_epoch patch to run_train.py")

    if "max_valid_samples" not in text:
        assert VALID_LOADER_ANCHOR in text, "Cannot find valid_loaders anchor in run_train.py"
        replacement = (
            "            generator=torch.Generator().manual_seed(args.seed),\n"
            "        )\n"
            "\n"
            + RUN_TRAIN_VALID_PATCH +
            "\n    loss_fn"
        )
        text = text.replace(VALID_LOADER_ANCHOR, replacement, 1)
        changed = True
        print("Applied max_valid_samples patch to run_train.py")

    if not changed:
        print("run_train.py already fully patched.")
        return

    path.write_text(text)
    print(f"Patched: {path}")

    for pyc in path.parent.rglob("run_train*.pyc"):
        pyc.unlink()
        print(f"Removed: {pyc}")


if __name__ == "__main__":
    patch_arg_parser()
    patch_run_train()
    print("Done. Verify with: mace_run_train --help | grep max_")

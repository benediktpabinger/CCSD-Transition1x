# Methods — the hinge test  ·  SUPERSEDED

**Superseded — §7 of [`methods_for_paper.md`](methods_for_paper.md) is the sole
source.**

This file documented the hinge test at the OMol25-level NEB geometries only. It
predates [`results/hinge_t1x.csv`](../results/hinge_t1x.csv) and the single
build script [`pipeline/hinge_tables.py`](../pipeline/hinge_tables.py), and it
carried breaking depths taken from `uks_sp` rather than `uks_engrad`, which made
all fifteen of them too small by about 0.7 meV. Its §3 also attributed a
multireference class to four of the twelve excluded reactions from an external
def2-TZVP analysis; those twelve have since been measured directly at the label
geometries, and three of them — not four — are unstable.

Nothing here is a source for any statement. The file is kept so the history
resolves; the content lives in §7 of `methods_for_paper.md`, with
column-by-column provenance in [`results/README.md`](../results/README.md).

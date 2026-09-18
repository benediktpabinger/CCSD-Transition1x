# Terminology and abbreviations

Working list of the wordings the thesis uses, where they are defined, and what they must not be confused with. Started 2026-09-16.

## Spin solutions and surfaces

Defined in the Background, Section "Breaking the Symmetry and Testing for It" (`sec:background-stability`):

```latex
\paragraph{Terminology.} A structure is \textbf{closed-shell} when the
stability analysis confirms the RKS solution as the lowest,
$\langle S^2 \rangle = 0$, and \textbf{broken-symmetry} when a lower UKS
solution exists, $\langle S^2 \rangle > 0$. In the first case the
restricted solution is called \textbf{stable}, in the second
\textbf{unstable}~\cite{seeger1977}.

\paragraph{Two surfaces.} Each spin treatment defines its own potential
energy surface. The \textbf{restricted surface} is the RKS energy at
every geometry. The \textbf{unrestricted surface} is the lowest UKS
solution at every geometry, found with the stability analysis: where
the restricted solution is stable the two surfaces coincide, where it
is unstable the unrestricted surface lies below. The stability analysis
is a statement about the electronic solution at one fixed geometry.
Whether that geometry is also a stationary point of the unrestricted
surface is a separate question, answered by the gradient of the
unrestricted solution at that geometry.
```

## Level, surface, protocol

- **Level of theory** = functional/basis only (Background, Section "Electronic Structure Methods"). Never includes the spin formalism.
- **Transition1x level** = \mbox{$\omega$B97X/6-31G(d)}, functional and basis only. The level Transition1x was generated at, restricted. Defined at first use in Chapter 4, Section "Residual forces at the training geometries".
- **OMol25 level** = \mbox{$\omega$B97M-V/def2-TZVPD}, functional and basis only. The level OMol25 computed its labels at. Says nothing about the spin formalism; pair it with a surface: "at the OMol25 level on the restricted surface" / "on the unrestricted surface". Defined at the same place. Not to be confused with Chapter 3's expensive level, \mbox{$\omega$B97M-V/def2-TZVP} without the D, which is written out in full there.
- **Restricted surface / unrestricted surface** = the spin formalism, stated next to the level: "at the OMol25 level on the unrestricted surface".
- **OMol25 protocol** = the complete recipe OMol25 used for its labels (Chapter 4, Table `tab:protocol`): \mbox{$\omega$B97M-V/def2-TZVPD}, def2/J with RIJCOSX, DEFGRID3, TightSCF, unrestricted, symmetry breaking attempted before the SCF by rotating HOMO and LUMO of the guess by 20°, ORCA 6.0.0. Our protocol is the same except that the symmetry breaking is tested after the SCF with a stability analysis, in ORCA 5.0.4. Use "protocol" when the whole recipe is meant, e.g. "recomputed with the OMol25 protocol" in the Background OMol25 table; use "level" when only functional and basis are meant.
- **Ground state** is used only for the true electronic ground state (Background). For the lowest Kohn--Sham solution say "the restricted solution is stable/unstable" or "the lowest solution is closed-shell/broken-symmetry".

## Wordings

- **Spin formalism**, not "spin treatment": the choice restricted / unrestricted. Used in the Background choices figure, the RKS/UKS table caption, the "Two surfaces" paragraph, the Chapter 4 label table and the comparison table. Changed 2026-09-16.
- **Restricted / unrestricted** in prose. **RKS / UKS** only in the Background where they are introduced, in ORCA input lines, and in the symbols $E_\mathrm{RKS}$, $E_\mathrm{BS}$.
- **Transition state** is the default word. **Saddle** is used where the geometric meaning is in the foreground ("re-optimises the saddles", "a saddle of the unrestricted surface", "before trusting a saddle"). The Background defines a transition state as a first-order saddle point, so the two are interchangeable; do not introduce a third word.
- **Levels always with \omega**: \mbox{$\omega$B97X/6-31G(d)} and \mbox{$\omega$B97M-V/def2-TZVP(D)} in prose; in math superscripts $E^{\omega\text{B97X}}$, $E^{\omega\text{B97M}}$ (Chapter 3 error split). Plain "wB97" only in ORCA input lines.
- **Closed-shell / broken-symmetry** are the two groups of structures, always hyphenated as adjectives. "Stable / unstable" describe the restricted solution, not the group.
- **Label** is reserved for training labels (energy and forces stored with a geometry). The two terms for that choice are the surfaces, not "labelled surface". Never "the surface they were labelled on"; write "the surface the labels were computed on" or "the surface the labels describe". Structures are "labelled at" a level of theory.
- **Single point** as a noun ("a DFT single point"), **single-point** as an adjective ("single-point calculation").
- **Relabel**, **re-optimise**: these spellings.

## Levels of theory

- **Transition1x level** = \mbox{$\omega$B97X/6-31G(d)}, the plain ωB97X of Chai & Head-Gordon 2008 (`chai2008`), WITHOUT the D3 dispersion correction; ORCA 5.0.2, chosen by the Transition1x authors for compatibility with ANI-1x. The dataset fields are `wB97x_6-31G(d).energy/.forces`. Never write ωB97X-D3 for it (corrected 2026-09-18; the earlier text and the first Chapter 3 single points used wB97X-D3, which in ORCA is Lin et al.'s reparametrised functional plus D3, not ωB97X plus D3).
- **Chapter 3 cheap-level single points** = `results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json` (ORCA 5.0.4, `! wB97X 6-31G(d) TightSCF EnGrad`). The older `eval_benchmark_sp_fixed_full.json` holds the wB97X-D3 run and is kept for the record only.

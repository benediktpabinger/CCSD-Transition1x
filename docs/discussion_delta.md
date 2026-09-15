> Superseded 2026-09-14: the current Conclusion & Outlook text is docs/overleaf/04_Conclusion_Outlook.tex (mirror of Overleaf). This file is the draft history.

# Discussion, delta chapter

Written 2026-09-11 in this chat. Numbers from `docs/chapter_delta.tex`
(Results); the "seven of ten" from `results/hinge_t1x.csv` (⟨S²⟩ > 0.05 at
the Transition1x transition state, OMol25 level) applied to the delta high
tier rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937,
rxn6196, rxn0346, rxn1150. Label `ch:mr` is the OMol25 chapter.

```latex
\section{Discussion}
\label{sec:delta-discussion}

Three things are certain. The correction reaches the higher level of
theory on the low-MR reactions, where the restricted solution is the
ground state: the barrier error drops to a third, in single points and
in searches the model drives itself, and the force error drops to a
third in every tier, with under 1\,\% of the dataset relabelled. The
correction does not change the quality of the geometries: the
transition states found with MACE+$\Delta$ lie as close to the
reference as those found with MACE (mean RMSD 0.054 against
0.056\,\AA, Figure~\ref{fig:rmsd-parity}). No improvement was to be
made here, since the transition states of
\mbox{$\omega$B97X-D3/6-31G(d)} and \mbox{$\omega$B97M-V/def2-TZVP}
lie within 0.007\,\AA{} of each other, below the geometry error of the
base model.

The total error of MACE+$\Delta$ is the sum of two terms,
Eq.~\eqref{eq:error-split}: the deviation of MACE from
$\omega$B97X-D3, and the deviation of the head from the level gap. The
head is trained on the gap alone and reads frozen features, so it can
reduce only the second term. The first enters the sum unchanged: of
the 56\,meV total energy error, 42\,meV are MACE's own error against
$\omega$B97X-D3. A further improvement therefore has to reduce both
terms, and the head itself cannot fix the larger of the two.

On the mid- and high-MR reactions the correction overshoots, for
different reasons. In the mid tier the restricted solution is the
ground state at nine of the ten transition states
(Chapter~\ref{ch:mr}), so the reference is sound, and these reactions
are drawn from the middle of the $N_\text{FOD}$ ranking, so the
training set contains reactions of the same character in large
numbers. The overshoot is the head's own error: the gap between the two
levels shrinks with MR character, and the head, reading scalar features
alone, applies the gap it learned without resolving where it shrinks.
Why the features do not carry that information is not resolved here;
more training data might have made the distinction easier to learn,
but this was not tested. In the high tier the same head error is
present, and a deeper one comes on top: at seven of the ten transition
states the restricted surface is not the ground state, at either level
of theory (Chapter~\ref{ch:mr}), so the reference barrier itself lies
on a surface that is not the true one, and correcting towards it
cannot give the right answer.

Four changes could have made the correction better.

\begin{itemize}
  \item \textbf{A trainable encoder.} The head reads frozen features,
    which is why the baseline passes through. Unfreezing the last MACE
    layer or the readout on the same 80\,000 points would let the model
    move its own surface towards the target instead of adding an
    offset. The risk is forgetting the cheap-level surface where the
    relabelled points are sparse.
  \item \textbf{An input that sees the gap change.} The head predicts
    the difference from scalar features alone. A feature that tracks
    multireference character, such as $N_\text{FOD}$ or the
    $\langle S^2 \rangle$ of the label, would let the head learn that
    the gap shrinks instead of over-correcting.
  \item \textbf{Labels on the right surface.} Where the restricted
    solution is unstable, the head learns a difference between two
    restricted energies. Relabelling those points unrestricted, with a
    stability analysis, would give it a difference that describes the
    ground state.
  \item \textbf{A better base.} Every improvement of MACE on its own
    level transfers one to one, because that term passes through the
    sum unchanged.
\end{itemize}

The limits of this chapter are its 30 reactions, the restricted
treatment throughout, and def2-TZVP without the diffuse functions of
the target level.

The correction was not pushed further, because the principle had been
shown at scale. Allen et al. relabelled 3119 Transition1x and Grambow
geometries at unrestricted CCSD(T) and fine-tuned a DFT-trained
potential on them, with the geometries left as they
were~\cite{allen2026}, and while this thesis was written, DeePEST-OS
appeared, a reactive potential trained on 75\,000 reactions with
semi-empirical geometries and DFT labels~\cite{ren2026}. Both confirm
that high-level labels on inherited geometries lift a model to the
higher level. The open question was therefore not whether relabelling
works, but where it stops, which is what Chapter~\ref{ch:mr} took up.

```
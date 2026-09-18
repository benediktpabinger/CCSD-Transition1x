# Overleaf mirror

Each file in this folder is a 1:1 copy of one file of the Overleaf project,
under the same name. Rule: a file here is only changed together with its
Overleaf counterpart. Drafts and history stay in docs/.

| mirror file here | Overleaf file | content | draft source in docs/ |
|---|---|---|---|
| 00_Preface.tex | Frontmatter/Approval.tex | Approval page + Preface on the scope of the project | chat only |
| 00a_Abstract.tex | Frontmatter/Abstract.tex | Abstract | chat only |
| 01_Introduction_V2.tex | Chapters/01_Introduction_V2.tex | Introduction + Background and Theory | introduction_v2.tex + background.tex |
| 02_Delta.tex | Chapters/02_Delta.tex | Chapter 3, delta head | chapter_delta.tex |
| 03_OMOL25_Failure_Modes_V2.tex | Chapters/03_OMOL25_Failure_Modes_V2.tex | Chapter 4, OMol25, sections 4.1 and 4.2 | chapter_omol25.tex |
| 04_Conclusion_Outlook.tex | Chapters/04_Conclusion.tex | Section 4.3 (MR geometries) + Chapter 5, Conclusion & Outlook | section_mr_outofscope.tex + discussion_delta.md + discussion_omol25.md |
| 05_Appendix_Marks.tex | Backmatter/Appendix.tex | Appendix A, stability of the Marks references | marks_stability/REPORT.md, marks_stability.csv |

State 2026-09-14: 01 differs from Overleaf by the nine agreed fixes (to be
pasted over); 02, 03, 04 are exact copies of Overleaf.

Overleaf file names read off the project tree on 2026-09-17 (Approval.tex is
assumed to hold the Preface as well; confirm). Bibliography: Overleaf uses
bibliography.bib; new entries are added to docs/background_refs.bib here and
must be copied over (guner2003, marks2024fsm added 2026-09-16/17).

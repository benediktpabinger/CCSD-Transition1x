# -*- coding: utf-8 -*-
"""Fuellt die drei Schrittfolien der RQ1-Strecke (8, 9, 10) mit Inhalt.

Arbeitet auf Defence_edited.pptx, wie es gerade ist -- die Folien 7 bis 11
sind von Hand angelegt worden, also wird hier nichts neu gebaut, sondern
ergaenzt und aufgeraeumt.

DIE SEITE WECHSELT ABSICHTLICH
    Folie 8 Architektur rechts, 9 links, 10 rechts. Der Sprung ist das
    Signal, auf welche Seite man schauen soll. Gleich bleibt dagegen die
    GROESSE: vorher war das Bild 4.01, 3.93 und 4.57 Zoll breit, was wie ein
    Versehen aussah statt wie eine Ansage.

    python docs/defence/fill_delta_slides.py <deck.pptx>
"""
import copy
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

NAVY = RGBColor(0x03, 0x0F, 0x4F)
ORANGE = RGBColor(0xE0, 0x73, 0x1F)
INK = RGBColor(0x33, 0x33, 0x38)
DIM = RGBColor(0x7F, 0x7F, 0x86)

ML, MR = 1.00, 12.33
BODY_TOP, BOTTOM = 1.62, 6.78
AW, AH = 4.22, 4.40                  # Architektur, auf jeder Folie gleich
GUT = 0.50
PICS = "docs/defence/pics/"

deck = sys.argv[1]
prs = Presentation(deck)
S = list(prs.slides)


def tbox(sl, x, y, w, h):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def put(tf, parts, size=13, color=INK, bold=False, space=7, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space)
    for txt, b in parts:
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.bold = b or bold
        r.font.color.rgb = color
    return p


def heading(sl, x, y, w, num, text):
    tf = tbox(sl, x, y, w, 0.34)
    put(tf, [(num + "   ", True), (text, True)], size=16, color=NAVY,
        first=True)


def drop(sl, pred):
    for sh in list(sl.shapes):
        if sh.has_text_frame and pred(sh.text_frame.text):
            sh._element.getparent().remove(sh._element)
            return True
    return False


# ------------------------------------------------------------ 1 Tippfehler
FIX = [("relable", "relabel"), ("reference date", "reference data"),
       ("by … - …",
        "by Δ = ωB97M-V/def2-TZVP − ωB97X/6-31G(d)")]
n = 0
for sl in S:
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                for a, b in FIX:
                    if a in r.text:
                        r.text = r.text.replace(a, b)
                        n += 1
            # 'reference date' steht ueber zwei Laeufe verteilt; eine Suche
            # innerhalb eines Laufs findet es nie. Also das einzelne Wort
            # ersetzen, aber nur in dem Absatz, in dem es falsch ist.
            if "reference" in para.text and "date" in para.text:
                for r in para.runs:
                    if "date" in r.text:
                        r.text = r.text.replace("date", "data")
                        n += 1
print("Tippfehler ersetzt:", n)

# --------------------------------------------- 2 der Rest hinter dem Bild
if drop(S[8], lambda t: t.strip().startswith("Step 1 Train MACE")):
    print("Folie 9: verdeckter Step-1-Kasten entfernt")

# ------------------------------------------------------------ 3 die Folien
for idx in (7, 8, 9):
    sl = S[idx]
    for sh in list(sl.shapes):
        if "PICTURE" in str(sh.shape_type):
            sh._element.getparent().remove(sh._element)
        elif sh.has_text_frame and not sh.is_placeholder:
            sh._element.getparent().remove(sh._element)

# ---- Folie 8: Schritt 1, Architektur rechts
sl = S[7]
sl.shapes.add_picture(PICS + "arch_base.png", Inches(MR - AW),
                      Inches(BODY_TOP + 0.08), height=Inches(AH))
TX, TW = ML, MR - AW - GUT - ML
heading(sl, TX, BODY_TOP, TW, "Step 1", "Train MACE on Transition1x")
tf = tbox(sl, TX, BODY_TOP + 0.48, TW, 2.3)
put(tf, [("trained from scratch on the Transition1x training split, on its "
          "own ωB97X/6-31G(d) energies and forces", False)], first=True)
put(tf, [("100,000 geometries per epoch, 362 epochs — about four passes "
          "over the 9 million", False)])
put(tf, [("72.4 M parameters; afterwards ", False),
         ("frozen", True),
         (", so its prediction is identical before and after the head "
          "exists", False)])

tf = tbox(sl, TX, BODY_TOP + 2.70, TW, 0.7)
put(tf, [("81 meV/Å", True)], size=40, color=ORANGE, first=True)
tf = tbox(sl, TX, BODY_TOP + 3.36, TW, 1.1)
put(tf, [("force RMSE on the Transition1x test split", False)], size=14,
    color=INK, first=True)
put(tf, [("against 136 meV/Å for the PaiNN the Transition1x authors "
          "trained on the same data", False)], size=13, color=DIM)

# ---- Folie 9: Schritt 2a und 2b, Architektur links
sl = S[8]
sl.shapes.add_picture(PICS + "arch_head.png", Inches(ML),
                      Inches(BODY_TOP + 0.08), height=Inches(AH))
TX = ML + AW + GUT
TW = MR - TX
heading(sl, TX, BODY_TOP, TW, "Step 2a", "Relabel 1 % of the geometries")
tf = tbox(sl, TX, BODY_TOP + 0.46, TW, 1.5)
put(tf, [("the target is the difference: ", False),
         ("Δ = ωB97M-V/def2-TZVP − ωB97X/6-31G(d)", True),
         (", energies and forces alike", False)], first=True)
put(tf, [("5,000 reactions at random, 20 of the ~950 geometries in each", False)])
put(tf, [("80,592 geometries · ~100,000 ORCA runs · 3 to 4 days", False)])

# Die Spalte ist 5.16 Zoll hoch und traegt vier Bloecke. Das Bild bekommt,
# was uebrig bleibt: 5.21 Zoll breit, bei 2.62 zu 1 also 1.99 hoch. Breiter
# gesetzt lief es in den dritten Aufzaehlungspunkt darueber, schmaler werden
# die Abschnittsnamen darunter unlesbar.
sl.shapes.add_picture(PICS + "sampling.png", Inches(TX + 0.45),
                      Inches(BODY_TOP + 1.78), width=Inches(TW - 0.90))

heading(sl, TX, BODY_TOP + 3.94, TW, "Step 2b", "Train the correction head")
tf = tbox(sl, TX, BODY_TOP + 4.38, TW, 0.9)
put(tf, [("Huber on energy and forces, force weight 2.0; the checkpoint "
          "kept is the lowest ", False), ("force", True),
         (" loss — the head is built to drive NEB runs", False)],
    size=12, first=True)
put(tf, [("131,136 parameters · about six hours on one GPU", False)],
    size=12)

# ---- Folie 10: Schritt 3a und 3b, Architektur rechts
sl = S[9]
sl.shapes.add_picture(PICS + "arch_all.png", Inches(MR - AW),
                      Inches(BODY_TOP + 0.08), height=Inches(AH))
TX, TW = ML, MR - AW - GUT - ML
heading(sl, TX, BODY_TOP, TW, "Step 3a", "Build the reference")
tf = tbox(sl, TX, BODY_TOP + 0.46, TW, 1.3)
put(tf, [("a restricted ωB97M-V/def2-TZVP CI-NEB for every reaction of "
          "the test split — ", False), ("279 of 287 converge", True)],
    first=True)
put(tf, [("each gives ten geometries with energies and forces, the "
          "transition state, and the barrier", False)])

heading(sl, TX, BODY_TOP + 2.00, TW, "Step 3b", "Evaluate MACE+Δ")
tf = tbox(sl, TX, BODY_TOP + 2.46, TW, 2.3)
put(tf, [("30 reactions, 10 high / 10 mid / 10 low multireference "
          "character — ", False),
         ("none of them in any training or validation set", True)], first=True)
put(tf, [("on fixed geometries: 300 images — energies, forces, "
          "barriers", False)])
put(tf, [("driving the NEB itself: the barrier it finds, and the transition "
          "state as RMSD after Kabsch alignment", False)])

tf = tbox(sl, TX, BOTTOM - 0.75, TW, 0.7)
put(tf, [("the reference is restricted DFT, so agreement shows the head "
          "learned what it was trained on — not that it is closer to "
          "the truth", False)], size=12, color=DIM, first=True)

prs.save(deck)
print("geschrieben:", deck)

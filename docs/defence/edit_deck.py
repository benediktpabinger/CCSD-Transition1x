# -*- coding: utf-8 -*-
"""Defence.pptx -> Defence_edited.pptx. Das Original wird nur gelesen.

Behoben wird, was messbar falsch ist: Text ueber den Folienrand hinaus, Bilder
halb ausserhalb, Bildunterschriften in der Fussleiste, Platzhaltermuell und
Tippfehler. Dazu eine Titelhierarchie -- bisher waren die Ueberschriften so
gross wie der Fliesstext.

Gearbeitet wird NUR bis zum Backup-Trenner (Layout 'Front/Pause A'); alles ab
dem Trenner bleibt unberuehrt.
"""
import sys

from pptx import Presentation
from pptx.util import Inches, Pt

SRC, DST = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
log = []


def drop_slides(prs, idxs):
    lst = prs.slides._sldIdLst
    items = list(lst)
    for i in sorted(idxs, reverse=True):
        rId = items[i].get(
            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        prs.part.drop_rel(rId)
        lst.remove(items[i])
        log.append("Folie %d geloescht (DTU-Hilfsfolie)" % (i + 1))


# ------------------------------------------------- 1 die beiden Hilfsfolien
# Beide sagen selbst: "This slide is for guidance only. Delete it before you
# present - so the slide numbers on the other slides are correct."
help_idx = []
for i, sl in enumerate(prs.slides):
    t = " ".join(sh.text_frame.text for sh in sl.shapes if sh.has_text_frame)
    if "Help with DTU's PowerPoint templates" in t:
        help_idx.append(i)
drop_slides(prs, help_idx)

# ------------------------------------------------- 2 wo endet der Hauptteil
S = list(prs.slides)
sep = next((i for i, sl in enumerate(S)
            if sl.slide_layout.name.strip() == "Front/Pause A"), len(S))
log.append("Backup-Trenner ist jetzt Folie %d; bearbeitet werden 1-%d"
           % (sep + 1, sep))
MAIN = S[:sep]

# ------------------------------------------------- 3 Text
REPL = [
    ("structureswas", "structures was"),
    ("fideslity", "fidelity"),
    ("accurace", "accurate"),
    ("cab", "can"),
    ("Omol25", "OMol25"),
    ("Uma-S", "UMA-S"),
    ("Uma-M", "UMA-M"),
    ("ffffffffff", ""),
]
hits = {o: 0 for o, _ in REPL}
for sl in MAIN:
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                for old, new in REPL:
                    if old in r.text:
                        r.text = r.text.replace(old, new)
                        hits[old] += 1
for o, n in hits.items():
    log.append("  Text %-16r %dx" % (o, n))

# ------------------------------------------------- 4 Geometrie
RIGHT = Inches(12.80)
LEFTMIN = Inches(0.50)
BOTTOM = Inches(6.90)


def fix_shape(sh, i):
    """Form ganz auf die Folie holen. Bilder massstabstreu."""
    if sh.left is None or sh.width is None:
        return
    pic = "PICTURE" in str(sh.shape_type)
    if sh.left > prs.slide_width or sh.left + sh.width < 0:
        sh._element.getparent().remove(sh._element)
        log.append("Folie %d: Form %d lag komplett neben der Folie, entfernt"
                   % (i, sh.shape_id))
        return
    if pic:
        ar = sh.height / float(sh.width)
        moved = False
        if sh.left < LEFTMIN:
            sh.left = LEFTMIN
            moved = True
        if sh.left + sh.width > RIGHT:
            sh.width = int(RIGHT - sh.left)
            sh.height = int(sh.width * ar)
            moved = True
        if sh.top + sh.height > BOTTOM:
            sh.height = int(BOTTOM - sh.top)
            sh.width = int(sh.height / ar)
            moved = True
        if moved:
            log.append("Folie %d: Bild %d auf die Folie geholt" % (i, sh.shape_id))
    else:
        if sh.left + sh.width > RIGHT and not sh.is_placeholder:
            sh.width = int(RIGHT - sh.left)
            log.append("Folie %d: Textkasten %d gekuerzt, ragte ueber den Rand"
                       % (i, sh.shape_id))


for i, sl in enumerate(MAIN, 1):
    for sh in list(sl.shapes):
        fix_shape(sh, i)

# ------------------------------------------------- 5 linker Rand
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if sh.left is None or "PICTURE" in str(sh.shape_type):
            continue
        if sh.left < 0:
            log.append("Folie %d: Form %d (%r) ragte links hinaus, "
                       "auf die Folie geholt"
                       % (i, sh.shape_id,
                          sh.text_frame.text[:20] if sh.has_text_frame else ""))
            sh.left = LEFTMIN

# ------------------------------------------------- 6 Titelhierarchie
# Die Ueberschriften waren so gross wie der Fliesstext und mal navy, mal
# schwarz. Einheitlich: fett, 20 pt, DTU-navy. Nur echte Kurztitel oben.
from pptx.dml.color import RGBColor
NAVY = RGBColor(0x03, 0x0F, 0x4F)
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if not sh.has_text_frame or sh.top is None:
            continue
        t = sh.text_frame.text.strip()
        if not t or sh.top > Inches(1.6) or len(t) > 70:
            continue
        if sh.is_placeholder and sh.placeholder_format.type in (13, 16):
            continue
        # Eine Spalte gleich breiter Kaesten ist eine Liste, kein Titel --
        # sonst wird der erste Listenpunkt zur Ueberschrift befoerdert.
        twins = sum(1 for o in sl.shapes
                    if o.has_text_frame and o.left is not None
                    and abs(o.left - sh.left) < Inches(0.05)
                    and abs((o.width or 0) - sh.width) < Inches(0.05))
        if twins >= 3:
            continue
        cur = max([r.font.size.pt for para in sh.text_frame.paragraphs
                   for r in para.runs if r.font.size] or [0])
        if cur >= 20:                      # schon ein richtiger Titel
            continue
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                r.font.size = Pt(20)
                r.font.bold = True
                r.font.color.rgb = NAVY
        log.append("Folie %d: Titel %r  %s -> 20 pt fett navy"
                   % (i, t[:40], ("%gpt" % cur) if cur else "geerbt"))

# ------------------------------------------------- 7 zwei Titel nebeneinander
# Auf der Relabelling-Folie stehen zwei Ueberschriften nebeneinander. Der linke
# Kasten war 7.36" breit und ragte in die rechte Spalte; bei 20 pt laufen die
# beiden ineinander. Also: jeder Titel in seine Spalte, und beide so weit
# hoch, dass die zweite Zeile nicht auf den Fliesstext faellt.
FIXUPS = [
    ("How can we upgrade", 1.32, 1.00, 5.20),
    ("Relabelling: Upgrading", 6.94, 1.00, 5.86),
]
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        for key, L, T, W in FIXUPS:
            if t.startswith(key):
                sh.left, sh.top, sh.width = Inches(L), Inches(T), Inches(W)
                log.append("Folie %d: Titel %r neu gesetzt (x=%s y=%s b=%s)"
                           % (i, t[:32], L, T, W))

# ------------------------------------------------- 8 RQ2-Spalte entzerren
# Auf der Zwei-Wege-Folie hat die RQ2-Spalte eine Zeile mehr als die RQ1-Spalte;
# der letzte Aufzaehlungspunkt lag auf der Bildunterschrift darunter.
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith(
                "To investigate we test"):
            for para in sh.text_frame.paragraphs:
                for r in para.runs:
                    if r.font.size and r.font.size.pt > 12:
                        r.font.size = Pt(12)
            log.append("Folie %d: RQ2-Text auf 12 pt" % i)
            # das Bild darunter etwas tiefer, aber ueber der Fussleiste
            for o in sl.shapes:
                if ("PICTURE" in str(o.shape_type) and o.left > Inches(9.5)
                        and o.top > Inches(4.0)):
                    ar = o.height / float(o.width)
                    o.top = Inches(4.95)
                    o.height = int(Inches(7.02) - o.top)
                    o.width = int(o.height / ar)
                    log.append("Folie %d: rechtes Bild tiefer gesetzt" % i)


# ================================================= 9 ein Raster fuer alle
# Bis hierher war nur repariert. Jetzt bekommt jede Folie dieselbe Geometrie:
# Titel oben links in einem festen Feld, darunter zwei Spalten auf festen
# x-Werten, Fliesstext 14 pt, Spaltenkopf 16 pt, Bildunterschrift 10 pt.
# Farben bleiben, wie sie waren -- nur Groesse und Ort werden vereinheitlicht.
ML, MR = 1.00, 12.33
CW = MR - ML                       # 11.33 Inhaltsbreite
TITLE_Y, TITLE_H = 0.50, 0.85
COL_L, COL_R, COLW = 1.00, 6.93, 5.40
BODY_TOP, BOTTOM = 1.60, 6.75


def place(sh, x=None, y=None, w=None, h=None, keep_ar=False):
    if keep_ar and w is not None and sh.width:
        ar = sh.height / float(sh.width)
        sh.width = Inches(w)
        sh.height = int(Inches(w) * ar)
    else:
        if w is not None:
            sh.width = Inches(w)
        if h is not None:
            sh.height = Inches(h)
    if x is not None:
        sh.left = Inches(x)
    if y is not None:
        sh.top = Inches(y)


def fit_below(sh, y, bottom=BOTTOM, x=None, col=None):
    """Bild auf y setzen und so skalieren, dass es ueber der Fussleiste endet."""
    ar = sh.height / float(sh.width)
    h = min(sh.height / 914400.0, bottom - y)
    w = h / ar
    if col is not None and w > COLW:
        w = COLW
        h = w * ar
    sh.width, sh.height = Inches(w), Inches(h)
    sh.top = Inches(y)
    sh.left = Inches(col + (COLW - w) / 2.0 if col is not None else x)


def setsize(sh, pt):
    for para in sh.text_frame.paragraphs:
        for r in para.runs:
            r.font.size = Pt(pt)


def byid(sl):
    return {sh.shape_id: sh for sh in sl.shapes}


BODY, HEAD = 14, 16

for i, sl in enumerate(MAIN, 1):
    d = byid(sl)

    # der leere Kasten, in dem frueher 'ffffffffff' stand
    if 4 in d and d[4].has_text_frame and not d[4].text_frame.text.strip():
        d[4]._element.getparent().remove(d[4]._element)
        log.append("Folie %d: leerer Restkasten entfernt" % i)
        d = byid(sl)

    if i == 1:                                     # Motivation
        place(d[5], ML, TITLE_Y, CW, TITLE_H)
        place(d[6], COL_R, BODY_TOP, COLW, 4.6); setsize(d[6], BODY)
        for k, sid in enumerate((7, 8, 9, 10)):
            place(d[sid], COL_L, 1.80 + 0.90 * k, COLW, 0.5)
            setsize(d[sid], HEAD)

    elif i == 2:                                   # Transition1x
        place(d[5], ML, TITLE_Y, CW, TITLE_H)
        place(d[6], COL_R, BODY_TOP, COLW, 3.4); setsize(d[6], BODY)
        place(d[12], COL_L, BODY_TOP, 1.50, keep_ar=True)
        fit_below(d[14], 3.85, col=COL_L)
        if 7 in d:                                 # der lose Kasten 'Molecules'
            d[7]._element.getparent().remove(d[7]._element)
            log.append("Folie %d: loser Kasten 'Molecules' entfernt" % i)

    elif i == 3:                                   # Niveau der Theorie
        place(d[5], ML, TITLE_Y, CW, TITLE_H)
        place(d[26], ML, 1.75, CW, keep_ar=True)
        place(d[22], ML, 6.05, CW, 0.55); setsize(d[22], BODY)

    elif i == 4:                                   # Relabelling
        place(d[9], ML, TITLE_Y, CW, TITLE_H)      # die Frage wird der Titel
        place(d[5], COL_R, BODY_TOP, COLW, 0.62); setsize(d[5], HEAD)
        place(d[6], COL_R, 2.30, COLW, 3.4); setsize(d[6], BODY)
        place(d[8], COL_L, 2.30, COLW, keep_ar=True)

    elif i == 5:                                   # die zwei Forschungsfragen
        for sid, col in ((9, COL_L), (5, COL_R)):
            place(d[sid], col, BODY_TOP, COLW, 0.95); setsize(d[sid], HEAD)
        for sid, col in ((15, COL_L), (14, COL_R)):
            place(d[sid], col, 2.70, COLW, 0.35); setsize(d[sid], BODY)
        for sid, col in ((17, COL_L), (19, COL_R)):
            place(d[sid], col, 3.20, COLW, keep_ar=True)
        for sid, col in ((12, COL_L), (10, COL_R)):
            fit_below(d[sid], 5.30, col=col)
        if 18 in d and d[18].has_text_frame and not d[18].text_frame.text.strip():
            d[18]._element.getparent().remove(d[18]._element)

    elif i == 6:                                   # Two ways to relabel
        place(d[2], ML, TITLE_Y, CW, TITLE_H)
        for sid in (3, 4, 5, 6):                   # linke Spalte
            d[sid].left = int(d[sid].left - Inches(0.90))
        for sid in (7, 8, 9, 10):                  # rechte Spalte
            d[sid].left = int(d[sid].left - Inches(0.62))
        d[11].left = Inches(6.66)                  # die Trennlinie
        place(d[3], COL_L, BODY_TOP, COLW, 0.80); setsize(d[3], HEAD)
        place(d[7], COL_R, BODY_TOP, COLW, 0.80); setsize(d[7], HEAD)
        place(d[4], COL_L, 2.55, COLW, 1.40); setsize(d[4], BODY)
        place(d[8], COL_R, 2.55, COLW, 1.90); setsize(d[8], BODY)

    elif i == 7:                                   # die knappe RQ-Folie
        for sid, col in ((9, COL_L), (5, COL_R)):
            place(d[sid], col, BODY_TOP, COLW, 0.95); setsize(d[sid], HEAD)
        for sid, col in ((12, COL_L), (10, COL_R)):
            fit_below(d[sid], 3.00, col=col)

    log.append("Folie %d: auf das Raster gesetzt" % i)

# Titel einheitlich 28 pt
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.top is not None and \
                abs(sh.top - Inches(TITLE_Y)) < Inches(0.02) and \
                abs(sh.left - Inches(ML)) < Inches(0.02):
            setsize(sh, 28)
            log.append("Folie %d: Titel 28 pt" % i)


# ================================================ 10 letzter Durchgang
from pptx.enum.text import MSO_ANCHOR

# a) Der Titelplatz bestimmt das Aussehen, nicht die Herkunft der Form:
#    28 pt, fett, navy, oben buendig -- auf jeder Folie gleich.
for i, sl in enumerate(MAIN, 1):
    for sh in sl.shapes:
        if not (sh.has_text_frame and sh.top is not None):
            continue
        if abs(sh.top - Inches(TITLE_Y)) > Inches(0.02) or \
                abs(sh.left - Inches(ML)) > Inches(0.02):
            continue
        sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                r.font.size = Pt(28)
                r.font.bold = True
                r.font.color.rgb = NAVY
        # b) Fragen bekommen ein Fragezeichen
        t = sh.text_frame.text.rstrip()
        if t.lower().startswith(("how ", "can ", "what ", "why ")) and \
                not t.endswith("?"):
            runs = [r for para in sh.text_frame.paragraphs for r in para.runs
                    if r.text.strip()]
            if runs:
                runs[-1].text = runs[-1].text.rstrip() + "?"
                log.append("Folie %d: Fragezeichen ergaenzt" % i)
        log.append("Folie %d: Titel 28 pt fett navy, oben buendig" % i)

# c) Folie 1: die Anwendungsliste beginnt auf derselben Hoehe wie der Text
#    rechts und steht enger, sonst zerfaellt die linke Spalte.
d = byid(MAIN[0])
for k, sid in enumerate((7, 8, 9, 10)):
    if sid in d:
        place(d[sid], COL_L, BODY_TOP + 0.75 * k, COLW, 0.45)

# d) Folie 6 traegt eine Zeile mehr als die anderen. Die Bilder tiefer, damit
#    der letzte Aufzaehlungspunkt nicht auf der Bildunterschrift liegt.
if len(MAIN) >= 6:
    d6 = byid(MAIN[5])
    for sid, x in ((5, 1.00), (6, 3.85), (9, 6.93), (10, 9.78)):
        if sid not in d6:
            continue
        sh = d6[sid]
        ar = sh.height / float(sh.width)
        h = min(sh.height / 914400.0, 6.75 - 4.75)
        sh.height, sh.width = Inches(h), Inches(h / ar)
        sh.top, sh.left = Inches(4.75), Inches(x)
    # Die RQ2-Spalte hat zwei Zeilen mehr als jede andere Spalte im Vortrag.
    # Sie bekommt als einzige 12 pt -- sonst laeuft der letzte Punkt in die
    # Bildunterschrift. Ueberlappender Text ist der groessere Bruch in der
    # Einheitlichkeit als ein Schriftgrad Unterschied.
    if 8 in d6:
        setsize(d6[8], 12)
        log.append("Folie 6: RQ2-Spalte auf 12 pt (zwei Zeilen mehr)")
    log.append("Folie 6: Bilder tiefer, Text bleibt frei")

# e) Folie 5 und 7 hatten als einzige gar keinen Titel -- der auffaelligste
#    Bruch in der Einheitlichkeit. Der Text ist beschreibend, nicht neu:
#    beide Folien zeigen genau die zwei Forschungsfragen.
for idx in (4, 6):
    if idx >= len(MAIN):
        continue
    sl = MAIN[idx]
    has = any(sh.has_text_frame and sh.top is not None
              and abs(sh.top - Inches(TITLE_Y)) < Inches(0.02)
              for sh in sl.shapes)
    if has:
        continue
    tb = sl.shapes.add_textbox(Inches(ML), Inches(TITLE_Y),
                               Inches(CW), Inches(TITLE_H))
    tb.text_frame.word_wrap = True
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = "The two research questions"
    r.font.size, r.font.bold, r.font.color.rgb = Pt(28), True, NAVY
    log.append("Folie %d: Titel ergaenzt (%r)" % (idx + 1, r.text))

prs.save(DST)


print("\n".join(log))
print("\ngeschrieben:", DST)

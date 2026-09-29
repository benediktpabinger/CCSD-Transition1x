# -*- coding: utf-8 -*-
"""Die vereinfachte Architektur von MACE+Delta, fuer die Verteidigung.

Die ausfuehrliche Fassung (Pictures/mace_delta_architecture_v4.png) bleibt der
Arbeit vorbehalten. Sie traegt Beschriftungen bei 1.5 % der Bildbreite; auf
einer Folie mit 3.8 Zoll waeren das 5 pt, also unlesbar.

WAS DIESES BILD SAGEN SOLL
    E_MACE  ist die Energie auf dem Transition1x-Niveau,
    Delta   ist die Differenz zweier Niveaus.
Das ist der Kern, und er steht deshalb IN den Kaesten, nicht darunter.
Eine fruehere Fassung haengte ihn als freien Fettsatz unter die Kaesten: der
Text war breiter als der Kasten, zu dem er gehoerte, das Minuszeichen
schwebte, und nichts war an irgendetwas ausgerichtet. Text, der zu einem
Kasten gehoert, gehoert in den Kasten.

Farben wie im uebrigen Foliensatz: grau = uebernommen und eingefroren,
orange = neu trainiert. (Die Arbeit benutzt dort blau; im Vortrag ist orange
durchgehend die Farbe fuer 'das Neue'.)

    python docs/defence/draw_arch_simple.py docs/defence/pics

Erzeugt eine Datei je Hervorhebung:
    arch_all     alles aktiv                -- Uebersichtsfolie und Schritt 4
    arch_base    Encoder + Energie-Readout  -- Schritt 1
    arch_target  der Delta-Ausgang          -- Schritt 2
    arch_head    der Korrekturkopf          -- Schritt 3
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ORANGE, ORANGE_F, ORANGE_T = "#E0731F", "#FBE3D0", "#9C4A15"
GREY_E, GREY_F = "#8C8C92", "#ECECEF"
INK, DIM = "#222222", "#5A5A60"
MUTE_E, MUTE_F, MUTE_T = "#D5D5DA", "#F6F6F8", "#B0B0B6"

TITLE, SUB, KERN = 15, 9.5, 10.5
# Die Achse ist 100 Einheiten hoch auf 6.4 Zoll, also ist eine Einheit
# 0.064 Zoll = 4.6 pt. Damit laesst sich eine Zeilenhoehe in Einheiten
# ausrechnen, statt sie zu raten -- vorher ueberlappten die Zeilen.
PT_PER_UNIT = 4.6
LEAD = 1.20                      # Abstand zwischen zwei Zeilen, in Einheiten
W = r"$\omega$"
NL = chr(10)


def style(kind, on):
    if not on:
        return MUTE_E, MUTE_F, MUTE_T, MUTE_T, 1.2
    if kind == "new":
        return ORANGE, ORANGE_F, ORANGE_T, ORANGE_T, 2.2
    return GREY_E, GREY_F, INK, DIM, 1.8


def box(ax, x0, y0, x1, y1, lines, kind, on, lx=None, corner=None):
    """lines: Liste (text, groesse, fett). Alle zentriert, im Kasten gestapelt.

    corner: kleine Beschriftung in der oberen linken Ecke. Sie sagt, in
    welchem Verhaeltnis der Kasten zu dem steht, was in ihm steht ('predicts'),
    ohne sich in die zentrierte Stapelung zu draengen. Frueher stand das Wort
    mitten im Kasten und war eine dritte Zeile, die um denselben Platz
    konkurrierte.
    """
    ec, fc, tc, sc, lw = style(kind, on)
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    drop = 0.0
    if corner:
        ax.text(x0 + 2.2, y1 - 2.2, corner, ha="left", va="top",
                fontsize=SUB - 0.5, style="italic", color=sc, zorder=4)
        drop = 1.4                       # Inhalt etwas tiefer, Ecke frei
    h = [s / PT_PER_UNIT for _, s, _ in lines]
    total = sum(h) + LEAD * (len(lines) - 1)
    y = 0.5 * (y0 + y1) + total / 2.0 - h[0] / 2.0 - drop
    for i, (txt, size, bold) in enumerate(lines):
        # Eine einzige Ausrichtung, alles zentriert. Zentriertes Symbol ueber
        # linksbuendiger Gleichung war der eigentliche Fehler: zwei Achsen in
        # einem Kasten, und nichts stand zu etwas anderem in Bezug.
        ax.text(0.5 * (x0 + x1), y, txt, ha="center", va="center",
                fontsize=size, fontweight="bold" if bold else "normal",
                color=tc if size >= KERN else sc, zorder=4)
        if i + 1 < len(lines):
            y -= h[i] / 2.0 + LEAD + h[i + 1] / 2.0


def arrow(ax, x0, y0, x1, y1, on):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.6,
                                 color=INK if on else MUTE_T,
                                 shrinkA=0, shrinkB=0, zorder=2))


def draw(mode, path):
    base = mode in ("all", "base")
    head = mode in ("all", "head")
    targ = mode in ("all", "target")
    allon = mode == "all"

    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    box(ax, 33, 92, 67, 99.5, [("geometry  R", TITLE, True)],
        "base", base or allon)
    arrow(ax, 50, 92, 50, 88, base or allon)

    box(ax, 17, 77, 83, 88, [("MACE encoder", TITLE, True),
                             ("72.4 M parameters, frozen", SUB, False)],
        "base", base or allon)
    arrow(ax, 50, 77, 50, 73, base or allon)

    box(ax, 6, 62, 94, 73, [("per-atom features", TITLE, True),
                            ("2,048 of 17,408 do not rotate", SUB, False)],
        "base", base or allon)

    arrow(ax, 40, 62, 26, 53, base or allon)
    arrow(ax, 60, 62, 74, 53, head or allon)

    box(ax, 3, 41, 48, 53, [("energy readout", TITLE, True),
                            ("17,442 parameters", SUB, False)],
        "base", base or allon)
    box(ax, 52, 41, 97, 53, [("correction head", TITLE, True),
                             ("131,136 parameters", SUB, False)],
        "new", head or allon)

    arrow(ax, 25, 41, 25, 36, base or allon)
    arrow(ax, 75, 41, 75, 36, head or allon)

    # Der Kern, im Kasten: Symbol, dann wofuer es steht. Delta braucht zwei
    # Zeilen, weil es eine Differenz ist -- dass die rechte Seite mehr zu
    # sagen hat, ist genau die Aussage.
    box(ax, 2, 18, 49, 36,
        [(r"$E_{\mathrm{MACE}}$", TITLE + 1, True),
         (W + "B97X/6-31G(d)", KERN, False)],
        "base", base or allon, corner="predicts")
    box(ax, 51, 18, 98, 36,
        [(r"$\Delta$", TITLE + 1, True),
         (W + "B97M-V/def2-TZVP", KERN, False),
         ("−  " + W + "B97X/6-31G(d)", KERN, False)],
        "new", targ or head or allon, corner="predicts")

    arrow(ax, 30, 18, 42, 13, allon)
    arrow(ax, 70, 18, 58, 13, allon)

    box(ax, 14, 2, 86, 13,
        [(r"$E_{\mathrm{MACE}} + \Delta$", TITLE, True),
         ("one forward pass, forces by differentiation", SUB, False)],
        "base", allon)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("geschrieben:", path)


out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
for m in ("all", "base", "target", "head"):
    draw(m, os.path.join(out, "arch_%s.png" % m))

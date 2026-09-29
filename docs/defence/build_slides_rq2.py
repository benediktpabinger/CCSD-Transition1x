"""RQ2 methods: an overview slide and four step slides, in the style of the
RQ1 step slides of the draft deck. Built in the DTU template so the slides
can be copied into the main deck.

Run from the repo root:  python docs/defence/build_slides_rq2.py
Output: docs/defence/slides_rq2.pptx
"""
import zipfile
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

TEMPLATE = "docs/defence/DTU Template 16_9 - Navy Blue EN.potx"
WORK = "docs/defence/_template_as_pptx.pptx"
OUT = "docs/defence/slides_rq2.pptx"
PICS = "docs/defence/pics/"
NAVY = RGBColor(0x03, 0x0F, 0x4F)
ORANGE = RGBColor(0xE0, 0x73, 0x1F)
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x7F, 0x7F, 0x86)
TITLE = "RQ2 – Higher-Fidelity Relabelling Without Re-optimising the Path Is Not Enough"

zin = zipfile.ZipFile(TEMPLATE)
zout = zipfile.ZipFile(WORK, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "[Content_Types].xml":
        data = data.replace(b"presentationml.template.main+xml", b"presentationml.presentation.main+xml")
    zout.writestr(it, data)
zout.close()
prs = Presentation(WORK)
sld = prs.slides._sldIdLst
for s in list(sld):
    prs.part.drop_rel(s.rId)
    sld.remove(s)
L = {l.name.strip(): l for l in prs.slide_masters[0].slide_layouts}


def textbox(slide, x, y, w, h, paras, size=14, space_after=6, anchor=MSO_ANCHOR.TOP):
    """paras: list of paragraphs; each paragraph a list of (text, opts) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        for txt, opts in runs:
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", False)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", INK)
    return tb


def new_slide(title):
    s = prs.slides.add_slide(L["Kun titel"])
    s.shapes.title.text_frame.text = title
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(22)
            r.font.color.rgb = NAVY
    return s


def step_head(txt):
    return [(txt, {"bold": True, "size": 18, "color": NAVY})]


def line(txt, bold_part=None):
    if bold_part:
        a, b = txt.split(bold_part, 1)
        return [(a, {}), (bold_part, {"bold": True}), (b, {})]
    return [(txt, {})]


def note(txt):
    return [(txt, {"size": 12, "color": GREY})]


# ---------------------------------------------------------------- overview
s = new_slide(TITLE)
s.shapes.add_picture(PICS + "pic_rq2_workflow.png", Inches(3.4), Inches(1.75), width=Inches(9.2))
textbox(s, 0.5, 1.6, 2.7, 5.3, [
    step_head("Step 1"), line("Select reactions where the two surfaces differ"),
    step_head("Step 2"), line("Let the OMol25 models search transition states"),
    step_head("Step 3"), line("Check every result with DFT on the unrestricted surface"),
    step_head("Step 4"), line("Sort into closed-shell and broken-symmetry, same metrics on both"),
], size=12, space_after=9)
s.notes_slide.notes_text_frame.text = (
    "Same test as RQ1: the model searches, DFT checks. Two differences. The models are the OMol25 models, "
    "trained on all of Transition1x relabelled unrestricted. And the check is done on the unrestricted surface, "
    "because that is the surface the labels came from.")

# ---------------------------------------------------------------- step 1
s = new_slide(TITLE)
s.shapes.add_picture(PICS + "pic_rq2_workflow.png", Inches(8.2), Inches(1.9), width=Inches(4.7))
textbox(s, 0.5, 1.7, 7.4, 5.2, [
    step_head("Step 1   Select reactions"),
    line("45 reactions from the Transition1x test split, ranked by multireference character (N₁FOD): 15 low, 15 mid, 15 high".replace("N₁FOD", "N_FOD")),
    line("at high MR the restricted solution is expected to be unstable, so broken-symmetry transition states appear there", "broken-symmetry transition states"),
    line("no unrestricted reference transition state exists for any of them. That is the point, so the check has to work without one", "no unrestricted reference transition state exists"),
], size=14, space_after=10)
s.notes_slide.notes_text_frame.text = (
    "The reactions are chosen so that the two surfaces differ for some of them and not for others. "
    "Nobody has an unrestricted reference for these transition states, and building one would answer the question by hand.")

# ---------------------------------------------------------------- step 2
s = new_slide(TITLE)
s.shapes.add_picture(PICS + "pic_rq2_workflow.png", Inches(8.2), Inches(1.9), width=Inches(4.7))
textbox(s, 0.5, 1.7, 7.4, 5.2, [
    step_head("Step 2   Search"),
    line("three OMol25 models, used as released: UMA-S, UMA-M, eSEN", "UMA-S, UMA-M, eSEN"),
    line("each runs the full workflow itself: relax reactant and product, CI-NEB with the same settings as the DFT reference"),
    line("135 searches, 133 report success. The workflow never says it could not find one", "133 report success"),
], size=14, space_after=10)
s.notes_slide.notes_text_frame.text = (
    "The models are not fine-tuned. They drive the same NEB machinery as the DFT reference, so what differs is the surface, not the search.")

# ---------------------------------------------------------------- step 3
s = new_slide(TITLE)
s.shapes.add_picture(PICS + "pic_rq2_workflow.png", Inches(8.2), Inches(1.9), width=Inches(4.7))
textbox(s, 0.5, 1.7, 7.4, 5.2, [
    step_head("Step 3   Check"),
    line("one DFT single point at every transition state a model delivers, with the OMol25 protocol: ωB97M-V/def2-TZVPD, unrestricted, plus a stability analysis", "OMol25 protocol"),
    line("no optimisation. The structure is judged where the model left it", "no optimisation"),
    line("it yields: the residual force on the unrestricted surface (is it a stationary point?), the energy (the barrier, and the model's own error at the point), and ⟨S²⟩"),
    [("", {})],
    note("the residual force is the largest force component at the structure; zero at a transition state relaxed on that surface"),
], size=14, space_after=10)
s.notes_slide.notes_text_frame.text = (
    "One single point per structure, the same calculation OMol25 used for its labels, plus the stability analysis "
    "that tells whether a lower unrestricted solution exists. Nothing is moved.")

# ---------------------------------------------------------------- step 4
s = new_slide(TITLE)
s.shapes.add_picture(PICS + "pic_rq2_workflow.png", Inches(8.2), Inches(1.9), width=Inches(4.7))
textbox(s, 0.5, 1.7, 7.4, 5.2, [
    step_head("Step 4   Sort and compare"),
    line("⟨S²⟩ = 0: closed-shell, 82 structures.  ⟨S²⟩ > 0: broken-symmetry, 53 structures. No borderline case", "No borderline case"),
    line("closed-shell: the two surfaces coincide, the models are on home ground"),
    line("broken-symmetry: the unrestricted surface has its own transition state, and no training geometry sat on it", "no training geometry sat on it"),
    line("same metrics on both groups. Do the models do as well in the second?", "Do the models do as well in the second?"),
    [("", {})],
    note("the comparison is between the groups, not against a reference; the residual force says how far from a stationary point a structure is, on the surface the labels came from"),
], size=14, space_after=10)
s.notes_slide.notes_text_frame.text = (
    "Where the restricted solution is stable the two surfaces are the same and the models are on home ground. "
    "Where it is unstable they are not. The results compare the two.")

prs.save(OUT)
print("written", OUT, "slides:", len(prs.slides))

"""
Build the self-contained Prism (Optica) submission package.

Copies the manuscript, supplement, macros, bibliography and Optica template
files into paper/prism_submission/, rewrites figure paths from
../code/figures/paper/ to figures/, copies only the figures the two documents
reference, compiles both documents inside the package (pdflatex, bibtex,
pdflatex x2) to prove the package builds on its own, and zips it to
paper/prism_submission.zip. The previous package is replaced.

Usage: python scripts/make_prism_package.py
"""
import os
import re
import shutil
import subprocess
import zipfile

PAPER = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
FIGS = os.path.normpath(os.path.join(PAPER, "..", "code", "figures", "paper"))
OUT = os.path.join(PAPER, "prism_submission")
ZIP = os.path.join(PAPER, "prism_submission.zip")
DOCS = ["oe_main", "oe_supplement"]
SUPPORT = ["numbers.tex", "refs.bib", "optica-article.cls", "opticajnl.cls",
           "opticajnl.bst", "jabbrv.sty", "jabbrv-ltwa-all.ldf", "jabbrv-ltwa-en.ldf",
           "cover_letter.md"]
FIG_RE = re.compile(r"\\includegraphics(\[[^\]]*\])?\{\.\./code/figures/paper/([^}]+)\}")


def run(cmd):
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    return r.returncode


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"))
    shutil.copytree(os.path.join(PAPER, "styles"), os.path.join(OUT, "styles"))
    for f in SUPPORT:
        shutil.copy2(os.path.join(PAPER, f), OUT)

    used = set()
    for doc in DOCS:
        src = open(os.path.join(PAPER, doc + ".tex"), encoding="utf-8").read()
        used |= {m.group(2) for m in FIG_RE.finditer(src)}
        src = FIG_RE.sub(lambda m: r"\includegraphics" + (m.group(1) or "")
                         + "{figures/" + m.group(2) + "}", src)
        assert "../code" not in src, f"{doc}: unrewritten repo path"
        with open(os.path.join(OUT, doc + ".tex"), "w", encoding="utf-8") as fh:
            fh.write(src)
    for fig in sorted(used):
        shutil.copy2(os.path.join(FIGS, fig), os.path.join(OUT, "figures", fig))

    for doc in DOCS:
        run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
        run(["bibtex", doc])
        run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
        rc = run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
        log = open(os.path.join(OUT, doc + ".log"), encoding="latin-1").read()
        errors = [l for l in log.splitlines() if l.startswith("!")]
        pages = re.search(r"Output written on \S+ \((\d+) pages", log)
        print(f"{doc}: rc={rc} pages={pages.group(1) if pages else '?'} errors={len(errors)}")
        if errors or not pages:
            raise SystemExit(f"{doc} failed to build inside the package: {errors[:3]}")

    # keep sources, figures, bbl (so the journal need not run bibtex) and PDFs
    for f in os.listdir(OUT):
        if f.endswith((".aux", ".log", ".blg", ".out")):
            os.remove(os.path.join(OUT, f))
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, OUT))
    print(f"wrote {ZIP} ({len(used)} figures)")


if __name__ == "__main__":
    main()

"""
Build the JOSA A submission package.

Copies the manuscript, supplement, cover letter, generated macros,
bibliography, Optica template files and every referenced figure into a clean
build directory, rewrites figure paths from ../code/figures/paper/ to figures/,
compiles all three documents there (pdflatex, bibtex, pdflatex x2) to prove the
sources are self-contained, fails on any LaTeX error, undefined reference or
[PENDING] macro, and writes the upload-ready files to part2/submission/:

  <title>.pdf                       manuscript (Prism: "Manuscript")
  <title> - Supplement 1.pdf        supplemental document (Prism: "Supplement 1")
  Cover Letter.pdf
  LaTeX source.zip                  self-contained sources, .bbl included
  Abstract (plain text).txt         for the Prism abstract field
  SUBMISSION_GUIDE.md               upload steps, form fields, reviewers, checklist

The previous package is replaced. part2/submission/ is git-ignored.

Usage (from part2/code): python scripts/make_submission_package.py
"""
import os
import re
import shutil
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "..", "paper"))
FIGS = os.path.normpath(os.path.join(PAPER, "..", "code", "figures", "paper"))
OUT = os.path.normpath(os.path.join(PAPER, "..", "submission"))
BUILD = os.path.join(OUT, "_build")

TITLE_FILE = ("Media-in-the-Loop Holography - Designing Photopolymer Exposures "
              "Through a Differentiable Recording Model")
DOCS = {"manuscript": f"{TITLE_FILE}.pdf",
        "supplement": f"{TITLE_FILE} - Supplement 1.pdf",
        "cover_letter": "Cover Letter.pdf"}
SUPPORT = ["numbers.tex", "refs.bib", "optica-article.cls", "opticajnl.bst",
           "jabbrv.sty", "jabbrv-ltwa-all.ldf", "jabbrv-ltwa-en.ldf"]
FIG_RE = re.compile(r"\\includegraphics(\[[^\]]*\])?\{\.\./code/figures/paper/([^}]+)\}")


def run(cmd):
    return subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True).returncode


def compile_doc(doc, bib):
    run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
    if bib:
        run(["bibtex", doc])
    run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
    run(["pdflatex", "-interaction=nonstopmode", doc + ".tex"])
    log = open(os.path.join(BUILD, doc + ".log"), encoding="latin-1").read()
    errors = [l for l in log.splitlines() if l.startswith("!")]
    undefined = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
    pages = re.search(r"Output written on \S+ \((\d+) pages?", log)
    if errors or undefined or not pages:
        raise SystemExit(f"{doc}: build failed: {errors[:3]} undefined={undefined[:5]}")
    pdftext = subprocess.run(["pdftotext", doc + ".pdf", "-"], cwd=BUILD,
                             capture_output=True, text=True, errors="replace").stdout
    if "PENDING" in pdftext:
        raise SystemExit(f"{doc}: a [PENDING] macro reached the PDF")
    print(f"  {doc}: {pages.group(1)} pages, no errors")
    return int(pages.group(1))


def plain_abstract():
    """The manuscript abstract with numbers.tex macros expanded and LaTeX
    stripped, for pasting into Prism's abstract field."""
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}",
                             open(os.path.join(PAPER, "numbers.tex"), encoding="utf-8").read()))
    src = open(os.path.join(PAPER, "manuscript.tex"), encoding="utf-8").read()
    text = re.search(r"\\begin\{abstract\*\}(.*?)\\end\{abstract\*\}", src, re.S).group(1)
    text = re.sub(r"\\(\w+)(\{\})?", lambda m: macros.get(m.group(1), m.group(0)), text)
    for a, b in [(r"$10^\circ$", "10°"), (r"\,", " "), (r"\%", "%"), ("--", "–")]:
        text = text.replace(a, b)
    return " ".join(text.split())


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(BUILD, "figures"))
    shutil.copytree(os.path.join(PAPER, "styles"), os.path.join(BUILD, "styles"))
    for f in SUPPORT:
        shutil.copy2(os.path.join(PAPER, f), BUILD)

    used = set()
    for doc in DOCS:
        src = open(os.path.join(PAPER, doc + ".tex"), encoding="utf-8").read()
        used |= {m.group(2) for m in FIG_RE.finditer(src)}
        src = FIG_RE.sub(lambda m: r"\includegraphics" + (m.group(1) or "")
                         + "{figures/" + m.group(2) + "}", src)
        assert "../code" not in src, f"{doc}: unrewritten repository path"
        with open(os.path.join(BUILD, doc + ".tex"), "w", encoding="utf-8") as fh:
            fh.write(src)
    for fig in sorted(used):
        shutil.copy2(os.path.join(FIGS, fig), os.path.join(BUILD, "figures", fig))

    print("compiling in", BUILD)
    pages = {doc: compile_doc(doc, bib=(doc != "cover_letter")) for doc in DOCS}
    if pages["manuscript"] > 10:
        raise SystemExit(f"manuscript is {pages['manuscript']} pages; JOSA A limit is 10")

    import make_lengthcheck
    make_lengthcheck.main()
    lc = "manuscript_lengthcheck"
    src = open(os.path.join(PAPER, lc + ".tex"), encoding="utf-8").read()
    with open(os.path.join(BUILD, lc + ".tex"), "w", encoding="utf-8") as fh:
        fh.write(FIG_RE.sub(lambda m: r"\includegraphics" + (m.group(1) or "")
                            + "{figures/" + m.group(2) + "}", src))
    shutil.copy2(os.path.join(PAPER, "opticajnl.cls"), BUILD)
    shutil.copytree(os.path.join(PAPER, "legacy-styles"), os.path.join(BUILD, "legacy-styles"))
    journal_pages = compile_doc(lc, bib=True)
    if journal_pages > 10:
        raise SystemExit(f"two-column layout is {journal_pages} pages; JOSA A limit is 10")
    print(f"  journal layout: {journal_pages} pages (limit 10)")

    for doc, name in DOCS.items():
        shutil.copy2(os.path.join(BUILD, doc + ".pdf"), os.path.join(OUT, name))

    src_zip = os.path.join(OUT, "LaTeX source.zip")
    keep = {d + ext for d in DOCS for ext in (".tex", ".bbl")} | set(SUPPORT)
    with zipfile.ZipFile(src_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(keep):
            if os.path.exists(os.path.join(BUILD, f)):
                z.write(os.path.join(BUILD, f), f)
        for sub in ("figures", "styles"):
            for f in sorted(os.listdir(os.path.join(BUILD, sub))):
                z.write(os.path.join(BUILD, sub, f), f"{sub}/{f}")

    with open(os.path.join(OUT, "Abstract (plain text).txt"), "w", encoding="utf-8") as f:
        f.write(plain_abstract() + "\n")
    shutil.copy2(os.path.join(PAPER, "SUBMISSION_GUIDE.md"), OUT)
    shutil.rmtree(BUILD)
    print(f"wrote {OUT}:")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}  ({os.path.getsize(os.path.join(OUT, f)) // 1024} KB)")


if __name__ == "__main__":
    main()

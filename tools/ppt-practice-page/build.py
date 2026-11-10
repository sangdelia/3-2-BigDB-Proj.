#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["python-pptx>=0.6.21"]
# ///
"""PPTX + git 저장소 → 발표 연습 페이지(Claude artifact용 index.html + slides/).

    uv run build.py DECK.pptx --out OUT [--repo DIR] [--extras extras.json] [--target 4:30]

PPT에서 가져오는 것: 슬라이드 이미지(PDF 렌더), 장 번호, 제목, 발표자 노트(대본), 부록 여부.
git에서 가져오는 것: 원격 주소, 브랜치, 기준 커밋, 기간, 기여자별 커밋 수, 날짜별 커밋 제목.
extras.json(선택): 제목·목표 시간·질의응답·용어·원칙·수행 과정·링크·장별 덮어쓰기.
자세한 형식은 README.md.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL_RE = re.compile(r"^(?:[A-Z]\d{1,2}|\d{1,3})$")
APPENDIX_TITLE_RE = re.compile(r"^\s*(부록|appendix|backup|참고)\b", re.I)
SEC_TAG_RE = re.compile(r"^\s*[\[(]\s*(?:(\d+):(\d{2})|(\d+)\s*(?:초|s|sec))\s*[\])]\s*", re.I)


def die(msg: str) -> None:
    sys.exit(f"[build] 오류: {msg}")


def parse_seconds(v) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*(\d+):(\d{1,2})\s*", str(v))
    if m:
        return int(m[1]) * 60 + int(m[2])
    m = re.fullmatch(r"\s*(\d+)\s*(?:s|초)?\s*", str(v))
    if m:
        return int(m[1])
    die(f"시간 형식을 읽지 못했습니다: {v!r} (예: 270, 4:30)")


# ---------------------------------------------------------------- PPTX text
def shape_texts(slide):
    """(text, max_font_pt, top_ratio) for every text frame, groups included."""
    out = []

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6 and hasattr(sh, "shapes"):  # group
                walk(sh.shapes)
                continue
            if not getattr(sh, "has_text_frame", False) or not sh.text_frame.text.strip():
                continue
            size = 0.0
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size:
                        size = max(size, r.font.size.pt)
            out.append((sh.text_frame.text.strip(), size, sh))

    walk(slide.shapes)
    return out


def extract_slides(pptx_path: Path, drop_res: list[re.Pattern], include_hidden: bool):
    try:
        from pptx import Presentation
    except ImportError:
        die("python-pptx가 필요합니다. `uv run build.py ...` 또는 `pip install python-pptx`")
    prs = Presentation(str(pptx_path))
    H = prs.slide_height or 1
    raw = []
    for i, s in enumerate(prs.slides, 1):
        hidden = s._element.get("show") == "0"
        texts = shape_texts(s)
        title_ph = s.shapes.title.text.strip() if s.shapes.title is not None and s.shapes.title.text.strip() else None
        notes = s.notes_slide.notes_text_frame.text if s.has_notes_slide and s.notes_slide.notes_text_frame else ""
        raw.append({"pos": i, "hidden": hidden, "texts": texts, "title_ph": title_ph, "notes": notes})

    # Text that repeats on many slides is a header/footer, not a title or label.
    cnt = collections.Counter(t for r in raw for t in {x[0] for x in r["texts"]})
    common = {t for t, c in cnt.items() if len(raw) >= 4 and c >= max(3, len(raw) * 0.5)}

    slides = []
    for r in raw:
        if r["hidden"] and not include_hidden:
            continue
        label = None
        cands = []
        for text, size, sh in r["texts"]:
            if text in common:
                continue
            if LABEL_RE.match(text):
                label = label or text
                continue
            top = (sh.top or 0) / H
            cands.append((size, -top, text))
        title = r["title_ph"]
        if not title and cands:
            # largest font in the upper 60% of the slide; ties → higher on the slide
            upper = [c for c in cands if -c[1] < 0.6] or cands
            title = max(upper)[2]
        title = re.sub(r"\s*\n\s*", " ", title or "").strip() or f"슬라이드 {r['pos']}"
        script, sec_tag = clean_notes(r["notes"], drop_res)
        slides.append({"pos": r["pos"], "label": label, "title": title, "script": script, "sec_tag": sec_tag})
    return slides, len(raw), prs


def clean_notes(text: str, drop_res: list[re.Pattern]):
    text = (text or "").replace("\r", "").replace("\x0b", "\n")
    sec = None
    m = SEC_TAG_RE.match(text)
    if m:
        sec = int(m[1]) * 60 + int(m[2]) if m[1] else int(m[3])
        text = text[m.end():]
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not any(rx.search(ln) for rx in drop_res)]
    return "\n".join(lines).strip(), sec


# ---------------------------------------------------------------- images
def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def pdf_pages(pdf: Path) -> int:
    try:
        out = run(["pdfinfo", str(pdf)]).stdout
        return int(re.search(r"^Pages:\s+(\d+)", out, re.M)[1])
    except Exception:
        return -1


def render_pdf(pptx: Path, tmp: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        die("LibreOffice(soffice)가 없습니다. --pdf로 PDF를 직접 주거나 LibreOffice를 설치하세요.")
    print("[build] LibreOffice로 PDF 변환 중…", file=sys.stderr)
    run([soffice, "--headless", f"-env:UserInstallation=file://{tmp}/lo", "--convert-to", "pdf", "--outdir", str(tmp), str(pptx)])
    pdf = tmp / (pptx.stem + ".pdf")
    if not pdf.exists():
        die("PDF 변환 결과가 없습니다.")
    return pdf


def pick_pdf(pptx: Path, given: str | None, n_all: int, n_visible: int, tmp: Path, use_sibling: bool) -> Path:
    if given:
        return Path(given)
    sib = pptx.with_suffix(".pdf")
    if use_sibling and sib.exists() and pdf_pages(sib) in (n_all, n_visible):
        if sib.stat().st_mtime + 1 < pptx.stat().st_mtime:
            print(f"[build] 주의: {sib.name}이 PPTX보다 오래됐습니다. 최신 렌더가 필요하면 --no-sibling-pdf", file=sys.stderr)
        print(f"[build] 같은 이름의 PDF 사용: {sib.name}", file=sys.stderr)
        return sib
    return render_pdf(pptx, tmp)


def make_images(pdf: Path, out_slides: Path, width: int, quality: int) -> list[Path]:
    if not shutil.which("pdftoppm"):
        die("pdftoppm(poppler-utils)이 필요합니다.")
    out_slides.mkdir(parents=True, exist_ok=True)
    for old in out_slides.glob("s*.jpg"):
        old.unlink()
    run(["pdftoppm", "-jpeg", "-jpegopt", f"quality={quality}", "-scale-to-x", str(width), "-scale-to-y", "-1", str(pdf), str(out_slides / "p")])
    pages = sorted(out_slides.glob("p-*.jpg"), key=lambda p: int(p.stem.split("-")[1]))
    return pages


# ---------------------------------------------------------------- git
def git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    except Exception:
        return ""


def web_url(remote: str):
    r = remote.strip().rstrip("/").rstrip(".")
    r = re.sub(r"\.git$", "", r)
    m = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", r)
    if m:
        r = f"https://{m[1]}/{m[2]}"
    r = re.sub(r"^https?://[^@/]+@", "https://", r)  # drop credentials
    if not r.startswith("http"):
        return None, None
    host = "github" if "github.com" in r else "gitlab" if "gitlab" in r else "other"
    return r, host


def collect_git(repo: Path, deck: Path, max_commits: int, author_map: dict) -> dict | None:
    top = git(repo, "rev-parse", "--show-toplevel")
    if not top:
        return None
    top = Path(top)
    info: dict = {"name": top.name}
    remote = git(top, "remote", "get-url", "origin") or (git(top, "remote").split() or [""])[0]
    if remote and not remote.startswith(("http", "git@", "ssh")):
        remote = git(top, "remote", "get-url", remote)
    if remote:
        info["web"], info["host"] = web_url(remote)
        if info["web"]:
            info["name"] = info["web"].rsplit("/", 1)[-1]
    info["branch"] = git(top, "rev-parse", "--abbrev-ref", "HEAD")
    info["head_full"] = git(top, "rev-parse", "HEAD")
    info["head"] = info["head_full"][:7]
    info["dirty"] = bool(git(top, "status", "--porcelain"))
    try:
        info["deck_path"] = str(deck.resolve().relative_to(top.resolve()))
    except ValueError:
        pass
    log = git(top, "log", "--no-merges", "--date=short", "--format=%h\x1f%ad\x1f%an\x1f%s")
    commits = []
    for line in log.splitlines():
        sha, date, author, subject = (line.split("\x1f") + ["", "", "", ""])[:4]
        commits.append({"sha": sha, "date": date, "author": author_map.get(author, author), "subject": subject})
    info["commits_total"] = len(commits)
    if commits:
        info["first_date"], info["last_date"] = commits[-1]["date"], commits[0]["date"]
    info["contributors"] = [{"name": n, "commits": c} for n, c in collections.Counter(c["author"] for c in commits).most_common()]
    single = len(info["contributors"]) <= 1
    by_date = collections.OrderedDict()
    for c in commits[:max_commits]:
        by_date.setdefault(c["date"], []).append({k: v for k, v in c.items() if k != "date" and not (single and k == "author")})
    info["by_date"] = [{"date": d, "commits": cs} for d, cs in by_date.items()]
    return info


# ---------------------------------------------------------------- assemble
def assign_labels(slides, appendix_from: str | None):
    """Main slides get 01, 02 …; appendix slides A1, A2 … unless the deck shows its own labels."""
    apx_started = False
    for s in slides:
        lab = s["label"]
        is_apx = bool(lab and lab[0].isalpha()) or bool(APPENDIX_TITLE_RE.match(s["title"]))
        if appendix_from and (lab == appendix_from or str(s["pos"]) == appendix_from):
            apx_started = True
        apx_started = apx_started or is_apx
        s["appendix"] = apx_started
    n_main = n_apx = 0
    seen = set()
    for s in slides:
        if s["appendix"]:
            n_apx += 1
            auto = f"A{n_apx}"
        else:
            n_main += 1
            auto = f"{n_main:02d}"
        lab = s["label"] if s["label"] and s["label"] not in seen else auto
        if not s["appendix"] and lab.isdigit():
            lab = f"{int(lab):02d}"
        while lab in seen:
            lab += "′"
        seen.add(lab)
        s["label"] = lab


def plan_seconds(main, target: int | None, rate: float):
    """Per-slide target: explicit tag in notes > extras override > share of the total by script length."""
    fixed = sum(s["sec"] for s in main if s.get("sec"))
    free = [s for s in main if not s.get("sec")]
    if not free:
        return
    if target:
        budget = max(len(free) * 3, target - fixed)
        weights = [max(len(s["script"]), 15) for s in free]
        tot = sum(weights)
        raw = [budget * w / tot for w in weights]
        secs = [max(3, int(x)) for x in raw]
        rest = budget - sum(secs)
        order = sorted(range(len(free)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in order[: max(0, rest)]:
            secs[i] += 1
        for s, v in zip(free, secs):
            s["sec"] = v
    else:
        for s in free:
            s["sec"] = max(5, round(len(s["script"]) / rate))


def main():
    ap = argparse.ArgumentParser(description="PPTX + git → 발표 연습 페이지 (Claude artifact)")
    ap.add_argument("pptx", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="출력 폴더 (index.html, slides/, deck_data.json)")
    ap.add_argument("--repo", type=Path, help="git 저장소 (기본: PPTX가 있는 폴더)")
    ap.add_argument("--extras", type=Path, help="질의응답·용어·수행 과정 등 추가 내용 JSON")
    ap.add_argument("--pdf", help="이미 있는 PDF로 이미지 생성 (LibreOffice 변환 생략)")
    ap.add_argument("--no-sibling-pdf", action="store_true", help="PPTX 옆 같은 이름 PDF를 쓰지 않고 새로 변환")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--title"); ap.add_argument("--subtitle")
    ap.add_argument("--target", help="본문 목표 시간 (예: 4:30, 270). 없으면 노트 길이 ÷ 속도")
    ap.add_argument("--limit", help="발표 제한 시간 (예: 5:00)")
    ap.add_argument("--rate", type=float, default=None, help="말하기 속도 자/초 (기본 5.5, --target이 있으면 대본에서 계산)")
    ap.add_argument("--appendix-from", help="이 장 번호/라벨부터 부록으로 처리")
    ap.add_argument("--drop", action="append", default=[], help="노트에서 지울 줄의 정규식 (여러 번)")
    ap.add_argument("--include-hidden", action="store_true", help="숨긴 슬라이드도 포함")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--max-commits", type=int, default=150)
    a = ap.parse_args()

    if not a.pptx.exists():
        die(f"PPTX가 없습니다: {a.pptx}")
    extras = json.loads(a.extras.read_text(encoding="utf-8")) if a.extras else {}
    meta_x = extras.get("meta", {})
    drop_res = [re.compile(p) for p in a.drop + extras.get("notes_drop", [])]

    slides, n_all, _ = extract_slides(a.pptx, drop_res, a.include_hidden)
    if not slides:
        die("슬라이드가 없습니다.")
    n_visible = len(slides)
    assign_labels(slides, a.appendix_from or meta_x.get("appendix_from"))

    # per-slide overrides from extras: {"03": {"title": …, "sec": 20, "script": …}}
    over = extras.get("slides", {})
    for s in slides:
        o = over.get(s["label"]) or over.get(str(s["pos"])) or {}
        for k in ("title", "script"):
            if k in o:
                s[k] = o[k]
        s["sec"] = parse_seconds(o.get("sec")) or s.pop("sec_tag", None)
        s.pop("sec_tag", None)

    target = parse_seconds(a.target or meta_x.get("target"))
    limit = parse_seconds(a.limit or meta_x.get("limit"))
    main_s = [s for s in slides if not s["appendix"]]
    apx_s = [s for s in slides if s["appendix"]]
    rate = a.rate or meta_x.get("rate")
    plan_seconds(main_s, target, rate or 5.5)
    if not rate:
        tot_sec = sum(s["sec"] for s in main_s) or 1
        rate = round(sum(len(s["script"]) for s in main_s) / tot_sec, 2) or 5.5
        rate = min(12.0, max(3.0, rate))  # 노트가 요점뿐이면 비현실적인 속도가 나오므로 범위 제한

    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pdf = pick_pdf(a.pptx, a.pdf, n_all, n_visible, Path(td), not a.no_sibling_pdf)
        pages = make_images(pdf, a.out / "slides", a.width, a.quality)
    if len(pages) == n_all:
        page_of = {s["pos"]: pages[s["pos"] - 1] for s in slides}
    elif len(pages) == n_visible:
        page_of = {s["pos"]: p for s, p in zip(slides, pages)}
    else:
        die(f"PDF 쪽수({len(pages)})가 슬라이드 수({n_all}, 표시 {n_visible})와 맞지 않습니다.")
    for s in slides:
        name = f"s{s['pos']:02d}.jpg"
        page_of[s["pos"]].rename(a.out / "slides" / name)
        s["img"] = f"slides/{name}"
    for left in (a.out / "slides").glob("p-*.jpg"):
        left.unlink()

    cum = 0
    main_out = []
    for s in main_s:
        cum += s["sec"]
        main_out.append({"no": s["pos"], "label": s["label"], "title": s["title"], "sec": s["sec"], "cum": cum, "script": s["script"], "img": s["img"]})
    apx_out = [{"no": s["pos"], "label": s["label"], "title": s["title"], "script": s["script"], "img": s["img"]} for s in apx_s]

    gi = None
    if not a.no_git:
        gi = collect_git(a.repo or a.pptx.resolve().parent, a.pptx, a.max_commits, extras.get("author_names", {}))
        if gi is None:
            print("[build] git 저장소를 찾지 못해 저장소 정보 없이 만듭니다.", file=sys.stderr)

    title = a.title or meta_x.get("title") or (main_out[0]["title"] if main_out else a.pptx.stem) + " 발표 연습"
    data = {
        "meta": {
            "title": title,
            "subtitle": a.subtitle or meta_x.get("subtitle") or "",
            "limit_sec": limit, "rate": rate, "deck": a.pptx.name,
            "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            **({"process_lead": meta_x["process_lead"]} if meta_x.get("process_lead") else {}),
        },
        "main": main_out, "appendix": apx_out,
        "qa": extras.get("qa", []), "glossary": extras.get("glossary", []),
        "rules": extras.get("rules", []), "timeline": extras.get("timeline", []),
        "links": extras.get("links", []), "git": gi,
    }
    (a.out / "deck_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    page = tpl.replace("/*TITLE*/", html.escape(title), 1).replace("/*DATA*/", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"), 1)
    (a.out / "index.html").write_text(page, encoding="utf-8")

    no_notes = [s["label"] for s in slides if not s["script"]]
    size = sum(p.stat().st_size for p in (a.out / "slides").glob("*.jpg"))
    print(f"[build] 본문 {len(main_out)}장 + 부록 {len(apx_out)}장 · 목표 {cum // 60}:{cum % 60:02d} · 속도 {rate}자/초 · 이미지 {size / 1e6:.1f}MB")
    if no_notes:
        print(f"[build] 노트 없는 장: {', '.join(no_notes)}")
    print(f"[build] 출력: {a.out.resolve()}/index.html")
    files = {s["img"]: str((a.out / s["img"]).resolve()) for s in slides}
    (a.out / "publish.json").write_text(json.dumps({"file_path": str((a.out / "index.html").resolve()), "files": files, "capabilities": {"db": {}}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build] Artifact 게시 인자: {a.out.resolve()}/publish.json")


if __name__ == "__main__":
    main()

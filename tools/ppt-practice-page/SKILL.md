---
name: ppt-practice-page
description: Turn a PPTX deck plus its git repository into a presentation-practice web page (slides, speaker-note scripts with timers, editable scripts, optional Q&A/glossary, git-based project timeline) and publish it as a Claude artifact. Use when the user asks for a 발표 연습 페이지 / practice page / rehearsal page from a PPT, or to rebuild one for another project.
---

# PPT → 발표 연습 페이지

Files next to this SKILL.md: `build.py` (generator), `template.html` (page), `README.md` (full options and the extras.json format). The README and the generated page's UI are written in Korean.

## Steps

1. **Find the deck and repo.** Ask for the PPTX path only if it cannot be found. By default the repo is the git repo that contains the deck (`--repo` to override). If a same-name PDF sits next to the PPTX, it is used for images when the page counts match.

2. **Dry build** to see what the deck yields:
   ```bash
   uv run <skill-dir>/build.py DECK.pptx --out <scratchpad>/practice [--target 4:30 --limit 5:00]
   ```
   Then check `deck_data.json`:
   - Are titles and labels right?
   - Where does the appendix start?
   - Which slides have no notes? The build prints them.
   - Do the notes carry boilerplate lines that should be dropped (`--drop REGEX` / `notes_drop`)?

3. **Optional `extras.json`.** Write one when the user wants Q&A, a glossary, working rules, a curated timeline, or a nicer title. Draw only from the deck, its notes and the repo's docs and results. Never invent numbers. Use slide labels as they appear in `deck_data.json` for `refs`. The schema is in README.md. Keep it next to the deck in the project, e.g. `ppts/presenter/extras.json`, so it can be rebuilt.

4. **Rebuild** with `--extras`, and look at the page:
   ```bash
   google-chrome --headless=new --no-sandbox --screenshot=… --window-size=1400,1000 file://…/index.html#practice
   ```
   Also capture `#script`, `#process`, and a 400px-wide view.

5. **Publish** using `OUT/publish.json`:
   - Artifact `file_path` = index.html
   - `files` = the slides map
   - `capabilities` = `{"db": {}}` on the first publish
   - `icon`: "slides"

   To update an existing practice page, pass its `url` (read it first) and omit `capabilities`. Edited scripts live in the artifact's db collection `scripts` (doc ids `m<label>` / `a<label>`). They survive republishes when labels stay the same.

## Notes

- Don't commit generated `OUT/` images unless the user wants them in the repo. They are reproducible from the PPTX.
- If speaker notes are bullet summaries rather than a spoken script, say so. The per-slide timer is then only a rough guide: pass `--target` so the total matches the real slot.
- Git info includes author names but never emails. Remote URLs are stripped of credentials.

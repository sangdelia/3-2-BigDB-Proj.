"""Post-midterm revision (feedback 11/04): problem definition, label→target conversion, text→numeric features,
model formulas, confusion matrix, baselines, learning curve, HDFS–Spark pipeline, false-positive hold rule."""
import copy, json, re, sys
from xml.sax.saxutils import escape
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

SRC, DST, LC, FP = sys.argv[1:5]
prs = Presentation(SRC)
S = list(prs.slides)
assert len(S) == 20

BLUE, LIGHT, DARK = RGBColor(0x65, 0x7E, 0xA7), RGBColor(0xB8, 0xC4, 0xD8), RGBColor(0x3E, 0x4C, 0x66)
INK, MUTED, WARM = RGBColor(0x33, 0x33, 0x33), RGBColor(0x62, 0x62, 0x62), RGBColor(0xC0, 0x69, 0x4F)
PALE, GREYF, GREYL = RGBColor(0xEA, 0xF0, 0xF8), RGBColor(0xF5, 0xF5, 0xF5), RGBColor(0xD0, 0xD0, 0xD0)
FONT = "Noto Sans CJK KR"
X0, W, TOP, BOTTOM = 857250, 9925050, 2520000, 5130000
ROOT = "/home/kyu/BigDB_Proj/"

# ------------------------------------------------------------------ text helpers
def ea(rpr):
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {}); rpr.append(el)
        el.set("typeface", FONT)

def style_run(r, size, bold=False, color=INK):
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    ea(r._r.get_or_add_rPr())

TOK = re.compile(r"(\*\*.+?\*\*|!!.+?!!)")
def rich(p, s, size, color=INK, bold=False):
    """**x** = accent bold, !!x!! = warm bold."""
    for part in TOK.split(s):
        if not part:
            continue
        r = p.add_run()
        if part.startswith("**"):
            r.text = part[2:-2]; style_run(r, size, True, BLUE)
        elif part.startswith("!!"):
            r.text = part[2:-2]; style_run(r, size, True, WARM)
        else:
            r.text = part; style_run(r, size, bold, color)

def named(slide, name):
    hits = [sh for sh in slide.shapes if sh.name == name]
    assert len(hits) == 1, (name, len(hits)); return hits[0]

def put(slide, name, s):
    """Replace a placeholder-like shape's text, keeping its first run's size/color/bold."""
    tf = named(slide, name).text_frame
    r0 = tf.paragraphs[0].runs[0]
    size = r0.font.size.pt; bold = bool(r0.font.bold)
    color = r0.font.color.rgb if r0.font.color and r0.font.color.type is not None else INK
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    p = tf.paragraphs[0]
    proto = copy.deepcopy(r0._r.get_or_add_rPr())
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    for part in TOK.split(s):
        if not part: continue
        r = p.add_run(); rp = r._r.get_or_add_rPr(); rp.getparent().replace(rp, copy.deepcopy(proto))
        if part[:2] in ("**", "!!"):
            r.text = part[2:-2]; r.font.bold = True; r.font.color.rgb = BLUE if part[:2] == "**" else WARM
        else:
            r.text = part

def box(slide, x, y, w, h, lines, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, margin=110000, space=3):
    sh = slide.shapes.add_shape(shape, Emu(x), Emu(y), Emu(w), Emu(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.06
    if fill is None: sh.fill.background()
    else: sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(margin); tf.margin_top = tf.margin_bottom = Emu(int(margin * 0.6))
    for k, (s, size, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        rich(p, s, size, color, bold)
    return sh

def text(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=3):
    return box(slide, x, y, w, h, lines, anchor=anchor, align=align, margin=0, shape=MSO_SHAPE.RECTANGLE, space=space)

def arrow(slide, x, y, w=200000, h=220000, down=False):
    a = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW, Emu(x), Emu(y), Emu(w), Emu(h))
    a.fill.solid(); a.fill.fore_color.rgb = LIGHT; a.line.fill.background(); a.shadow.inherit = False

def chart_frame(c):
    cs = c._chartSpace
    for old in cs.findall(qn("c:spPr")): cs.remove(old)
    sp = cs.makeelement(qn("c:spPr"), {})
    sp.append(sp.makeelement(qn("a:noFill"), {}))
    ln = sp.makeelement(qn("a:ln"), {}); ln.append(ln.makeelement(qn("a:noFill"), {})); sp.append(ln)
    cs.find(qn("c:chart")).addnext(sp)

def chart(slide, x, y, w, h, cats, series, kind, fmt, title=None, vmax=None, vmin=0, legend=False, point_colors=None,
          label_size=11, cat_size=11, line=False):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _ in series: cd.add_series(name, vals)
    c = slide.shapes.add_chart(kind, Emu(x), Emu(y), Emu(w), Emu(h), cd).chart
    c.font.size = Pt(cat_size); c.font.color.rgb = INK; ea(c.font._rPr)
    c.has_title = bool(title)
    if title:
        tf = c.chart_title.text_frame; tf.text = title; style_run(tf.paragraphs[0].runs[0], 13, True, INK)
    c.has_legend = legend
    if legend:
        c.legend.position = XL_LEGEND_POSITION.TOP; c.legend.include_in_layout = False
        c.legend.font.size = Pt(11); ea(c.legend.font._rPr)
    plot = c.plots[0]
    if not line:
        plot.gap_width = 60; plot.overlap = -10 if len(series) > 1 else 0
    for s, (_, _, color) in zip(plot.series, series):
        if line:
            s.format.line.color.rgb = color; s.format.line.width = Pt(2.5); s.smooth = False
            s.marker.style = 8; s.marker.size = 8
            s.marker.format.fill.solid(); s.marker.format.fill.fore_color.rgb = color; s.marker.format.line.color.rgb = color
        else:
            s.format.fill.solid(); s.format.fill.fore_color.rgb = color; s.invert_if_negative = False
    if point_colors:
        for idx, col in point_colors.items():
            pt = plot.series[0].points[idx]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = col
    plot.has_data_labels = True
    dl = plot.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(label_size); dl.font.bold = True; ea(dl.font._rPr)
    dl.position = XL_LABEL_POSITION.ABOVE if line else XL_LABEL_POSITION.OUTSIDE_END
    va = c.value_axis; va.visible = False; va.has_major_gridlines = False
    va.minimum_scale = vmin
    if vmax: va.maximum_scale = vmax
    ca = c.category_axis; ca.format.line.color.rgb = GREYL; ca.has_major_gridlines = False
    ca.tick_labels.font.size = Pt(cat_size); ea(ca.tick_labels.font._rPr)
    if line and len(series) > 1:
        s0 = plot.series[0]; s0.data_labels.position = XL_LABEL_POSITION.BELOW
        s0.data_labels.number_format = fmt; s0.data_labels.number_format_is_linked = False
        s0.data_labels.font.size = Pt(label_size - 1); s0.data_labels.font.color.rgb = MUTED; ea(s0.data_labels.font._rPr)
    chart_frame(c)
    return c

# ------------------------------------------------------------------ tables (same look as existing deck tables)
LN = '<a:{s} w="6191"><a:solidFill><a:srgbClr val="444444" /></a:solidFill><a:prstDash val="solid" /></a:{s}>'
def cell(t, header, first, align, sz):
    color = "333333"
    bold = header
    if t.startswith("**"): bold, t = True, t[2:]
    if t.startswith("!!"): bold, color, t = True, "C0694F", t[2:]
    if t.startswith("^^"): bold, color, t = True, "657EA7", t[2:]
    lines = ''.join(LN.format(s=s) for s in (["lnL"] if first else []) + ["lnR"] + (["lnT"] if header else []) + ["lnB"])
    fill = "E8E8E8" if header else "FFFFFF"
    paras = ''.join(
        f'<a:p><a:pPr algn="{align}" /><a:r><a:rPr lang="ko-KR" sz="{sz}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}" /></a:solidFill>'
        f'<a:latin typeface="{FONT}" /><a:ea typeface="{FONT}" /><a:cs typeface="{FONT}" /></a:rPr><a:t>{escape(line)}</a:t></a:r></a:p>'
        for line in t.split("\n"))
    return ('<a:tc><a:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"><a:noAutofit /></a:bodyPr><a:lstStyle />'
            f'{paras}</a:txBody><a:tcPr marL="72000" marR="72000" marT="36000" marB="36000" anchor="ctr">{lines}'
            f'<a:solidFill><a:srgbClr val="{fill}" /></a:solidFill></a:tcPr></a:tc>')

TABLE_TPL = etree.tostring([sh for sh in S[6].shapes if sh.has_table][0]._element, encoding="unicode")
def table(slide, x, y, rows, widths, sz=1200, align=None, rowh=380000):
    align = align or ["l"] + ["ctr"] * (len(widths) - 1)
    assert all(len(r) == len(widths) for r in rows)
    g = TABLE_TPL
    trs = ''.join(f'<a:tr h="{rowh}">' + ''.join(cell(c, i == 0, j == 0, align[j], sz) for j, c in enumerate(r)) + '</a:tr>'
                  for i, r in enumerate(rows))
    grid = '<a:tblGrid>' + ''.join(f'<a:gridCol w="{w}" />' for w in widths) + '</a:tblGrid>'
    g = re.sub(r'<a:tblGrid>.*?</a:tblGrid>', grid, g, flags=re.S)
    g = re.sub(r'(</a:tblGrid>).*(</a:tbl>)', lambda m: m.group(1) + trs + m.group(2), g, flags=re.S)
    g = re.sub(r'<p:xfrm>.*?</p:xfrm>', f'<p:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{sum(widths)}" cy="{rowh * len(rows)}" /></p:xfrm>', g, flags=re.S)
    el = parse_xml(g)
    slide.shapes._spTree.append(el)
    return el

# ------------------------------------------------------------------ slide factory
KEEP = {"reference-backsheet", "reference-paper", "reference-bottomfold", "title-band", "title-rule-top",
        "title-rule-bottom", "sidebar", "page-number", "headline", "subtitle", "body-summary", "qualification"}
BASE = S[2]  # EDA slide: frame + text slots
def new_slide(title, sub, body, qual):
    s = prs.slides.add_slide(BASE.slide_layout)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in BASE.shapes:
        if sh.name in KEEP:
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    put(s, "headline", title); put(s, "subtitle", sub)
    if body is None: named(s, "body-summary")._element.getparent().remove(named(s, "body-summary")._element)
    else: put(s, "body-summary", body)
    if qual is None: named(s, "qualification")._element.getparent().remove(named(s, "qualification")._element)
    else: put(s, "qualification", qual)
    return s

def notes(slide, talk, sources):
    slide.notes_slide.notes_text_frame.text = "\n".join(
        ["발표 요점", talk, "", "근거 파일(" + ROOT + "): " + " · ".join(sources)])

lc = {round(json.load(open(f"{LC}/lc_{f}.json"))["train_fraction"], 2): json.load(open(f"{LC}/lc_{f}.json")) for f in ("0.05", "0.1", "0.25", "0.5", "1.0")}
maj = json.load(open(f"{LC}/majority.json"))["majority_all_harmful"]
fp = json.load(open(f"{FP}/fp_review3.json"))
tb, tsel, sel = fp["test_base"], fp["test_selected"], fp["selected"]
pct = lambda v: f"{v * 100:.2f}%"

# =================================================================== NEW slides
# N2 problem definition
n2 = new_slide("문제 정의: 한국어 문장의 유해 여부 **분류**",
               "과제 = 정답이 있는 지도학습 · 주 과제는 **분류(모델 A)**, 보조 과제는 **점수 예측(모델 B)**",
               "EDA로 현황 파악 → 라벨을 **정답(0/1)**으로, 문장을 **숫자 속성(벡터)**으로 변환 → 학습·평가",
               "초기 제안(임베딩 Top-K 유사 사례 검색)은 후속 작업 · 언어 구조·데이터 양·라벨 체계 차이로 한국어 6종에 집중")
bw, gap = 2250000, 290000
xs = [X0 + i * (bw + gap) for i in range(4)]
y1, bh = TOP + 30000, 1000000
box(n2, xs[0], y1, bw, bh, [("입력", 12, True, MUTED), ("한국어 문장 1개", 15, True, INK), ("1–1000자 · 한글 포함", 11, False, MUTED)], fill=GREYF, line=GREYL, anchor=MSO_ANCHOR.MIDDLE)
arrow(n2, xs[0] + bw + 45000, y1 + bh // 2 - 110000)
box(n2, xs[1], y1, bw, bh, [("숫자 속성 x", 12, True, MUTED), ("문자 2–5gram TF-IDF", 14, True, BLUE), ("문장 → 숫자 벡터 (6장)", 11, False, MUTED)], fill=PALE, line=BLUE, anchor=MSO_ANCHOR.MIDDLE)
arrow(n2, xs[1] + bw + 45000, y1 + bh // 2 - 110000)
box(n2, xs[2], y1, bw, bh, [("모델 A · 분류", 12, True, MUTED), ("정상 / 유해", 15, True, BLUE), ("6종 통합 데이터로 학습", 11, False, MUTED)], fill=PALE, line=BLUE, lw=1.6, anchor=MSO_ANCHOR.MIDDLE)
box(n2, xs[3], y1, bw, bh, [("모델 B · 점수 예측", 12, True, MUTED), ("Hate·공격성·욕설 0–100", 13, True, INK), ("K-HATERS 평점으로 학습", 11, False, MUTED)], fill=GREYF, line=GREYL, anchor=MSO_ANCHOR.MIDDLE)
table(n2, X0, y1 + bh + 170000, [
    ["구분", "과제 유형", "정답 y", "평가 지표"],
    ["모델 A", "**이진 분류", "정상 0 / 유해 1 (6종 라벨 변환, 311,661행)", "Macro-F1 · 유해 재현율 · 정상 오탐률"],
    ["모델 B", "3등급 분류 → 점수 예측", "평점 0·1·2 (K-HATERS 172,097행)", "MAE(0–100점) · 등급 Macro-F1"],
], [1300000, 2300000, 3725050, 2600000], sz=1150, rowh=330000)
notes(n2, "피드백(무엇을 하려는지)에 대한 답. 이 프로젝트는 문장을 입력받아 정상/유해를 분류하는 지도학습이 주 과제이고, 같은 입력에 대해 Hate·공격성·욕설 점수를 예측하는 모델 B를 보조로 둔다. 모든 단계는 '정답 만들기 → 숫자 속성 만들기 → 학습 → 평가' 순서.",
      ["docs/KOREAN_EXPERIMENT_PLAN.md", "docs/KHATERS_HATE_SCORE.md"])

# N3 label structure
n3 = new_slide("6종 라벨 구조와 **정답(0/1)** 변환",
               "출처마다 라벨 체계가 달라 **정상 = 0 / 유해 = 1**로 통일 · 같은 데이터 안에서 모순된 라벨은 제외",
               "정답 y: 정상 **115,328행 (37.0%)** · 유해 **196,333행 (63.0%)** — 합계 311,661행",
               "세부 유형·대상·표현 위치는 이진 학습에 쓰지 않음 · 출처별 '유해' 기준 차이는 E4에서 분석")
table(n3, X0, TOP - 40000, [
    ["출처", "원래 라벨 체계", "정상 = 0", "유해 = 1", "제외"],
    ["NIKL", "대화 맥락 + 대상 발화의 적절/부적절", "적절", "부적절", "—"],
    ["UnSmile", "혐오 유형 9개·개인 지칭 다중 표시 + '깨끗함'", "깨끗함만 표시", "혐오 유형 1개 이상", "!!깨끗함+혐오 동시"],
    ["KOLD", "뉴스 댓글 공격 여부 + 대상·표현 위치", "공격 아님", "공격적", "—"],
    ["K-MHaS", "혐오 유형 번호 다중(0–7) + 혐오 아님(8)", "8만", "0–7만", "!!8과 0–7 혼재"],
    ["K-HATERS", "4단계: 정상·공격적·약한 혐오·강한 혐오", "정상", "공격적·약한·강한", "그 외 라벨·빈 문장"],
    ["BEEP", "연예 댓글 3단계: 없음·공격적·혐오", "없음", "공격적·혐오", "—"],
], [1200000, 3725050, 1550000, 1750000, 1700000], sz=1100, rowh=355000)
notes(n3, "피드백(정답 데이터를 만들어 학습)에 대한 답. 여섯 데이터의 라벨 체계를 하나의 정답 y∈{0,1}로 바꿨다. 같은 출처 안에서 '깨끗함'과 '혐오'가 동시에 붙는 등 모순되면 제외했다. 이 y가 모델 A의 학습·평가 정답이다.",
      ["src/bigdb_mvp/data/arrow_sources.py", "src/bigdb_mvp/data/remote_sources.py", "src/bigdb_mvp/data/normalize.py", "reports/eda_20261029/hadoop/aggregates.tsv"])

# N6 text -> numeric features
n6 = new_slide("문장을 **숫자 속성**으로: 문자 n-gram TF-IDF",
               "예: '너는 정말 멍청한 바보야.' → 2–5글자 조각 46개 → 사전(20만)에 있는 **28개만 0이 아닌 값**",
               "정답 y는 0/1, 속성 x는 **200,000차원 숫자 벡터** → 모델 A·B의 공통 입력 형식",
               "출처·그룹 ID·원래 세부 라벨은 속성으로 쓰지 않음 · 조각 사전과 idf는 train에서만 계산")
fw = 4500000
box(n6, X0, TOP, fw, 2560000, [
    ("① 조각 빈도  tf(t, d) = 문장 d에서 조각 t의 횟수", 13, False, INK),
    ("② TF-IDF", 13, True, MUTED), ("**x_t = (1 + ln tf) × idf(t)**", 16, False, INK),
    ("idf(t) = ln((1 + N) / (1 + df(t))) + 1", 14, False, INK), ("N = train 249,732문장, df = 조각이 나온 문장 수", 11.5, False, MUTED),
    ("③ 길이 정규화  x ← x / ‖x‖₂", 13, False, INK),
], fill=GREYF, line=GREYL, space=7, anchor=MSO_ANCHOR.MIDDLE)
table(n6, X0 + fw + 250000, TOP, [
    ["조각 t", "x_t (TF-IDF)", "w_t (모델 A)", "x_t · w_t"],
    ["멍청", "0.190", "4.273", "^^0.812"],
    ["␣멍", "0.187", "2.698", "^^0.505"],
    ["바보", "0.195", "2.510", "^^0.489"],
    ["멍청한", "0.208", "1.832", "^^0.381"],
    ["청한", "0.206", "1.764", "^^0.362"],
    ["… 나머지 23개", "", "", "…"],
    ["**합계 Σ x·w", "", "", "**5.733"],
], [1450000, 1250000, 1250000, 1225050], sz=1150, rowh=320000)
notes(n6, "피드백(속성을 숫자로 정리해 표현)에 대한 답. 문장을 2–5글자 조각으로 나누고 TF-IDF로 숫자를 매긴 벡터가 속성 x다. 오른쪽 표는 실제 모델에서 이 문장의 조각별 값과 가중치이며, 합 5.733이 다음 장 수식의 wᵀx가 된다. 띄어쓰기(␣)도 조각에 포함된다.",
      ["artifacts/korean_short_v1/char_2_5_200000.joblib", "src/bigdb_mvp/experiments/classification.py"])

# N9 HDFS–Spark pipeline
n9 = new_slide("HDFS–Spark **연동** 파이프라인",
               "중간발표: 두 도구를 따로 실행 → 보완: **Spark가 HDFS에서 읽고 HDFS에 쓰도록** 연결",
               "Bronze → Silver(**학습 데이터용 정제**) → Gold(집계) 모두 HDFS · 단계마다 Python과 대조",
               "!!구성 완료 · 전체 실행 검증은 재실행 후 반영!! · 단일 서버 · 다중 서버 확장성은 미측정")
steps = [("Bronze · HDFS", "원천 316,571행\nParquet 적재", "HDFS 해시 = 로컬 해시"),
         ("Silver · Spark", "HDFS에서 읽어\n정규화·충돌 제외·중복 병합", "출력 키 해시 = Python"),
         ("Gold · Spark SQL", "Silver를 읽어\n출처·라벨·길이 27개 키", "Spark 집계 = Python"),
         ("Gold · MapReduce", "HDFS Silver 입력\nMap → Combine → Reduce", "MapReduce = Spark = Python")]
sw, sg = 2250000, 308350
for i, (h, d, v) in enumerate(steps):
    x = X0 + i * (sw + sg)
    hi = i in (1, 3)
    box(n9, x, TOP + 20000, sw, 1650000, [(h, 14, True, BLUE if hi else INK)] + [(ln, 12.5, False, INK) for ln in d.split("\n")]
        + [("검증: " + v, 11.5, True, BLUE)], fill=PALE if hi else GREYF, line=BLUE if hi else GREYL, lw=1.4 if hi else 1.0, anchor=MSO_ANCHOR.MIDDLE, space=5)
    if i < 3: arrow(n9, x + sw + 50000, TOP + 20000 + 825000 - 110000)
text(n9, X0, TOP + 1800000, W, 700000, [
    ("hdfs://…/bigdb/pipeline/  bronze/raw_input.parquet  →  silver/corpus/ · silver/mr_input/  →  gold/spark_aggregates/ · gold/mapreduce_aggregates/", 11, False, MUTED),
    ("실행: bigdata/run_pipeline.sh (HDFS 기동 → 적재 → Spark → MapReduce → 검증)", 11, False, MUTED)])
notes(n9, "피드백(속성 정리를 위한 것이 Spark·Hadoop 전처리)에 맞춰 두 도구를 HDFS로 연결했다. 원천을 HDFS에 올리고, Spark가 HDFS에서 읽어 정제한 결과를 다시 HDFS에 쓰며, MapReduce는 그 결과를 읽어 집계한다. 스크립트는 저장소 bigdata/에 있고, 실행 검증 결과는 재실행 후 반영한다.",
      ["bigdata/README.md (worktree 브랜치 sangdelia/blobfish)", "bigdata/run_pipeline.sh", "bigdata/spark_pipeline.py", "bigdata/verify_pipeline.py"])

# N11 model A formula
n11 = new_slide("모델 A 수식: 이진 **로지스틱 회귀**",
                "6종 통합 train **249,732행** · 속성 x = 문자 2–5gram TF-IDF 200,000차원 · 정답 y ∈ {0, 1}",
                "예: '너는 정말 멍청한 바보야.'  z = 5.733 − 1.341 = 4.392 → P = 0.988 → **98.8점 ≥ 43 → 유해**",
                "기준 0.43과 설정(2–5gram·20만)은 validation Macro-F1로 선택 · test는 최종 보고에만 사용")
cw, cg = 3141683, 250000
cols = [("① 확률", ["**z = wᵀx + b**", "**P(유해 | x) = σ(z) = 1 / (1 + e⁻ᶻ)**", "w: 조각별 가중치 20만 개", "b: 절편"]),
        ("② 학습 (손실 최소화)", ["min ½‖w‖² + C·Σᵢ c_yᵢ · log(1 + e^(−ỹᵢ zᵢ))", "ỹ = ±1, C = 1 (L2 규제)", "**클래스 가중치 c_y = N / (2·N_y)**", "→ 유해 63% 불균형 보정"]),
        ("③ 판정", ["점수 = 100 × P", "**P ≥ 0.43 → 유해**", "P < 0.43 → 정상", "인식 조각 0개 → 판단 보류"])]
for i, (h, ls) in enumerate(cols):
    box(n11, X0 + i * (cw + cg), TOP + 20000, cw, 2560000, [(h, 14, True, MUTED)] + [(l, 15 if k < 2 else 13, False, INK) for k, l in enumerate(ls)],
        fill=PALE if i == 0 else GREYF, line=BLUE if i == 0 else GREYL, space=14, anchor=MSO_ANCHOR.MIDDLE)
notes(n11, "피드백(모델 A·B를 만들었으면 수식)에 대한 답. 모델 A는 속성 x에 가중치 w를 곱해 더한 z를 시그모이드로 확률로 바꾼다. 학습은 L2 규제가 있는 가중 로지스틱 손실 최소화(scikit-learn liblinear)이고, 유해가 63%라 클래스 가중치로 균형을 맞췄다. 판정 기준 0.43은 validation에서 정했다.",
      ["src/bigdb_mvp/experiments/classification.py", "reports/korean_short_v1/char_2_5_200000.json"])

# N12 model B formula
n12 = new_slide("모델 B 수식: 평점 3등급 분류 → **점수**",
                "K-HATERS Ratings train **172,097행** · x = 문자 2–5gram TF-IDF 10만 차원 · 점수 3개 각각 학습",
                "판정 기준 없이 **세 점수를 함께 표시** · 점수는 정답 확률이나 피해 심각도가 아님",
                "모델 A와 별도 데이터·모델 · 온도 T는 validation NLL로 선택(1.0) · 정규화 v2는 validation MAE로 채택")
cols = [("① 정답 (각 0·1·2)", ["Hate = max(성별·연령·장애·직업·인종·종교·정치)", "공격성 = max(평점 13개 전체)", "욕설 = 욕설 평점"]),
        ("② 확률 (다항 로지스틱 회귀)", ["**z_k = w_kᵀx + b_k,  k ∈ {0, 1, 2}**", "**P(k | x) = e^(z_k/T) / Σⱼ e^(z_j/T)**", "lbfgs · T = 1.0"]),
        ("③ 점수", ["**S = 50·P(1) + 100·P(2)**", "= 50 × (P(1) + 2·P(2))", "0점(모두 0등급) ~ 100점(모두 2등급)"])]
for i, (h, ls) in enumerate(cols):
    box(n12, X0 + i * (cw + cg), TOP + 20000, cw, 2560000, [(h, 14, True, MUTED)] + [(l, 14 if i else 13, False, INK) for l in ls],
        fill=PALE if i == 2 else GREYF, line=BLUE if i == 2 else GREYL, space=16, anchor=MSO_ANCHOR.MIDDLE)
notes(n12, "모델 B는 K-HATERS의 항목별 평점(0·1·2)을 정답으로 3등급 분류를 학습하고, 등급 확률의 기댓값을 0–100점으로 바꾼다. Hate·공격성·욕설 각각 따로 학습한다.",
      ["src/bigdb_mvp/experiments/khaters_score_train.py", "src/bigdb_mvp/experiments/khaters_ratings.py", "reports/khaters_score_v2/metrics.json"])

# N14 confusion matrix
n14 = new_slide("모델 A 혼동행렬과 **지표 계산**",
                "test 31,349건 (정상 11,599 · 유해 19,750) · 판정 기준 43점",
                "정상 문장 4개 중 1개 이상을 유해로 판정 → **오탐 줄이기가 핵심 과제** (18장)",
                "정확도 79.33%는 유해가 63%인 불균형에서 과대평가될 수 있어 Macro-F1·재현율·오탐률을 함께 봄")
table(n14, X0, TOP + 150000, [
    ["", "예측 정상", "예측 유해", "합계"],
    ["**실제 정상", "8,372 (TN)", "!!3,227 (FP)", "11,599"],
    ["**실제 유해", "3,254 (FN)", "^^16,496 (TP)", "19,750"],
    ["**합계", "11,626", "19,723", "31,349"],
], [1300000, 1350000, 1350000, 1000000], sz=1300, rowh=520000)
text(n14, X0 + 5250000, TOP + 100000, 4675050, 2200000, [
    ("!!정상 오탐률!! = FP / 실제 정상 = 3,227 / 11,599 = !!27.82%!!", 14, False, INK),
    ("**유해 재현율** = TP / 실제 유해 = 16,496 / 19,750 = **83.52%**", 14, False, INK),
    ("유해 정밀도 = TP / 예측 유해 = 16,496 / 19,723 = 83.64%", 13, False, INK),
    ("F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)", 12, False, MUTED),
    ("**Macro-F1** = (F1 정상 0.7209 + F1 유해 0.8358) / 2 = **0.7784**", 14, False, INK),
], space=12, anchor=MSO_ANCHOR.MIDDLE)
notes(n14, "피드백(27%가 어떻게 나왔나)에 대한 답. 31만 1,661행을 연결 그룹 단위로 나눈 test 31,349건에 모델 A를 적용한 혼동행렬이다. 실제 정상 11,599건 중 3,227건을 유해로 판정해 정상 오탐률 27.82%가 나온다.",
      ["reports/korean_short_v1/char_2_5_200000.json (test tn/fp/fn/tp)"])

# N15 baselines
n15 = new_slide("얼마나 잘 만들었나: **기준선**과 비교",
                "같은 test에서 단순 기준선·다른 설정과 비교 · 설정 선택은 validation 기준",
                f"모델 A: 다수 클래스 기준선 **{maj['macro_f1']:.4f} → 0.7784** · 모델 B: 평균값 기준선보다 MAE **27–39% 낮음**",
                "다수 클래스 = 모든 문장을 유해로 판정 · 평균값 = train 평균 점수를 항상 출력 · 통계적 유의성 검정은 하지 않음")
chart(n15, X0, TOP - 60000, 5300000, 2700000,
      ["다수 클래스\n(전부 유해)", "3–5gram\n6만", "3–5gram\n20만", "2–5gram\n6만", "2–5gram\n20만 (채택)"],
      [("test Macro-F1", (round(maj["macro_f1"], 4), 0.7575, 0.7616, 0.7749, 0.7784), LIGHT)],
      XL_CHART_TYPE.COLUMN_CLUSTERED, "0.0000", title="모델 A · test Macro-F1", vmax=0.95, point_colors={0: GREYL, 4: BLUE}, cat_size=10, label_size=10)
chart(n15, X0 + 5500000, TOP - 60000, 4425050, 2700000, ["Hate", "공격성", "욕설"],
      [("평균값 기준선", (32.44, 35.22, 23.16), GREYL), ("모델 B", (19.77, 25.76, 15.02), BLUE)],
      XL_CHART_TYPE.COLUMN_CLUSTERED, "0.00", title="모델 B · test MAE (낮을수록 좋음)", vmax=42, legend=True, cat_size=11, label_size=10)
notes(n15, "피드백(얼마나 잘 만들었냐)에 대한 답. 모델 A는 모든 문장을 유해로 찍는 기준선(Macro-F1 0.3865)보다 크게 높고, 조각 길이·특징 수를 바꾼 설정 중 2–5gram·20만이 가장 좋았다. 모델 B는 항상 평균을 내는 기준선보다 세 점수 모두 오차가 낮다.",
      ["[브랜치 sangdelia/blobfish] reports/postmidterm_20261105/majority.json", "docs/progress/2026-10-10.md §16", "reports/khaters_score_v2/metrics.json"])

# N16 learning curve
fr = (0.05, 0.1, 0.25, 0.5, 1.0)
n16 = new_slide("데이터 규모 실험: 학습 데이터가 늘면 성능은?",
                "train만 5–100%로 줄여 재학습 · validation·test·모델 설정(2–5gram, 20만, 가중치)은 고정",
                f"train {lc[0.05]['train_count']:,}행 → {lc[1.0]['train_count']:,}행: test Macro-F1 **{lc[0.05]['test']['macro_f1']:.4f} → {lc[1.0]['test']['macro_f1']:.4f}**, 데이터가 늘수록 계속 상승",
                f"증가 폭은 줄어드는 추세(50% → 100%: +{lc[1.0]['test']['macro_f1'] - lc[0.5]['test']['macro_f1']:.4f}) · 같은 6종 안의 비교이며 새 출처 추가 효과는 아님")
cats = [f"{int(f * 100)}%\n{lc[f]['train_count'] / 10000:.1f}만" for f in fr]
chart(n16, X0, TOP - 80000, W, 2750000, cats,
      [("validation", tuple(round(lc[f]["validation"]["macro_f1"], 4) for f in fr), LIGHT),
       ("test", tuple(round(lc[f]["test"]["macro_f1"], 4) for f in fr), BLUE)],
      XL_CHART_TYPE.LINE_MARKERS, "0.0000", title="Macro-F1 (train 비율 · 행 수)", vmin=0.70, vmax=0.80, legend=True, line=True, cat_size=11, label_size=10)
notes(n16, "피드백(추가 데이터로 테스트하면 정확도가 오를 것)에 대한 답. 학습 데이터 양만 바꿔 다시 학습해 보니 5%에서 100%로 늘릴수록 Macro-F1이 꾸준히 올랐다. 증가 폭은 줄지만 아직 포화되지 않아, 데이터를 더 모으면 개선 여지가 있다.",
      ["[브랜치 sangdelia/blobfish] reports/postmidterm_20261105/learning_curve/", "[브랜치] experiments/learning_curve.py"])

# N18 false positives + hold rule
n18 = new_slide("오탐 원인과 **판단 보류** 규칙",
                "모델은 그대로, 불확실한 '유해' 판정만 **보류** · 규칙은 validation으로 선택, test 1회 평가",
                f"test: 보류 **{pct(tsel['review_rate'])}** · 자동 판정의 정상 오탐률 **{pct(tb['false_positive_rate'])} → {pct(tsel['false_positive_rate'])}** · Macro-F1 {tb['macro_f1']:.4f} → **{tsel['macro_f1']:.4f}**",
                f"!!보류 {tsel['review']:,}건 중 {tsel['review_true_harmful']:,}건은 실제 유해!! → 사람 검토가 필요 · 부정문·동음이의어 자체는 문맥 모델(E3)로 해결해야 함")
pr = {p["text"]: p for p in fp["probes"]}
table(n18, X0, TOP - 60000, [
    ["입력", "점수", "원인", "규칙 적용 후"],
    ["시래기", "74.33", "조각 1개('래기')가 점수를 결정 · 학습 데이터 9건 모두 '쓰레기' 변형", "^^보류"],
    ["너는 바보가 아니야.", "43.80", "기준 43 바로 위 · 부정문 안의 '바보'도 유해 쪽", "^^보류"],
    ["쓰레기 / 시발", "99.76 / 95.57", "점수가 높아 규칙 대상 아님", "유해 유지"],
    ["닥쳐", "55.94", "조각 1개 · 점수 80 미만", "!!보류 (부작용)"],
], [1900000, 1150000, 4600000, 1300000], sz=1100, rowh=330000, align=["l", "ctr", "l", "ctr"])
box(n18, X0, TOP + 1650000 + 20000, W, 800000, [
    (f"**R1 경계 보류**: 43 ≤ 점수 < {round((fp['threshold'] + sel['b']) * 100)} → 보류      "
     f"**R2 단일 조각 보류**: 인식 조각 ≤ {sel['k']}개 · 최대 조각 비중 ≥ {int(sel['d'] * 100)}% · 점수 < {int(sel['s'] * 100)} → 보류", 12, False, INK),
    ("선택 기준: validation 보류율 5% 이하에서 자동 판정 Macro-F1 최대 (경계 폭·조각 수·비중·점수 상한 격자 탐색)", 11, False, MUTED)],
    fill=PALE, line=BLUE, anchor=MSO_ANCHOR.MIDDLE, space=4)
notes(n18, "발표 중 찾은 오탐(시래기, 부정문)을 고친 실험. 모델을 다시 학습하지 않고, 기준선 바로 위의 애매한 점수와 조각 하나가 점수를 좌우하는 짧은 입력을 '판단 보류'로 돌렸다. test에서 자동 판정 대상의 정상 오탐률이 4%p 넘게 줄었지만, 보류된 문장의 절반 이상은 실제 유해라 사람 검토가 필요하다. 처음 정한 선택 기준(재현율 유지)으로는 어떤 규칙도 뽑히지 않아 기준을 'Macro-F1 최대'로 바꾼 뒤 test를 1회 평가했다. 웹 서비스에는 아직 반영하지 않았다.",
      ["[브랜치 sangdelia/blobfish] reports/postmidterm_20261105/fp_review3.json", "[브랜치] experiments/fp_review3.py", "artifacts/korean_short_v1/char_2_5_200000.joblib"])

# =================================================================== edits to kept slides
S3, S4, S5, S6, S7, S8, S9, S10 = S[2], S[3], S[4], S[5], S[6], S[7], S[8], S[9]
put(S3, "body-summary", "K-HATERS **54.1%** · 유해 **63.0%** → !!불균형!! → Macro-F1 평가 · 클래스 가중치 학습")
put(S4, "subtitle", "**정답(0/1) 데이터와 학습용 분할**을 만드는 단계 · 행 수는 각 단계 적용 후 · 원문은 보존")
put(S8, "headline", "모델 A (6종 이진 분류) test 결과")
put(S8, "subtitle", "311,661행을 연결 그룹 단위로 나눈 **test 31,349건** (학습에 쓰지 않은 10%)")
put(S8, "body-summary", "정상 오탐률 27.82% = 실제 정상 11,599건 중 **3,227건**을 유해로 판정 (계산은 14장)")
put(S10, "body-summary", "E2 1차 결과: 보류 규칙으로 자동 판정 오탐률 **4.2%p 감소** (18장) → 웹 반영 검토")

# =================================================================== order, numbering, drop old slide 2
order = [S[0], n2, n3, S3, S4, n6, S5, S6, n9, S7, n11, n12, S8, n14, n15, n16, S9, n18, S10, S[10], S[11]] + S[12:20]
drop = S[1]
lst = prs.slides._sldIdLst
els = {id(s): el for s, el in zip(prs.slides, list(lst))}
new_ids = [els[id(s)] for s in order]
rid = els[id(drop)].get(qn("r:id"))
for el in list(lst): lst.remove(el)
for el in new_ids: lst.append(el)
prs.part.drop_rel(rid)
for i, s in enumerate(list(prs.slides)[:21], 1):
    put(s, "page-number", f"{i:02d}")
    tf = s.notes_slide.notes_text_frame
    ps = [q.text for q in tf.paragraphs]
    if ps and ps[0].startswith("발표용 대본"):
        keep = [ps[1]] + [q for q in ps[2:] if q.startswith("근거 파일")]
        tf.text = "\n".join(["발표 요점", keep[0], ""] + keep[1:])
prs.save(DST)
print("slides", len(prs.slides))

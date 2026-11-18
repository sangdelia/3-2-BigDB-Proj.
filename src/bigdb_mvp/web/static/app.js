"use strict";
const form = document.querySelector("#analysis-form");
const message = document.querySelector("#message");
const submit = document.querySelector("#submit");
const clear = document.querySelector("#clear");
const examples = Array.from(document.querySelectorAll("[data-example]"));
const results = document.querySelector("#results");
const empty = document.querySelector("#empty-state");
const status = document.querySelector("#status");
const error = document.querySelector("#input-error");
let busy = false;
let controller = null;
let version = 0;

function invalidate() {
  version += 1;
  if (controller) controller.abort();
  results.hidden = true;
  empty.hidden = false;
  document.querySelector("#candidate-list").replaceChildren();
  document.querySelector("#highlighted-message").replaceChildren();
  for (const id of ["positive-features", "negative-features", "decision-margin", "decision-calculation", "feature-counts"]) {
    document.querySelector(`#${id}`).replaceChildren();
  }
  document.querySelector("#decision-evidence").hidden = true;
  error.hidden = true;
  message.removeAttribute("aria-invalid");
  status.textContent = "메시지를 입력하면 분석 결과가 나타납니다.";
}
function updateCount() {
  document.querySelector("#character-count").textContent = `${Array.from(message.value).length} / 1,000`;
}
function setBusy(value) {
  busy = value;
  submit.disabled = value;
  clear.disabled = value;
  examples.forEach(button => { button.disabled = value; });
  message.readOnly = value;
  document.querySelector("#result-panel").setAttribute("aria-busy", String(value));
  submit.firstChild.textContent = value ? "분석 중… " : "분석하기 ";
}
function showError(text) {
  error.textContent = text;
  error.hidden = false;
  message.setAttribute("aria-invalid", "true");
  status.textContent = "분석하지 못했습니다. 입력과 안내를 확인하세요.";
  message.focus();
}
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function renderEvidence(data) {
  const evidence = data.decision_evidence;
  document.querySelector("#decision-evidence").hidden = !evidence;
  if (!evidence) return;
  const margin = evidence.margin_points;
  const relation = margin >= 0 ? "높아 유해·공격 표현" : "낮아 일반 표현";
  document.querySelector("#decision-margin").textContent = `분류 점수 ${data.harmful_score.toFixed(2)}점이 기준 ${data.threshold_score.toFixed(2)}점보다 ${Math.abs(margin).toFixed(2)}점 ${relation}으로 판단했습니다. 문맥에 따라 오판할 수 있습니다.`;
  for (const direction of ["positive", "negative"]) {
    const list = document.querySelector(`#${direction}-features`);
    const items = evidence[`${direction}_features`];
    list.replaceChildren(...items.map(item => node("li", `“${item.ngram}” · ${item.contribution > 0 ? "+" : ""}${item.contribution.toFixed(3)}`)));
    if (items.length === 0) list.append(node("li", "해당 방향의 특징 없음"));
  }
  document.querySelector("#feature-counts").textContent = `올리는 특징 ${evidence.positive_count}개, 내리는 특징 ${evidence.negative_count}개 중 방향별 최대 6개를 표시합니다. 전체 계산에는 모두 포함됩니다.`;
  document.querySelector("#decision-calculation").textContent = `기본값 ${evidence.intercept.toFixed(6)} + 올리는 기여 ${evidence.positive_sum.toFixed(6)} + 내리는 기여 (${evidence.negative_sum.toFixed(6)}) = ${evidence.logit.toFixed(6)} (로짓). 이 합을 sigmoid로 변환하고 100을 곱한 값이 분류 점수입니다. 표시된 숫자는 반올림돼 있습니다.`;
}
function render(data, text) {
  renderEvidence(data);
  const rating = data.rating_scores;
  for (const kind of ["hate", "abuse", "swear"]) {
    const value = rating ? rating[`${kind}_score`] : null;
    document.querySelector(`#${kind}-score`).textContent = value === null ? "—" : `${value.toFixed(2)} / 100`;
  }
  document.querySelector("#rating-note").textContent = !rating ? "평점 모델을 사용할 수 없습니다. 아래 분류 결과를 확인하세요." : rating.recognized_features === 0 ? "인식한 문자 특징이 없어 평점은 판단 보류입니다." : "원래 평점 0·1·2를 0·50·100에 대응한 예측 평균입니다. 정답 확률이나 객관적인 피해 심각도가 아닙니다.";
  const score = data.harmful_score.toFixed(2);
  const review = data.label === "review";
  document.querySelector("#score").textContent = review ? "—" : score;
  document.querySelector(".score-unit").hidden = review;
  document.querySelector("#score-meter").hidden = review;
  document.querySelector("#score-meter").value = data.harmful_score;
  document.querySelector("#score-meter").setAttribute("aria-valuetext", `${score}점`);
  const harmful = data.label === "harmful";
  const verdict = document.querySelector("#verdict");
  verdict.textContent = review ? "판단 보류" : harmful ? "유해·공격 표현 판단" : "일반 표현 판단";
  verdict.className = review ? "verdict review" : harmful ? "verdict harmful" : "verdict";
  document.querySelector("#threshold-note").textContent = review ? "학습된 문자 특징을 인식하지 못했습니다. 기본 계산값은 분석 점수로 표시하지 않습니다." : `판단 기준 ${data.threshold_score.toFixed(1)}점 · 인식 특징 ${data.recognized_feature_count}개 · 정답 확률이나 혐오 강도가 아닌 모델 점수입니다.`;
  const candidates = data.candidates.filter(item => item.highlight_as_harmful_candidate);
  const preview = document.querySelector("#highlighted-message");
  preview.replaceChildren();
  const points = Array.from(text);
  let cursor = 0;
  [...candidates].sort((a,b) => a.start - b.start).forEach(item => {
    preview.append(document.createTextNode(points.slice(cursor,item.start).join("")));
    const mark = node("mark", points.slice(item.start,item.end).join(""));
    mark.title = `제거 시 점수 ${item.score_drop_points.toFixed(2)}점 감소`;
    preview.append(mark);
    cursor = item.end;
  });
  preview.append(document.createTextNode(points.slice(cursor).join("")));
  document.querySelector("#candidate-note").textContent = review ? "일반 표현이나 유해 표현으로 확정할 근거가 없습니다. 문맥을 포함해 다시 입력하거나 사람이 검토하세요." : candidates.length ? `${candidates.length}개 후보 · 밑줄과 점수 차이는 6종 분류 모델의 근거이며 K-HATERS 평점의 근거는 아닙니다.` : (harmful ? "강조할 표현의 근거가 부족합니다. 문장 판단과 구간 설명은 별개입니다." : "유해 후보로 강조한 구간이 없습니다. 단어 하나만으로 유해 표현을 확정하지 않습니다.");
  const list = document.querySelector("#candidate-list");
  list.replaceChildren();
  const comparisons = harmful ? data.candidates.filter(item => item.highlight_as_harmful_candidate || item.score_drop_points === null) : [];
  comparisons.forEach((item,index) => {
    const card = node("article", undefined, "candidate");
    const heading = node("div", undefined, "candidate-heading");
    const valid = item.score_drop_points !== null && item.score_without !== null;
    heading.append(node("h3", `후보 ${index+1} · ${item.text}`, "candidate-title"), node("span", valid ? `−${item.score_drop_points.toFixed(2)}점` : "비교 보류", valid ? "delta" : "metric-label"));
    card.append(heading, node("p", valid ? `이 구간 제거 시 ${score} → ${item.score_without.toFixed(2)}점. 삭제에 따라 문장 의미도 달라질 수 있습니다.` : "이 구간을 지우면 인식 특징이 0개입니다. 기본값과의 차이를 유해성 근거로 표시하지 않습니다."));
    const features = node("ul", undefined, "feature-list");
    features.setAttribute("aria-label", `후보 ${index+1}의 실제 모델 특징과 기여`);
    item.evidence.forEach(evidence => features.append(node("li", `“${evidence.ngram}” · ${evidence.occurrence_contribution.toFixed(3)}`)));
    const details = node("details", undefined, "limitations");
    details.append(node("summary", "이 후보의 양의 특징 보기"), features);
    card.append(details);
    list.append(card);
  });
  const limitations = document.querySelector("#limitations-list");
  limitations.replaceChildren(...data.limitations.map(text => node("li", text)));
  empty.hidden = true;
  results.hidden = false;
  status.textContent = "분석이 완료됐습니다. 점수와 표현 후보의 근거를 확인하세요.";
  document.querySelector("#result-title").focus();
}
message.addEventListener("input", () => { invalidate(); updateCount(); });
clear.addEventListener("click", () => { if (busy) return; message.value = ""; invalidate(); updateCount(); message.focus(); });
examples.forEach(button => button.addEventListener("click", () => { if (busy) return; message.value = button.dataset.example; invalidate(); updateCount(); message.focus(); }));
form.addEventListener("submit", async event => {
  event.preventDefault();
  if (busy) return;
  invalidate();
  const text = message.value;
  if (!text.trim() || !/[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]/u.test(text)) { showError("한글이 포함된 메시지를 입력하세요."); return; }
  if (Array.from(text).length > 1000) { showError("메시지를 1,000자 이내로 줄여주세요."); return; }
  const current = version;
  controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  setBusy(true);
  status.textContent = "학습된 한국어 모델로 메시지와 표현 후보를 분석하고 있습니다.";
  try {
    const response = await fetch("/api/korean/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text}),signal:controller.signal,cache:"no-store",credentials:"omit"});
    if (!response.ok) {
      if (response.status === 503) throw new Error("분석 모델을 사용할 수 없습니다. 서버의 모델 준비 상태를 확인해주세요.");
      if (response.status === 422 || response.status === 413) throw new Error("입력 형식이나 길이를 확인해주세요.");
      throw new Error("분석 요청에 실패했습니다. 잠시 후 다시 시도해주세요.");
    }
    const data = await response.json();
    if (current === version) render(data, text);
  } catch (failure) {
    if (current === version) showError(failure.name === "AbortError" ? "응답 시간이 길어 분석을 중단했습니다. 다시 시도해주세요." : failure.message);
  } finally {
    clearTimeout(timeout);
    controller = null;
    setBusy(false);
  }
});
window.addEventListener("pagehide", () => { if (controller) controller.abort(); });
window.addEventListener("pageshow", () => { message.value=""; invalidate(); updateCount(); setBusy(false); });

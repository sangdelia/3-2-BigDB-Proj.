// 진입점: 화면 상태(로딩·오류·결과), 출처 필터, 테마, 인쇄, 내려받기
import { REPORT_URL, ReportError, loadReport } from './contract.js';
import { fill, h } from './dom.js';
import { renderAnnotations, renderLabels, renderLength, renderSplits } from './render-distribution.js';
import {
  renderMeta,
  renderMethods,
  renderMissing,
  renderNotes,
  renderPipeline,
  renderRuntime,
  renderSourceOptions,
  renderSourceTable,
} from './render-quality.js';

const root = document.documentElement;
const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
const byId = (id) => document.getElementById(id);
const ui = Object.freeze({
  main: byId('main'),
  status: byId('status'),
  statusText: byId('status-text'),
  retry: byId('retry'),
  report: byId('report'),
  toc: byId('toc'),
  meta: byId('meta'),
  download: byId('download'),
  theme: byId('theme-toggle'),
  print: byId('print'),
  sourceForm: byId('source-filters'),
  sourceFilter: byId('source-filter'),
  sourceSearch: byId('source-search'),
  sourceCount: byId('source-count'),
});

// 마지막으로 성공한 집계와 진행 중 여부. 재시도 실패 시 이전 결과를 유지한다.
const state = { report: null, loading: false };

// ---------- 출처 필터 ----------

function renderSources() {
  if (!state.report) return;
  ui.sourceCount.textContent = renderSourceTable(
    state.report.sources,
    { selected: ui.sourceFilter.value, query: ui.sourceSearch.value },
    resetSourceFilters
  );
}

function resetSourceFilters() {
  ui.sourceFilter.value = '';
  ui.sourceSearch.value = '';
  renderSources();
  ui.sourceSearch.focus();
}

function renderReport(data) {
  renderMeta(ui.meta, data);
  renderPipeline(data);
  renderSourceOptions(ui.sourceFilter, data.sources);
  renderSources();
  renderMissing(data);
  renderLabels(data);
  renderLength(data);
  renderSplits(data);
  renderAnnotations(data);
  renderMethods(data);
  renderRuntime(data);
  renderNotes(data);
}

// ---------- 화면 상태 ----------

function setDownload(enabled) {
  if (enabled) {
    ui.download.setAttribute('href', REPORT_URL);
    ui.download.setAttribute('download', 'bigdb-eda-report.json');
    ui.download.removeAttribute('aria-disabled');
  } else {
    ui.download.removeAttribute('href');
    ui.download.removeAttribute('download');
    ui.download.setAttribute('aria-disabled', 'true');
  }
}

function showLoading() {
  ui.status.hidden = false;
  ui.status.classList.remove('is-error');
  ui.statusText.textContent = '집계를 불러오는 중입니다.';
  ui.retry.hidden = true;
  ui.retry.disabled = true;
  ui.main.setAttribute('aria-busy', 'true');
}

function showError(message) {
  ui.status.hidden = false;
  ui.status.classList.add('is-error');
  ui.statusText.textContent = message;
  ui.retry.hidden = false;
  ui.retry.disabled = false;
  ui.main.setAttribute('aria-busy', 'false');
  if (state.report) return;
  ui.report.hidden = true;
  ui.toc.hidden = true;
  setDownload(false);
  fill(ui.meta, h('p', { class: 'muted' }, '집계 정보를 표시할 수 없습니다.'));
}

function showReport() {
  ui.status.hidden = true;
  ui.report.hidden = false;
  ui.toc.hidden = false;
  ui.main.setAttribute('aria-busy', 'false');
  setDownload(true);
}

// 렌더 실패는 원인과 함께 오류 상태로 바꾸고, 이전 성공 결과로 되돌린다.
function applyReport(data) {
  const previous = state.report;
  state.report = data;
  try {
    renderReport(data);
  } catch (err) {
    state.report = previous;
    if (err instanceof Error) throw new ReportError('render', '집계를 화면에 그리지 못했습니다: ' + err.message);
    throw err;
  }
}

async function load() {
  if (state.loading) return;
  state.loading = true;
  showLoading();
  try {
    applyReport(await loadReport());
    showReport();
  } catch (err) {
    if (err instanceof ReportError) {
      showError(err.message);
      return;
    }
    // 예상하지 못한 오류: 화면에는 알리고 콘솔로도 전파한다.
    showError('알 수 없는 오류로 집계를 표시하지 못했습니다.');
    throw err;
  } finally {
    state.loading = false;
  }
}

// ---------- 테마·인쇄 ----------

function effectiveTheme() {
  return root.dataset.theme || (darkQuery.matches ? 'dark' : 'light');
}

function updateThemeButton() {
  ui.theme.textContent = effectiveTheme() === 'dark' ? '밝은 화면으로 보기' : '어두운 화면으로 보기';
}

ui.theme.addEventListener('click', () => {
  root.dataset.theme = effectiveTheme() === 'dark' ? 'light' : 'dark';
  updateThemeButton();
});
darkQuery.addEventListener('change', updateThemeButton);
ui.print.addEventListener('click', () => window.print());
ui.download.addEventListener('click', (event) => {
  if (ui.download.getAttribute('aria-disabled') === 'true') event.preventDefault();
});
ui.retry.addEventListener('click', load);
ui.sourceForm.addEventListener('submit', (event) => event.preventDefault());
ui.sourceFilter.addEventListener('change', renderSources);
ui.sourceSearch.addEventListener('input', renderSources);

updateThemeButton();
load();

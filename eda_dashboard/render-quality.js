// 품질·처리 렌더: 출처 정보, 처리 흐름, 출처 표, 결측, 방법 비교, 실행 환경, 참고 사항
import { barWithValue, cell, definitionPairs, emptyRow, fill, h, numCell, rowHeader } from './dom.js';
import { fmt, fmtDate, fmtSeconds, percent, share, sumBy } from './format.js';

const SOURCE_COLUMNS = Object.freeze([
  ['원본', 'original_rows'],
  ['정제', 'parsed_rows'],
  ['최종', 'final_rows'],
  ['clean', 'clean'],
  ['harmful', 'harmful'],
  ['문맥 보유', 'context_rows'],
  ['대상 주석', 'target_annotated_rows'],
  ['표현 주석', 'expression_annotated_rows'],
]);

export function sourceDisplayName(src) {
  return src.name || src.source;
}

export function renderMeta(target, data) {
  const p = data.provenance;
  fill(
    target,
    h(
      'dl',
      null,
      definitionPairs([
        ['집계일', h('time', { datetime: data.generated_at }, fmtDate(data.generated_at))],
        ['실행 범위', p.scope],
        ['최종 데이터 SHA-256', h('code', null, p.dataset_sha256)],
        ['원본 데이터 SHA-256', h('code', null, p.raw_dataset_sha256)],
      ])
    )
  );
}

function flowItem(stepText, name, rows, removed, isFinal) {
  return h(
    'li',
    { class: isFinal ? 'is-final' : null },
    h('span', { class: 'flow-step' }, stepText),
    h('span', { class: 'flow-name' }, name),
    h('span', { class: 'flow-rows' }, fmt(rows) + '행'),
    removed === null ? null : h('span', { class: 'flow-removed' }, '제외 ' + fmt(removed) + '행')
  );
}

export function renderPipeline(data) {
  const s = data.summary;
  const steps = data.pipeline;
  fill(
    'pipeline-lead',
    `원본 ${fmt(s.raw_rows)}행 중 최종 ${fmt(s.final_rows)}행(${percent(share(s.final_rows, s.raw_rows))})이 남았습니다. ` +
      `필수값 검사에서 ${fmt(s.invalid_rows)}행, 라벨 충돌로 ${fmt(s.conflicting_rows)}행, 중복 병합으로 ${fmt(s.duplicate_rows)}행이 빠졌습니다.`
  );
  fill(
    'pipeline-flow',
    steps.length ? flowItem('시작', '원본', steps[0].input_rows, null, false) : null,
    steps.map((step, index) => flowItem(index + 1 + '단계', step.label, step.output_rows, step.removed_rows, index === steps.length - 1))
  );
  fill(
    'pipeline-rows',
    steps.map((step, index) =>
      h(
        'tr',
        null,
        rowHeader(index + 1 + '. ' + step.label),
        numCell('입력 행', step.input_rows),
        numCell('제외 행', step.removed_rows),
        numCell('출력 행', step.output_rows),
        cell('처리 내용', step.description, 'desc')
      )
    )
  );
}

export function renderSourceOptions(select, sources) {
  fill(
    select,
    h('option', { value: '' }, '전체 출처'),
    sources.map((src) => h('option', { value: src.source }, sourceDisplayName(src)))
  );
}

function sourceRow(src) {
  return h(
    'tr',
    null,
    rowHeader(sourceDisplayName(src), src.name && src.name !== src.source ? src.source : null),
    SOURCE_COLUMNS.map(([label, key]) => numCell(label, src[key]))
  );
}

function matchesSource(src, selected, query) {
  if (selected && src.source !== selected) return false;
  if (!query) return true;
  return String(src.source).toLowerCase().includes(query) || String(src.name || '').toLowerCase().includes(query);
}

// filter: { selected, query } (query는 사용자가 입력한 그대로). 표시 결과를 돌려준다.
export function renderSourceTable(sources, filter, onReset) {
  const query = filter.query.trim();
  const rows = sources.filter((src) => matchesSource(src, filter.selected, query.toLowerCase()));
  const filtered = filter.selected !== '' || query !== '';

  if (rows.length) {
    fill('source-rows', rows.map(sourceRow));
  } else {
    const reset = h('button', { class: 'btn btn-secondary', type: 'button' }, '필터 초기화');
    reset.addEventListener('click', onReset);
    const message = query ? `“${query}”와 일치하는 출처가 없습니다.` : '선택한 조건에 맞는 출처가 없습니다.';
    fill('source-rows', emptyRow(SOURCE_COLUMNS.length + 1, h('div', { class: 'empty-state' }, h('span', null, message), reset)));
  }

  fill(
    'source-total',
    filtered || !sources.length
      ? null
      : h(
          'tr',
          null,
          h('th', { scope: 'row', class: 'row-first' }, '전체 출처 합계'),
          SOURCE_COLUMNS.map(([label, key]) => numCell(label, sumBy(sources, key)))
        )
  );

  return filtered ? `출처 ${fmt(sources.length)}개 중 ${fmt(rows.length)}개 표시` : `출처 ${fmt(sources.length)}개`;
}

function missingRows(items, countLabel) {
  if (!items.length) return [emptyRow(4, '해당하는 필드가 없습니다.')];
  return items.map((item) =>
    h(
      'tr',
      null,
      rowHeader(item.label, h('code', null, item.field)),
      cell(countLabel, `${fmt(item.missing_rows)} / ${fmt(item.total_rows)}`, 'num'),
      cell('비율', barWithValue(item.rate, null), 'bar-cell'),
      cell('의미', item.meaning, 'desc')
    )
  );
}

export function renderMissing(data) {
  fill('missing-required', missingRows(data.missing.filter((m) => m.kind === 'required'), '결측 행'));
  fill('missing-optional', missingRows(data.missing.filter((m) => m.kind === 'optional'), '원천 미제공 행'));
}

export function renderMethods(data) {
  fill(
    'method-rows',
    data.methods.map((m) =>
      h(
        'tr',
        { class: m.selected ? 'is-selected' : null },
        rowHeader(m.title),
        cell('판단', h('span', { class: m.selected ? 'chip tone-selected' : 'chip' }, m.selected ? '채택' : '비교용')),
        numCell('시작 행', m.start_rows),
        numCell('남는 행', m.retained_rows),
        numCell('삭제 행', m.removed_rows),
        cell('남는 비율', barWithValue(share(m.retained_rows, m.start_rows), null), 'bar-cell'),
        cell('이유', m.reason, 'desc')
      )
    )
  );
}

// Hadoop 상태는 completed 또는 pending/failed. 완료가 아닌 상태는 완료처럼 보이지 않게 pending 톤을 쓴다.
// 그 밖의 값(Spark의 historical 등)은 중립 톤.
function statusTone(status) {
  switch (status) {
    case 'completed':
      return 'done';
    case 'pending':
    case 'failed':
      return 'pending';
    default:
      return 'neutral';
  }
}

function statusChip(runtime) {
  return h(
    'span',
    null,
    h('span', { class: 'chip tone-' + statusTone(runtime.status) }, runtime.label || runtime.status),
    runtime.label && runtime.status ? h('code', { class: 'status-raw' }, runtime.status) : null
  );
}

export function renderRuntime(data) {
  const { hadoop, spark } = data.runtime;
  fill(
    'runtime-hadoop',
    definitionPairs([
      ['현재 상태', statusChip(hadoop)],
      ['사유·향후 적용', hadoop.reason || '—'],
    ])
  );
  fill(
    'runtime-spark',
    definitionPairs([
      ['현재 상태', statusChip(spark)],
      ['실행 범위', spark.scope || '—'],
      ['Python 처리 시간', fmtSeconds(spark.python_seconds)],
      ['Spark 처리 시간', fmtSeconds(spark.spark_seconds)],
      ['Spark 시작 시간', fmtSeconds(spark.startup_seconds)],
    ])
  );
}

export function renderNotes(data) {
  fill('note-list', data.notes.length ? data.notes.map((note) => h('li', null, note)) : h('li', null, '기록된 참고 사항이 없습니다.'));
}

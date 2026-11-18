// 분포·구성 렌더: 라벨 분포, 출처별 라벨 구성, 문자 길이, 분할, 선택 주석 분포
import { barItems, barWithValue, cell, fill, h, numCell, rowHeader, stackBar } from './dom.js';
import { fmt, fmtDecimal, labelName, percent, share, sumBy } from './format.js';
import { sourceDisplayName } from './render-quality.js';

function labelTone(label) {
  return label === 'clean' || label === 'harmful' ? label : null;
}

function byCountDesc(items) {
  return items.slice().sort((a, b) => b.count - a.count);
}

export function renderLabels(data) {
  fill('label-bars', barItems(data.labels, { total: sumBy(data.labels, 'count'), toneOf: labelTone, nameOf: labelName }));
  fill(
    'label-by-source',
    data.sources.map((src) => {
      const total = src.clean + src.harmful;
      return h(
        'li',
        null,
        h(
          'span',
          { class: 'stack-head' },
          h('strong', null, sourceDisplayName(src)),
          h('span', { class: 'stack-detail' }, `최종 ${fmt(src.final_rows)}행`)
        ),
        stackBar(src.clean, src.harmful),
        h(
          'span',
          { class: 'stack-detail' },
          `clean ${fmt(src.clean)} (${percent(share(src.clean, total))}) · harmful ${fmt(src.harmful)} (${percent(share(src.harmful, total))})`
        )
      );
    })
  );
}

export function renderLength(data) {
  const { unit, stats, histogram, by_source: bySource } = data.length;
  const nameOf = (source) => {
    const match = data.sources.find((src) => src.source === source);
    return match ? sourceDisplayName(match) : source;
  };

  fill('length-lead', `길이 기준: ${unit || '저장된 원문의 Unicode 코드포인트 수'}. 정규화한 문장의 길이가 아닙니다.`);
  fill(
    'length-stats',
    [
      ['문장 수', fmt(stats.count)],
      ['최소', fmt(stats.min)],
      ['중앙값', fmt(stats.median)],
      ['95백분위', fmt(stats.p95)],
      ['최대', fmt(stats.max)],
      ['평균', fmtDecimal(stats.mean)],
    ].map(([term, value]) => h('div', null, h('dt', null, term), h('dd', null, value)))
  );
  fill('length-hist', barItems(histogram, { total: sumBy(histogram, 'count') }));
  fill(
    'length-by-source',
    bySource.map((row) =>
      h(
        'tr',
        null,
        rowHeader(nameOf(row.source)),
        numCell('문장 수', row.count),
        numCell('중앙값', row.median),
        numCell('95백분위', row.p95),
        numCell('최대', row.max)
      )
    )
  );
}

export function renderSplits(data) {
  const s = data.summary;
  fill('splits-lead', `분할 사이에 겹치는 문장 ${fmt(s.text_overlap)}건, 겹치는 그룹 ${fmt(s.group_overlap)}건이 집계되었습니다.`);
  fill(
    'split-rows',
    data.splits.map((row) =>
      h(
        'tr',
        null,
        rowHeader(row.split),
        numCell('문장 수', row.count),
        numCell('clean', row.clean),
        numCell('harmful', row.harmful),
        cell('harmful 비율', barWithValue(share(row.harmful, row.count), 'harmful'), 'bar-cell')
      )
    )
  );
}

// 원천 접두사가 붙은 주석: 비율 없이 건수만 보인다
export function renderAnnotations(data) {
  fill('target-bars', barItems(byCountDesc(data.targets)));
  fill('expression-bars', barItems(byCountDesc(data.expressions)));
}

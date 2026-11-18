// DOM 프리미티브. 외부 문자열은 항상 text node로 넣고 innerHTML을 쓰지 않는다.
import { fmt, percent, share } from './format.js';

function toNode(child) {
  return child instanceof Node ? child : document.createTextNode(String(child));
}

function isPresent(value) {
  return value !== null && value !== undefined && value !== false;
}

export function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (!isPresent(value)) continue;
    if (key === 'class') node.className = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter(isPresent).map(toNode));
  return node;
}

export function fill(target, ...children) {
  const node = typeof target === 'string' ? document.getElementById(target) : target;
  node.replaceChildren(...children.flat().filter(isPresent).map(toNode));
}

// ---------- 표 ----------

export function cell(label, content, cls) {
  return h('td', { 'data-label': label, class: cls }, content);
}

export function numCell(label, value) {
  return cell(label, fmt(value), 'num');
}

export function rowHeader(title, sub) {
  return h('th', { scope: 'row', class: 'row-title' }, title, sub ? h('span', { class: 'row-sub' }, sub) : null);
}

export function emptyRow(colspan, ...content) {
  return h('tr', { class: 'empty-row' }, h('td', { colspan: String(colspan) }, ...content));
}

// ---------- 막대: 장식이므로 aria-hidden, 숫자는 항상 옆에 텍스트로 둔다 ----------

function widthOf(ratio) {
  return (Number.isFinite(ratio) ? Math.max(0, Math.min(1, ratio)) : 0) * 100 + '%';
}

export function bar(ratio, tone) {
  const fillSpan = h('span', { class: tone ? 'tone-' + tone : null });
  fillSpan.style.width = widthOf(ratio);
  return h('span', { class: 'bar', 'aria-hidden': 'true' }, fillSpan);
}

export function barWithValue(ratio, tone) {
  return h('span', { class: 'bar-inline' }, bar(ratio, tone), percent(ratio));
}

export function stackBar(clean, harmful) {
  const total = clean + harmful;
  const cleanSpan = h('span', { class: 'tone-clean' });
  const harmfulSpan = h('span', { class: 'tone-harmful' });
  cleanSpan.style.width = widthOf(share(clean, total));
  harmfulSpan.style.width = widthOf(share(harmful, total));
  return h('span', { class: 'bar stack-bar', 'aria-hidden': 'true' }, cleanSpan, harmfulSpan);
}

// bar-list 항목: 라벨 / 막대(최댓값 기준) / 값(total이 있으면 비율 포함)
export function barItems(items, { total = 0, toneOf = () => null, nameOf = (label) => label } = {}) {
  if (!items.length) {
    return [h('li', null, h('span', { class: 'empty' }, '집계된 항목이 없습니다.'))];
  }
  const max = Math.max(...items.map((item) => item.count));
  return items.map((item) =>
    h(
      'li',
      null,
      h('span', { class: 'bar-label' }, nameOf(item.label)),
      bar(share(item.count, max), toneOf(item.label)),
      h('span', { class: 'bar-value' }, fmt(item.count), total > 0 ? h('small', null, percent(share(item.count, total))) : null)
    )
  );
}

export function definitionPairs(pairs) {
  return pairs.flatMap(([term, value]) => [h('dt', null, term), h('dd', null, value)]);
}

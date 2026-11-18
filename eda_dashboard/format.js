// 숫자·비율·라벨 표기. DOM에 의존하지 않는다.

const numberFormat = new Intl.NumberFormat('ko-KR');
const decimalFormat = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 1 });
const dateFormat = new Intl.DateTimeFormat('ko-KR', { dateStyle: 'long', timeStyle: 'short' });
const LABEL_NAMES = Object.freeze({ clean: '정상 (clean)', harmful: '유해 (harmful)' });

export function fmt(value) {
  return Number.isFinite(value) ? numberFormat.format(value) : '—';
}

export function fmtDecimal(value) {
  return Number.isFinite(value) ? decimalFormat.format(value) : '—';
}

export function fmtDate(isoText) {
  const date = new Date(isoText);
  return Number.isNaN(date.getTime()) ? String(isoText) : dateFormat.format(date);
}

export function fmtSeconds(value) {
  return Number.isFinite(value) ? fmtDecimal(value) + '초' : '기록 없음';
}

// ratio는 0~1. 1% 미만은 소수 둘째 자리까지 보인다.
export function percent(ratio) {
  if (!Number.isFinite(ratio)) return '—';
  const value = ratio * 100;
  const digits = value > 0 && value < 1 ? 2 : 1;
  return value.toFixed(digits) + '%';
}

export function share(part, total) {
  return total > 0 && Number.isFinite(part) ? part / total : NaN;
}

export function labelName(label) {
  return LABEL_NAMES[label] || label;
}

export function sumBy(items, key) {
  return items.reduce((acc, item) => acc + (Number.isFinite(item[key]) ? item[key] : 0), 0);
}

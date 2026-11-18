// report.json 입력 경계: 불러오기와 계약 검증을 여기서 한 번만 한다.

export const REPORT_URL = 'report.json';

const OBJECT_KEYS = Object.freeze(['provenance', 'summary', 'length', 'runtime']);
const ARRAY_KEYS = Object.freeze(['pipeline', 'sources', 'labels', 'splits', 'missing', 'targets', 'expressions', 'methods', 'notes']);

// kind: 'network' | 'http' | 'json' | 'contract' | 'render'
export class ReportError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = 'ReportError';
    this.kind = kind;
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// 문제가 있으면 사유 문자열, 없으면 null
function contractProblem(data) {
  if (!isPlainObject(data)) return '집계 파일이 JSON 객체가 아닙니다.';
  if (data.schema_version !== 1) return '지원하지 않는 schema_version입니다: ' + String(data.schema_version);
  const badObject = OBJECT_KEYS.find((key) => !isPlainObject(data[key]));
  if (badObject) return '필수 항목 ' + badObject + '이(가) 없거나 형식이 다릅니다.';
  const badArray = ARRAY_KEYS.find((key) => !Array.isArray(data[key]));
  if (badArray) return '필수 목록 ' + badArray + '이(가) 없거나 형식이 다릅니다.';
  const { length, runtime } = data;
  if (!isPlainObject(length.stats) || !Array.isArray(length.histogram) || !Array.isArray(length.by_source)) {
    return 'length 항목의 stats, histogram, by_source가 필요합니다.';
  }
  if (!isPlainObject(runtime.hadoop) || !isPlainObject(runtime.spark)) return 'runtime 항목에 hadoop과 spark가 필요합니다.';
  return null;
}

async function request(url) {
  try {
    return await fetch(url, { cache: 'no-store' });
  } catch (err) {
    if (err instanceof TypeError) {
      throw new ReportError(
        'network',
        '집계 파일(report.json)에 연결하지 못했습니다. 정적 파일 서버로 열었는지 확인하세요. 파일을 직접 열면 브라우저가 요청을 막습니다.'
      );
    }
    throw err;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (err) {
    if (err instanceof SyntaxError) throw new ReportError('json', '집계 파일(report.json)이 올바른 JSON이 아닙니다.');
    throw err;
  }
}

export async function loadReport(url = REPORT_URL) {
  const response = await request(url);
  if (!response.ok) {
    throw new ReportError('http', `집계 파일(report.json)을 불러오지 못했습니다. 서버 응답: ${response.status} ${response.statusText}`.trim());
  }
  const data = await readJson(response);
  const problem = contractProblem(data);
  if (problem) throw new ReportError('contract', '집계 파일 형식 오류: ' + problem);
  return data;
}

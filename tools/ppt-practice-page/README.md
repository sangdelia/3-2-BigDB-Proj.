# ppt-practice-page · PPT + git → 발표 연습 페이지

PPTX 파일 하나와 git 저장소로 발표 연습 페이지(`index.html` + `slides/`)를 만든다. 결과는 Claude artifact로 바로 게시할 수 있다. 프로젝트 내용은 템플릿에 들어 있지 않으므로 다른 발표에도 그대로 쓸 수 있다.

```bash
uv run tools/ppt-practice-page/build.py 발표.pptx --out /tmp/practice \
    [--extras extras.json] [--target 4:30] [--limit 5:00] [--repo .]
```

`uv`가 없을 때: `pip install python-pptx` 후 `python3 build.py …`. LibreOffice(`soffice`)와 `pdftoppm`(poppler-utils)이 필요하다. PPTX 옆에 같은 이름의 PDF가 있고 쪽수가 맞으면 변환하지 않고 그 PDF를 쓴다.

## 자동으로 가져오는 것

| 출처 | 내용 | 규칙 |
|---|---|---|
| PPT 슬라이드 | 이미지 | PDF → JPG, 가로 1280px (`--width`) |
| PPT 슬라이드 | 장 번호 | 슬라이드 안의 `01`·`A1` 같은 짧은 글자. 없으면 본문 `01…`, 부록 `A1…` |
| PPT 슬라이드 | 제목 | 제목 개체 → 없으면 위쪽 60% 안에서 가장 큰 글자. 여러 장에 반복되는 머리글·바닥글은 제외 |
| PPT 슬라이드 | 부록 | 장 번호가 영문자로 시작하거나, 제목이 `부록`·`Appendix`·`Backup`·`참고`로 시작하는 장부터. `--appendix-from`으로 지정할 수도 있음 |
| PPT 슬라이드 | 숨긴 슬라이드 | 기본 제외 (`--include-hidden`) |
| 발표자 노트 | 대본 | 노트 그대로. `--drop REGEX`나 extras `notes_drop`에 맞는 줄은 삭제 |
| 발표자 노트 | 장별 목표 시간 | 노트 맨 앞의 `[30초]`·`[30s]`·`[0:30]` 태그. 태그가 없으면 `--target`을 대본 길이 비율로 나누고, `--target`도 없으면 글자 수 ÷ 5.5자/초 |
| git | 저장소 정보 | 원격 주소(웹 링크로 변환, 인증 정보 제거), 브랜치, 기준 커밋, 커밋 안 된 변경 여부, 기간, 커밋 수 |
| git | 기여자 | 이름별 커밋 수. 이메일은 넣지 않음 |
| git | 수행 과정 | 날짜별 커밋 제목 (최근 `--max-commits`개). extras에 `timeline`이 있으면 그것을 대신 씀 |

## 페이지 구성

| 탭 | 내용 |
|---|---|
| 발표 연습 | 큰 슬라이드, 대본, 전체·장별 타이머, 계획 대비 늦음 표시, 장별 실제 시간 기록, 부록 카드 |
| 대본 전체 | 모든 장의 대본. claude.ai에서 열면 **대본 수정**이 가능하고, 저장한 내용은 페이지를 여는 모든 사람에게 보임 (db `scripts/<m01 · aA1>`) |
| 질의응답 | extras `qa`가 있을 때만 보임. 답 가리고 연습, 장 번호 칩 |
| 용어 설명 | extras `glossary`가 있을 때만 보임. 검색 |
| 수행 과정 | 저장소 정보, 기여자, extras `rules`, 타임라인 (extras 또는 git) |

## extras.json (모두 선택)

```jsonc
{
  "meta": {"title": "팀 발표 연습", "subtitle": "중간발표", "target": "4:30", "limit": "5:00",
           "rate": 6.0, "appendix_from": "A1", "process_lead": "타임라인 위에 붙는 설명"},
  "notes_drop": ["^근거 파일"],                  // 노트에서 지울 줄
  "author_names": {"kyu": "조규상"},             // git 작성자 이름 바꾸기
  "slides": {"03": {"sec": 25, "title": "…", "script": "…"}},   // 장별 덮어쓰기
  "qa": [{"g": "그룹", "items": [{"q": "질문", "a": "답", "refs": [3, "A1"], "avoid": "하지 말 말"}]}],
  "glossary": [{"g": "그룹", "items": [{"t": "용어", "d": "뜻", "h": "이 프로젝트에서", "s": 6}]}],
  "rules": [{"b": "원칙", "s": "설명"}],
  "timeline": [{"w": "2026-10-10", "h": "한 일", "d": ["세부"], "r": "결과", "refs": [5], "plan": false}],
  "links": [{"label": "시연", "url": "https://…"}]
}
```

`refs`·`s`에는 장 번호(`3`, `"03"`)나 부록 라벨(`"A1"`)을 쓴다. `qa`·`glossary`·`rules`·`timeline`의 문자열에는 `<b>` 같은 HTML을 쓸 수 있다. PPT와 git에서 가져온 글자는 모두 이스케이프된다. 예시는 `examples/hyeomochain.extras.json`에 있다.

## Claude artifact로 게시

빌드가 끝나면 `OUT/publish.json`이 생긴다. Claude Code에서 다음처럼 요청한다.

> OUT/publish.json대로 Artifact로 게시해줘

이렇게 하면 `file_path=OUT/index.html`, `files={slides/sNN.jpg: …}`, `capabilities={"db": {}}`로 게시된다. 같은 artifact를 갱신할 때는 그 URL을 함께 주고, `capabilities`는 생략한다.

## 출력

```
OUT/index.html      게시할 페이지 (데이터 포함)
OUT/slides/sNN.jpg  슬라이드 이미지 (NN = PPT 안의 원래 순서)
OUT/deck_data.json  추출 결과 (확인·수정용)
OUT/publish.json    Artifact 게시 인자
```

# 프로젝트 규칙

## 환경
- 파이썬은 venv 로 실행: `.venv/Scripts/python.exe`

## 자율 개발 중 명령어
- cd+&& 조합을 하지말고 절대 경로 단일 명령으로만 실행
- 내 수동 승인을 묻는 예시 (나쁜 예시): `cd "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3" && ".venv/Scripts/python.exe" synth_engine_v3/scripts/phase0_smoke.py --batch 64 --batches 30 2>&1`
- 내 승인을 묻지 않는 예시 (좋은 예시): `"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/.venv/Scripts/python.exe" "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v3/scripts/phase0_smoke.py" --batch 64 --batches 30 --out "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v3/out/01_phase0_smoke" 2>&1`

## 자율 모드 규칙
이 프로젝트는 `.claude/settings.local.json` 에서 `bypassPermissions` 활성. 멈추지 않고 계획을 실행하되 다음 규칙은 반드시 지킨다.

### 출력 폴더 명명
테스트·생성 산출물은 **순서 번호 prefix** 로. 사용자가 어느 폴더가 최신인지 직관적으로 알 수 있어야 함.
- 형식: `out/NN_purpose/` (예: `out/01_smoke/`, `out/02_phase0_pytorch_check/`, `out/03_mask_adapter/`)
- 번호는 zero-pad 2자리, 단조 증가
- v2 의 `out/ehanja_test`, `out/multi_test` 같은 작명은 **금지**

### 스크립트 파일 명명
출력 폴더와 같은 원칙. 단 Python import 제약 때문에 **실행 스크립트만** prefix.
- **실행 (CLI entry point) 스크립트**: `NN_name.py` 로 번호 prefix
  - 예: `00_phase0_smoke.py`, `04_phase4_verify.py`, `05_generate_v3.py`
  - 이 파일들은 `python file.py ...` 로만 실행되고, 다른 파일에서 import 되지 않음
- **라이브러리 모듈** (다른 스크립트에서 import 하는 파일): prefix **금지**, 이름 그대로 유지
  - 예: `mask_adapter.py`, `pipeline_gpu.py`, `style_gpu.py`, `augment_gpu.py`
  - Python 은 `01_foo` 같이 숫자로 시작하는 모듈명을 import 할 수 없으므로 prefix 불가
- 라이브러리가 여러 phase 에 걸쳐 갱신되면 파일 상단 주석에 "added at phase X" 표기

### 단계 완료 보고
계획 문서 (예: `synth_engine_v3/V3_DESIGN.md`) 의 체크리스트 항목을 완료할 때마다 **즉시 [x] 표시 + 1줄 코멘트**. 사용자가 진행률을 한 눈에 파악 가능해야 함.

### v2 정합성 점검
각 Phase 끝에서 **v2 의 원래 목적과 괴리 없는지** 확인:
- v2 ENGINE_V2_DESIGN.md 의 의도된 동작 (3블록 분리 / source 다양성 / per-stroke 변주 의미론 등) 이 v3 에서도 보존되는가
- 잃어버린 기능 / 의도 변경 발견 시 V3_DESIGN.md "v2 와의 호환성 주의" 섹션에 기록

### 다음 Phase 진행 조건
위 3개 (출력 명명 + 체크리스트 업데이트 + v2 정합성 점검) 가 모두 충족된 경우에만 다음 Phase 자율 시작. 미흡하면 사용자에게 보고 후 멈춤.

### Self-discipline (bypass 모드여도 유지)
- `git commit` 은 명시 요청 시에만
- destructive 작업 (외부 파일 삭제, db_src 덮어쓰기 등) 은 사전 status report
- smoke test 우선 — 큰 배치 전에 50 샘플 검증
- 같은 에러 3회 반복되면 멈추고 사용자에게 보고
- 각 phase 완료 시 한 줄 진행 보고

# userEmail
yoonguri21@gmail.com

# currentDate
Today's date is 2026-04-19.

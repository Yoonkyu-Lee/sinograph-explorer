# Missing Classes Audit — corpus 94_production_102k_x200

작성: 2026-04-26.  대상: `synth_engine_v3/out/94_production_102k_x200/`.

8h 39m 의 corpus 생성 후 102,944 class 중 **4,775 (4.6%) 가 폰트 미지원으로
미렌더링** 되어 학습 데이터 0건. 본 문서는 어느 코드포인트가 누락됐는지,
왜 누락됐는지, 그리고 후속 처리를 기록.

전체 누락 list 는 [missing_classes.txt](../synth_engine_v3/out/94_production_102k_x200/missing_classes.txt).

---

## 1. Coverage 표

| Unicode block | Rendered | Total | Coverage | Missing |
|---|---:|---:|---:|---:|
| CJK Unified Ideographs | 20,990 | 20,990 | **100.0%** | 0 |
| CJK Ext A | 6,592 | 6,592 | **100.0%** | 0 |
| CJK Ext B (Unicode 3.1, 2001) | 42,708 | 42,708 | **100.0%** | 0 |
| CJK Ext C (Unicode 5.2, 2009) | 4,160 | 4,160 | **100.0%** | 0 |
| CJK Ext D (Unicode 6.0, 2010) | 220 | 220 | **100.0%** | 0 |
| CJK Ext E (Unicode 8.0, 2015) | 5,758 | 5,758 | **100.0%** | 0 |
| CJK Ext F (Unicode 10.0, 2017) | 7,463 | 7,463 | **100.0%** | 0 |
| CJK Ext G (Unicode 13.0, 2020) | 4,933 | 4,933 | **100.0%** | 0 |
| CJK Ext H (Unicode 15.0, 2022) | 4,186 | 4,186 | **100.0%** | 0 |
| CJK Ext I (Unicode 15.1, 2023) | 622 | 622 | **100.0%** | 0 |
| CJK Compat Ideographs | 472 | 472 | **100.0%** | 0 |
| CJK Radicals Suppl | 7 | 7 | **100.0%** | 0 |
| **CJK Ext J (Unicode 16.0, 2024)** | 0 | 4,291 | **0.0%** | 4,291 |
| **CJK Compat Supplement** | 58 | 542 | **10.7%** | 484 |
| **합계** | **98,169** | **102,944** | **95.4%** | **4,775** |

---

## 2. 누락 원인 분석

### 2.1 CJK Ext J — 4,291 개 (90% of missing)

- **블록**: U+323B0 — U+3347F
- **Unicode 16.0** (2024-09 출시) 신규 영역
- **Windows 11 시스템 폰트는 아직 미지원** — Microsoft YaHei, SimSun, MS Mincho, Yu Gothic, Malgun Gothic 등 어느 것도 이 블록을 cover 하지 않음
- 글자 성격: 매우 희귀한 역사·방언·인명용 문자

**해결 가능성**: 폰트 측 시간 문제. `noto-cjk` dev branch 또는 향후 Windows
폰트 업데이트 시 자연 해결. 현 시점에선 불가피.

### 2.2 CJK Compat Supplement — 484 개 (10% of missing)

- **블록**: U+2F800 — U+2FA1F
- **블록 자체가 호환성 변형 영역** — 이미 CJK Unified / Ext B 에 본체가 존재
  하는 글자의 시각적 변형을 legacy 시스템 (구 한국·일본 정부 표준) 호환을
  위해 별도 코드포인트로 등록한 영역
- 본체 글자는 이미 100% cover 되어 있음 → **의미상 손실 0**
- 폰트가 cover 하지 않는 것이 normal 상태 (실제로 이 영역을 그리는 폰트는
  거의 없음)

**해결 가능성**: 의미가 없어 미해결도 무방.

---

## 3. 학습 영향 평가

**진짜 사용 한자 (CJK Unified, Ext A-F, 한국·중국·일본 일상 한자)**: **100% cover**.

학습-측면 손실: 사실상 0. 이유:
- 누락된 4,775 글자는 평생 OCR 입력으로 들어올 가능성이 극히 낮음
- 들어오더라도, 그 글자의 본체 (Compat Supplement 의 경우) 또는 시각적
  유사 글자를 모델이 추론할 가능성 있음

---

## 4. 후속 처리 — class_index 트리밍

### 4.1 결정

- **트리밍 진행** — 102,944 → 98,169
- 사유: dead 4,775 슬롯이 final classifier 의 2.4M params (4.6%) 를 차지.
  inference latency 미세 영향 + ONNX export 사이즈. 정리해두는 게 깔끔.

### 4.2 스크립트

`synth_engine_v3/scripts/12_trim_class_index.py` — 다음 작업 수행:
1. 현 class_index.json → `class_index_full.json` 백업
2. alive 클래스만 → 새 class_index.json (idx 0..98168 연속)
3. 모든 3,927 shards 의 `labels` 배열을 새 idx 로 remap
4. shard 는 atomic 한 temp+rename 으로 in-place 갱신

### 4.3 트리밍 후 영향

- **train_engine_v3 코드 변경 불필요** — shard 가 새 idx 로 정합되고 class_index
  도 일관됨. 모델의 final softmax 만 102,944 → 98,169 로 작아짐
- **aux_labels.npz 재생성 필요** — class_index 가 바뀌었으므로 sidecar 도
  갱신
- **class_index_hash 변경됨** — 학습 시 train_engine_v3.aux_labels 가 hash
  검증함. 재생성 후 자동 일치
- **복구 가능성** — `class_index_full.json` 백업 + remap 역변환으로 언제든
  롤백 가능. 단 shard 는 새 idx 로 변환됐으므로, 롤백은 class_index 만 돌리는
  게 아니라 shard 도 다시 remap 해야 함

---

## 5. 향후 권고

- **Ext J 학습이 필요해질 시**: noto-cjk dev branch 설치 → corpus 재생성. Ext J 의
  본질적 희귀성 때문에 현 demo / 실용 OCR 에선 우선순위 낮음.
- **새 corpus 생성 시 사전 검사 추가**: corpus 생성 직전 font coverage 검사를
  돌려서 누락 비율을 미리 알리면 좋음. `class_list_practical.jsonl` 단계에
  font-availability 필터를 넣는 게 cleanest. 향후 개선 사항.

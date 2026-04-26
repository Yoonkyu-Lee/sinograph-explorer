# Two-Stage Workflow: Synthetic Generation → Training

## Summary

Lab 3의 OCR 파이프라인은 두 단계로 나뉜다. 두 단계의 input/output 방향이
서로 **거울상**이라는 점이 설계의 핵심이다.

```
Stage 1 (now)                          Stage 2 (later)
  input  : 문자 1개의 식별자             input  : (glyph PNG, 라벨) N쌍
           (literal "鑑" / notation       (라벨은 동일 문자 식별자)
            "U+9451" / 정수 0x9451
            → 전부 동일 대상)
  engine : 커스터마이즈 이미지 생성기      engine : CNN 분류기 + INT8 양자화
  output : glyph PNG 다수 (+ 라벨)        output : 배포 가능한 TFLite 모델
```

Inference 시점(실기기)은 Stage 2 모델의 사용 단계이며 다시 한 번 방향이
뒤집힌다.

```
Stage 2 inference (Pi + Coral TPU)
  input  : glyph (카메라 픽셀)
  engine : OCR 엔진 → confidence gate → fallback 분류기 (Stage 2 산출물)
  output : codepoint 예측 + DB lookup 결과
```

즉 전체 흐름에서 **codepoint ↔ glyph 변환이 두 번** 일어난다.
Stage 1은 codepoint → glyph, Stage 2는 glyph → codepoint.

---

## 용어 확정

### 문자의 세 가지 표현 층위

`鑑`을 예로 들면:

| 층위 | 한국어 | 영어 | 값 (예시) | 정체 |
|------|-------|-----|----------|-----|
| 정수 | 코드포인트 | **codepoint** | `0x9451` (= `37969`) | Unicode가 부여한 정수 하나. 이게 "진짜 codepoint" |
| 표기 | 코드포인트 표기 | codepoint notation | `"U+9451"` | 그 정수를 표기하는 표준 문자열 형태 |
| 실제 문자 | 문자 리터럴 | character literal | `"鑑"` | 그 codepoint로 인코딩된 길이-1 string |
| 시각적 모양 | 자형 / 글리프 | **glyph** | (이미지) | 실제로 그려진 모양. 폰트·스타일·손글씨마다 다름 |

변환:
```python
ord("鑑")                # → 37969           (literal → codepoint 정수)
f"U+{ord('鑑'):04X}"     # → 'U+9451'        (literal → notation)
chr(0x9451)              # → '鑑'            (정수 → literal)
int("9451", 16)          # → 37969           (notation 후반 → 정수)
```

### 데이터에서는 어떻게 저장하나

canonical DB는 **notation과 literal을 병렬로 저장**한다:

```json
{ "character": "鑑", "codepoint": "U+9451" }
```

진짜 codepoint(정수 `37969`)는 직접 저장하지 않고 필요 시 복원한다.

### 기타 용어

| 한국어 | 영어 | 설명 |
|-------|------|------|
| 서체 / 폰트 | typeface / font | 같은 스타일로 묶인 glyph 세트 |
| 이체자 | variant character | 같은 글자의 다른 형태. **Codepoint 자체가 다름** (예: 學 U+5B78 / 学 U+5B66 / 斈 U+6588) |

### 한 줄 요약

프로젝트 전반에서 **glyph = 이미지, codepoint = 라벨**이라고 기억하면
input/output 혼동이 사라진다.
"codepoint"라는 단어는 맥락에 따라 정수 자체 / notation 문자열 / 그 정수가
가리키는 literal을 모두 느슨하게 의미할 수 있지만, **같은 문자 동일성**을
가리킨다는 점에서 혼용해도 의미는 통한다.

---

## Stage 1 — Synthetic Dataset Generation (현재 작업)

### 목적
희귀/지역/이체/스타일 한자에 대해 `(glyph, codepoint)` 학습 쌍이
실세계에 거의 없으므로, **codepoint로부터 glyph를 합성**해서 쌍을
프로그램적으로 만든다.

### Input
- 문자 1개의 식별자. 실제로는 다음 중 어느 형태로 넘겨도 동등:
  - character literal: `"鑑"`  ← 현재 CLI 컨벤션 (`render_systemfonts.py 鑑`)
  - codepoint notation: `"U+9451"`  ← 파일명/로그 컨벤션
  - 정수 codepoint: `0x9451`  ← 내부 비교/저장
- canonical DB row의 보조 정보 (선택적)
  - MakeMeAHanzi `stroke_svg_paths`, `stroke_medians` — 폰트 없는 문자의 fallback 렌더 소스
  - `variants.family_members` — 이체자 관계 (혼동 분석/negative sampling용, v1에선 사용 안 함)
  - `source_flags`, `total_strokes` — tier/seed selection용

### Engine: `synth_engine_v1/` (v2 in `synth_engine_v2/` 작업 중)

```
base renderer          augment pipeline
  (codepoint → RGB       (RGB → RGB)
   clean glyph)           다양한 실세계 조건 시뮬레이션

  ├─ font               ├─ geometric: rotate, perspective, shear, scale
  │   (46 CJK faces)    │              translate
  ├─ stylized           ├─ photometric: brightness, contrast, gamma,
  │   (포스터 19종)      │               saturation, color_jitter, invert
  └─ stroke-SVG         ├─ degradation: gaussian/motion blur, gauss/
      (MakeMeAHanzi     │               salt-pepper noise, downscale,
       fallback)        │               JPEG compression
                        ├─ scan_sim:   paper texture, ink bleed, binarize,
                        │              shadow gradient, vignette
                        └─ camera_sim: defocus, chromatic aberration,
                                       lens distort, low-light + noise
```

- base renderer는 codepoint → clean glyph 변환을 담당.
- augment pipeline은 각 op이 파라미터 range를 받아 per-sample 랜덤화.
- config 한 장으로 `clean_fonts / camera_sim / poster_heavy / full_mix` 등
  프리셋 구성 가능.

### Output
- `out/<notation>/*.png` — 256×256 RGB glyph 이미지
  (폴더명은 notation 형태 `U+9451/` 로 통일 — 고정폭 + sortable)
- 각 파일의 라벨은 **입력으로 넣은 문자 식별자 자체** (라벨 잡음 없음)
- sidecar metadata(`json` 또는 CSV)에 base source / 적용된 aug 목록 기록
  (provenance + 실패 모드 분석용)

### 왜 이 방식인가
- 희귀 codepoint는 real labeled image가 존재하지 않음 → synthetic이 유일한 길
- 라벨을 이미 아는 상태에서 시작하므로 **라벨 노이즈 0**
- 생성기를 조정하면 학습 분포를 우리가 완전히 제어 가능
  (카메라-heavy, 포스터-heavy, 저조도 등)
- 폰트가 없는 tail codepoint도 stroke-SVG fallback으로 커버 가능

### Scope 제약 (v1)
- 한 번에 한 codepoint만 처리 (배치는 나중에)
- ground-truth는 codepoint 동일성만. 이체자 간 혼동은 Stage 2에서 평가
- stroke-SVG 경로는 MakeMeAHanzi 커버리지(9,574자) 한도에서만 동작

---

## Stage 2 — Classifier Training + Edge Deployment (나중 작업)

### 목적
Stage 1이 만든 `(glyph, codepoint)` 쌍으로 분류기를 학습하고,
Raspberry Pi + Google Coral Edge TPU에 배포 가능한 형태로 변환한다.

### Input
- Stage 1이 만든 이미지셋 (glyph PNG + codepoint 라벨)
- tier별 codepoint 선정 목록 (common N자 / extended M자 / tail K자)

### Engine
1. **학습 (PC, RTX 4080)**
   - backbone: ResNet/EfficientNet 등
   - loss/sampling: class-balanced sampling, focal loss, label smoothing
   - long-tail 대응: logit adjustment 또는 embedding + kNN retrieval 헤드
2. **양자화**
   - PyTorch → ONNX → TFLite
   - Post-Training Quantization INT8 (calibration set은 Stage 1 샘플 일부)
3. **변환 검증**
   - 양자화 전후 top-1 accuracy drop 측정
   - Coral TPU compile 통과 확인

### Output
- `.tflite` 모델 파일 (Coral-compiled)
- `deploy/` 번들: 모델 + 라벨맵 + DB lookup 인덱스

### Inference 시점 (Pi + Coral)
- Pi 카메라 → OCR 1차(표준 엔진, 예: PaddleOCR/Tesseract)
- confidence 낮으면 → Coral 분류기로 fallback
- 분류기 예측 codepoint → canonical DB lookup (readings/meanings/family)
- 디스플레이

### 평가
- synthetic held-out (font-unseen split)
- **lab2 실패 케이스 real set** — baseline vs ours
- tier별 top-1/top-5
- **family-aware accuracy** — 예측이 틀려도 canonical family 내면 부분점수

---

## 두 단계가 맞물리는 지점

| 경계 | 내용 |
|------|------|
| Stage 1 output → Stage 2 input | glyph PNG + codepoint 라벨. 중간 형식은 파일시스템(간단) 또는 webdataset/tar(확장 시) |
| Stage 2 output → Stage 2 inference | TFLite 모델. Coral Edge TPU 제약(int8, 연산 지원) 준수 |
| Stage 2 inference → DB | 예측 codepoint로 canonical DB 조회 → Sinograph Explorer 표시 |

Stage 1의 augment 분포가 **Stage 2 inference 환경(라이브 카메라)을 얼마나
잘 근사하느냐**가 실제 성능의 상한을 결정한다. 따라서 Stage 1은 단순한
전처리가 아니라 이 프로젝트의 **기술적 핵심** 중 하나다.

---

## 체크포인트

- [x] 용어 확정 (codepoint vs glyph vs font vs variant)
- [x] Stage 1 base renderer (font 경로, stylized 경로)
- [ ] Stage 1 augment pipeline (geometric/photometric/degradation/scan/camera)
- [ ] Stage 1 unified generator CLI (config 기반)
- [ ] Stage 1 SVG stroke fallback renderer
- [ ] tier 선정 및 codepoint 리스트
- [ ] Stage 2 학습 루프
- [ ] Stage 2 양자화 + TFLite 변환
- [ ] Stage 2 Coral 배포 및 Pi 연동
- [ ] Family-aware 평가 스크립트

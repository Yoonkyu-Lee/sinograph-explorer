# OCR Inference Test Plan — Mini 모델 (1k class × 5 epoch)

현재 학습 중인 `out/02_mini_1k/` 모델의 실전 광학 인식 검증 계획.

## 배경

**모델 사양**:
- ResNet-18 @ 128², AMP
- 클래스: 1,000 (sorted notation U+4E00 ~ 약 U+51E8, 주로 저-codepoint Korean hanja)
- 학습 데이터: Stage 1 합성 PNG × 클래스당 ~450 장 (10% val split)
- Stage 2 transforms: Resize + CenterCrop + GPU normalize. RandomCrop ±4px

**한계 인지**:
- **Single-character classifier** — 한 이미지 = 한 글자. 여러 글자 이미지는 crop 후 따로 처리
- **1k class 범위 밖은 못 맞춤** — 모르는 label 이라 high confidence wrong answer 나올 수 있음
- detection 기능 없음 → `31_predict_batch.py` 가 직접 전체 이미지 = 한 글자 가정

---

## 3 단계 테스트 구조

목적이 다른 3 tier. 각각 다른 질문에 답함.

### (A) In-distribution sanity — 파이프라인 정확성
- **소스**: Stage 1 corpus (`synth_engine_v3/out/42_production_mobile/`) 에서 직접 복사
- **샘플 수**: 10~20 장 (각 추천 12 자에서 각 1~2 장)
- **기대 top-1**: ≥ 99%
- **답하는 질문**: 전처리 (Resize / Normalize / GPU pipeline) 가 학습과 정확히 일치하는가
- **실패시 의미**: 모델 문제 아님, **inference 코드 버그**. Lab2 FaceNet `[-1,1]` normalize mismatch 와 같은 류

### (B) Near-OOD — 분포 내 unseen
- **소스**: Stage 1 엔진 재실행 (다른 seed 로 새 500 장 생성). **지금은 선택**, 시간 남을 때
- **기대 top-1**: 95 ~ 98%
- **답하는 질문**: 과적합 여부 — 학습 분포에서 뽑은 새 샘플도 맞추는가
- 현 단계 **우선순위 낮음**. A 와 C 가 먼저

### (C) Far-OOD — 실제 카메라 이미지 ⭐ 핵심
- **소스**: 폰 / 카메라 / 웹 스크린샷 — 학습에 쓰이지 않은 실제 이미지
- **샘플 수**: 6~12 장 (최소 6 이면 midpoint 증빙 충분)
- **기대 top-1**: 미지수 (proposal 의 핵심 질문)
- **답하는 질문**: 합성 데이터로만 학습해도 실제 이미지 인식되는가 = **novelty 증명**

---

## 추천 테스트 문자 (12 자)

Mini 범위 (U+4E00 ~ 약 U+51E8) 내 고른 다양성 균형 세트.

| # | 문자 | codepoint | 획수 | 의미 | 찾기 난이도 |
|---|---|---|---|---|---|
| 1 | 一 | U+4E00 | 1 | 한 일 | 쉬움 |
| 2 | 二 | U+4E8C | 2 | 두 이 | 쉬움 |
| 3 | 三 | U+4E09 | 3 | 석 삼 | 쉬움 |
| 4 | 上 | U+4E0A | 3 | 위 상 | 쉬움 (엘리베이터) |
| 5 | 中 | U+4E2D | 4 | 가운데 중 | 매우 쉬움 (중식당) |
| 6 | 人 | U+4EBA | 2 | 사람 인 | 매우 쉬움 |
| 7 | 入 | U+5165 | 2 | 들 입 | 쉬움 (入口) |
| 8 | 元 | U+5143 | 4 | 으뜸 원 | 쉬움 (元旦) |
| 9 | 光 | U+5149 | 6 | 빛 광 | 쉬움 (光州/상호) |
| 10 | 信 | U+4FE1 | 9 | 믿을 신 | 중간 (은행) |
| 11 | 保 | U+4FDD | 9 | 보전할 보 | 중간 (保險) |
| 12 | 億 | U+5104 | 15 | 억 억 | 어려움 (복잡도 stress test) |

### 의도된 설계

- **1-7 간단 ~ 중간**: 기본 맞춤 확인
- **8-11 실생활 빈출**: 한국 간판/책 잘 찾기
- **12 복잡**: 15 획 stress test
- **人-入 / 一-二-三 쌍**: 혼동 유발 가능. 올바른 구분 부가 증거

### 이미지 출처 제안

| 소스 | 추천 문자 |
|---|---|
| 중식당 간판 | 中, 人 |
| 엘리베이터 버튼 | 上 (B1, 옥상) |
| 서점 / 도서관 책 제목 | 信, 光, 人 |
| 한자 문제집 / 학습지 | 一 ~ 三, 元, 億 |
| 구글 이미지 검색 ("一 書") | 전부 (서체 다양성) |

### 촬영 / 수집 주의

- **한 글자씩 crop**: single-char classifier. "入口" 면 入 / 口 따로
- **정사각형 근사**: 글자 주변 10~20 % 여백
- **검은 글자 + 밝은 배경** 선호 (Stage 1 분포 가까움)
- **해상도**: 무엇이든 OK (resize 처리). 100 px 미만은 약함

---

## 파일명 규칙 (자동 채점)

```
test_images/
├── A_in_dist/
│   ├── 000000_U+4E00_malgun-0.png    # corpus 에서 복사, 원래 이름 유지
│   ├── 042195_U+4FE1_msyhbd-1.png
│   └── ...
└── C_real/
    ├── IMG_001_U+4E00.jpg            # 파일명에 정답 codepoint 박음
    ├── photo_002_U+5149.png
    ├── signboard_U+4E2D.jpg
    └── ...
```

**필수 패턴**: 파일명 어딘가에 `U+XXXX` 포함. `31_predict_batch.py` 가 regex 로 추출해 ground truth 로 사용.

---

## 실행 스크립트

### 단일 이미지 (quick check)
```bash
python train_engine_v1/scripts/30_predict.py \
  --ckpt train_engine_v1/out/02_mini_1k/ckpt_epoch_05.pth \
  --class-index train_engine_v1/out/02_mini_1k/class_index.json \
  --config train_engine_v1/configs/resnet18_t1_mini.yaml \
  --image path/to/image.jpg \
  --topk 5 \
  --family-db sinograph_canonical_v2/out/sinograph_canonical_v2.sqlite
```

### 폴더 일괄 (TODO — 작성 예정)
```bash
python train_engine_v1/scripts/31_predict_batch.py \
  --ckpt ... --class-index ... --config ... \
  --image-dir train_engine_v1/test_images/C_real \
  --out train_engine_v1/out/02_mini_1k/test_C_real_results.jsonl
```

출력:
- `results.jsonl` — 샘플마다 한 줄, 경로 / top-5 (codepoint + prob) / ground truth / correct 여부
- 콘솔 요약 — top-1 / top-5 accuracy, 실패 케이스 리스트

---

## 결과 해석 가이드

### (A) 결과
- **top-1 ≥ 99%**: 파이프라인 정상. (C) 로 진행
- **top-1 < 90%**: 전처리 불일치 의심. training transform vs inference transform 대조
- **top-1 50% 근처**: 모델 가중치 로딩 실패 또는 class_index mismatch

### (C) 결과 (proposal 의 핵심 지표)
- **top-1 > 80%**: 우수. Stage 1 augment 가 real-camera 분포를 잘 근사한다는 강한 증거
- **top-1 50 ~ 80%**: 합리적. 일부 폰트/각도/조명에서 약함. 개선 여지 있지만 demo 가능
- **top-1 < 50%**: 심각. Stage 1 augment 재설계 필요 가능. 실패 케이스 분석 → 어떤 종류 입력에서 깨지는지
- **top-5 > top-1**: family member / 혼동 쌍에서 밀림 정도. proposal 의 "family-aware partial credit" 개념으로 완화 가능

### 실패 케이스 예상 유형

| 유형 | 예시 | 원인 | 대응 |
|---|---|---|---|
| 저조도 | 어두운 간판 사진 | Stage 1 `low_light` prob 0.2 로 부족 | v3 config 에서 prob ↑ |
| 강한 motion blur | 움직이는 차 안에서 촬영 | `motion_blur` v3 아직 미포팅 | v3 엔진 보강 |
| 특이한 서예체 | 붓글씨 | 현재 font 46종에 서예체 적음 | 폰트 풀 확장 |
| 극단 각도 | 위에서 내려본 관점 | `perspective` strength 로 커버하는지 확인 필요 | strength range 넓힘 |

---

## Midpoint 증빙 최소 요건 (4/20)

1. **(A) 결과 표** — 12 자 × 1 장 = 12 샘플 top-1 99%+
2. **(C) 결과 표** — 6~8 실사진 top-1 / top-5
3. **(C) 샘플 이미지 + 예측 grid** — 슬라이드용, 2 ~ 3 성공 + 1 실패 케이스
4. **(curves.png)** — 학습 loss 곡선 (이미 자동 생성됨)

---

## 작성 예정

- [ ] `scripts/31_predict_batch.py` — 폴더 glob + 파일명 자동 채점
- [ ] (선택) `scripts/32_test_report.py` — JSONL → markdown 결과 표 포매터

## 변경 이력

- 2026-04-20 — 초안. Mini 모델 학습 중. 추천 12자 선정, 3-tier 체계 확정.

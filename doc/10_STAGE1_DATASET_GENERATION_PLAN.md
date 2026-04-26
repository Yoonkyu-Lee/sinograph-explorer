# Stage 1 Dataset Generation — Production Plan

v3 엔진 (`synth_engine_v3/`) 의 **활용 문서**. 내부 설계는 `synth_engine_v3/V3_DESIGN.md`. 이
문서는 Stage 2 학습에 쓸 이미지셋을 어떤 구성·규모·순서로 만들지 결정하기 위한
운영 가이드.

## 상위 맥락 (왜 이 문서가 있나)

프로젝트 전체 흐름은 두 단계 — `doc/07_TWO_STAGE_WORKFLOW.md` 참고.
- **Stage 1 (이 문서 범위)**: codepoint → glyph 이미지 생성 → (glyph, codepoint) 쌍
- **Stage 2 (이후)**: 이 쌍으로 CNN 학습 → INT8 TFLite → Coral TPU 배포

v3 엔진은 이제 v2 의 모든 시각 원리 (22 style + 25 augment + 6 base_source kinds +
stroke_ops) 를 100% 보존한 채로 GPU 배치 처리로 재구현되어 있고, 실측 기준
production throughput ≈ **167 s/s (full-mix)** / **270 s/s (font-only)** 에 도달
했다. 즉 **엔진 단은 준비 완료**. 남은 것은 "**어떤 글자를 몇 장 만들지**" 를 정
해서 실행하는 것.

---

## "답 (label)" 은 어디에 있나

### 원칙
Stage 1 의 입력은 항상 **문자 1개의 식별자** — literal (`"鑑"`) / notation (`"U+9451"`) /
정수 codepoint (`0x9451`). 이들은 전부 같은 대상을 가리킨다 (`doc/07` 의 "문자의
세 가지 표현 층위" 표). 생성기는 그 식별자 하나를 받아 이미지 N 장을 만드는
함수이므로 **라벨은 항상 그 입력 식별자**. 즉 **라벨 노이즈 0** 이 프로젝트의
핵심 가정.

### 라벨이 실제 저장되는 위치

#### (1) 파일명 — 1차 label 소스
v3 `generate_corpus_v3.py` 의 기본 출력 포맷:
```
out/corpus_<config_tag>/{idx:06d}_{notation}_{picked_source}.png
예:  out/corpus_full_random_multi/000042_U+9451_malgun-0.png
                                           ↑           ↑
                                           label      provenance
```
여기서 `notation = f"U+{ord(char):04X}"` (예: `U+9451`). 이 6자리 hex 에서 바로
정수 codepoint 복원 가능: `int("9451", 16) → 37969`, 그리고 `chr(37969) → "鑑"`.
**파일명만으로 학습이 가능**.

#### (2) `corpus_manifest.jsonl` — 구조화된 label + provenance (선택)
`--metadata` 플래그를 켜면 per-sample JSON Line 하나씩:
```json
{
  "idx": 42,
  "char": "鑑",
  "notation": "U+9451",
  "block": "CJK_Unified",
  "base_source_kind": "font",
  "picked_source": "malgun-0",
  "tag": "malgun-0",
  "seed": 1000042,
  "filename": "000042_U+9451_malgun-0.png"
}
```
`--metadata` 를 권장 — Stage 2 의 custom `Dataset` 구현이 훨씬 단순해지고, 실패
모드 분석 (특정 font / source_kind 에서 accuracy drop 등) 에 필수.

#### (3) 단일 글자 드라이버 (`05_generate_v3.py`) 는 폴더 분리
```
out/<run>/<notation>/{idx:04d}_{notation}_{tag}.png
예:  out/prod/U+9451/0000_U+9451_malgun-0.png
```
PyTorch `ImageFolder` 가 그대로 읽을 수 있는 layout.
하지만 production 규모 (글자 수천) 에서는 corpus 드라이버가 훨씬 빠르므로 이 레이
아웃은 debug/미니 실험용.

### Stage 2 에서 label 을 끌어 쓰는 세 가지 방법
- **(A) ImageFolder 호환 layout 으로 재조직** — corpus 산출물을
  `notation/filename` 구조로 이동/링크. Stage 2 코드가 제일 단순 (post-process
  1회).
- **(B) manifest.jsonl 기반 custom `Dataset`** — 재조직 없이 `__getitem__` 안에서
  manifest 의 `notation` 을 class index 로 매핑. 제일 빠름 (파일 이동 비용 0).
- **(C) 파일명 파싱 `Dataset`** — manifest 없어도 `filename.split("_")[1]` 로
  notation 복원. fallback.

현재 corpus 레이아웃 + `--metadata` 조합이면 세 방법 모두 가능. 권장은 **(B)**.

---

## 실전 진행 순서 (4 phase)

### Phase A — 품질 파일럿 (5~10분, 생성)
엔진·config·augment 분포가 실제로 원하는 "real-camera 시뮬"인지 **사람 눈으로**
확인. Stage 2 없이 종료.

```
python 10_generate_corpus_v3.py \
  --config synth_engine_v2/configs/full_random_multi.yaml \
  --total 5000 --pool union --strategy stratified_by_block \
  --mask-workers 8 --save-workers 4 --batch-size 64 --metadata \
  --out synth_engine_v3/out/30_pilot_5k --seed 0
```
**체크 사항**:
- 40~60 샘플 랜덤 열어서 글자 모양이 원본에서 너무 벗어나지 않는지 (라벨 무효화
  위험: 과격한 elastic, drop_stroke 전부 적용 시)
- 배경·조명·각도 분포가 실제 카메라 조건에 가까운지
- ehanja/kanjivg/MMH 필체 variation 이 font 와 구별되는지

Pass 기준: **샘플 95%+ 가 codepoint 를 시각적으로 맞게 표시**. 실패하면 config
tweak (elastic alpha, background.scene prob 등) 후 재생성.

### Phase B — 학습 수렴 파일럿 (생성 2~5분 + 학습 30분~1시간)
"class 당 몇 장이 충분한가" 를 경험적으로 확정. 아직 production 아님.

```
python 10_generate_corpus_v3.py \
  --config synth_engine_v2/configs/full_random_multi.yaml \
  --total 25000 --pool ... \
  --out synth_engine_v3/out/31_convergence_pilot --metadata --seed 1
```
- class 50~100개 × class 당 250~500 샘플 (총 25k~50k)
- Stage 2 학습 루프 돌려 **validation accuracy 수렴 곡선**만 확인
- 목표: class 당 몇 장에서 plateau 하는지 관찰. 아직 plateau 전이면 production
  단계에서 더 많이 뽑아야 함. 이미 plateau 이면 현 수준으로 충분.

**결정 output**: "class 당 X 장" 이라는 숫자 하나.

### Phase C — Production 생성 (1.5~8시간)

Phase B 에서 나온 숫자 × Phase D 에서 정한 class 리스트 = `--total`. 실행 1회.

예시 (class 6k × 300 = 1.8M):
```
python 10_generate_corpus_v3.py \
  --config synth_engine_v2/configs/full_random_multi.yaml \
  --total 1800000 --pool <tier pool 또는 custom> \
  --strategy stratified_by_block \
  --mask-workers 8 --save-workers 4 --batch-size 64 --metadata \
  --out synth_engine_v3/out/40_prod_<tag> \
  --seed <고정>
```
**디스크**: 256×256 PNG per-sample ~30~60 KB (compress_level=1). 2M = 60~120 GB,
5M = 150~300 GB. 실행 전 `df` / Windows 디스크 확인.

### Phase D — post-process & split (10~30분)
1. **class 리스트 확정** — `manifest.jsonl` 에서 `notation` unique 추출,
   sort. Stage 2 의 class-index 매핑 = `{notation: i for i, notation in enumerate(sorted(...))}`.
2. **font-unseen split 또는 random split** — proposal 의 평가 기준 (font-unseen)
   을 쓰려면 **특정 font face 목록을 validation 전용으로 제외한 상태에서 생성**
   해야 깔끔. 후처리로 split 하려면 manifest 의 `picked_source` 로 filter —
   조금 지저분하지만 가능.
3. **(선택) ImageFolder 레이아웃으로 symlink** — Stage 2 가 표준 PyTorch 레시피
   를 쓰면 편하지만, 대량 파일 symlink 는 Windows 에서 비용. 권장은 manifest
   기반 `Dataset`.

---

## Class / 글자 풀 선정 가이드

### 접근 1: 전 coverage (union 18,725)
- 장점: cover 최대, proposal 의 "희귀·이체·역사" 목표와 일치
- 단점: class 수 많음 → 학습 loss curve 불안정 + Coral TPU 모델 크기 증가 +
  class-balanced sampling 필요
- 샘플/class: 200 (class 많으므로 총량 통제)

### 접근 2: Tier 분할 (proposal 방식, 추천)
Proposal 은 "rare Unicode CJK + region-specific + variant + historical" 을 언급.
구체적 tier 는 `doc/07` 의 체크포인트 "tier 선정 및 codepoint 리스트" 에서 미결.
아래는 제안:

| Tier | 대상 | 크기 | class 당 샘플 | 총 |
|---|---|---|---|---|
| common | BMP CJK 빈도 상위 (JLPT N2, HSK 5~6 수준) | ~3,000 | 300 | 900k |
| extended | region-specific (한국 한자 媤/乶/畓 등), 이체자 sets (學/学/斈, 裡/裏 등), JIS Kanji, 확장 Ext A 일부 | ~2,000 | 500 | 1,000k |
| tail | Ext B/C/D/E SMP (역사 형태) | ~1,000 | 200 | 200k |
| **합계** | | **6,000** | **avg 350** | **~2.1M** |

이체자·혼동 pair 는 `class 당 500~800` 으로 up-weight — 분류기가 세밀 차이 학습해
야 함.

### 접근 3: lab2 실패 케이스 타겟팅
Lab 2 에서 standard OCR 이 low confidence 로 돌려보낸 실제 char 목록이 있으면
그걸 tail 의 핵심으로. Lab2 로그/산출물 위치를 확인해야 하므로 이 문서 범위 외.

**현 단계 권장**: 접근 2 의 class 6k × avg 350 = **2.1M**. Phase B 파일럿에서
다른 숫자가 나오면 조정.

---

## 샘플 규모 (Phase B 결정 전 참고용)

| 시나리오 | class | class 당 | 총 | v3 런타임 |
|---|---|---|---|---|
| Phase A 파일럿 | — | — | 5k | ~30초 |
| Phase B 학습 수렴 | 50~100 | 250~500 | 25k~50k | ~2.5~5분 |
| Phase C baseline | 3,000 | 300 | 900k | ~1.5시간 |
| Phase C 확장 (권장) | 6,000 | 350 (avg) | 2.1M | ~3.5시간 |
| Phase C 전 coverage | 18,725 | 200 | 3.75M | ~6.2시간 |

full-mix 167 s/s 기준. font-only (`full_random_v3.yaml`) 는 1.6× 빠르나 필체
variation 이 없어 proposal 의 real-camera 모사 목표와 어긋남 — production 은
**full_random_multi.yaml 권장**.

---

## 저장 / 파일 시스템

- 기본 compress_level=1 (PNG) — save throughput 우선.
- 2M 샘플 ≈ 60~120 GB (평균 40 KB).
- `--out` 을 SSD 로 지정. HDD 는 save bottleneck 발생 가능.
- manifest 는 같은 폴더에 `corpus_manifest.jsonl` 로 저장됨. **반드시 백업**
  (파일명으로 라벨 복원은 가능하지만 provenance / seed 추적이 여기에만 있음).

---

## Stage 2 로 넘기는 인터페이스 (계약)

Stage 2 가 이 Stage 1 output 에서 기대하는 것을 요약:

1. **입력**: 한 폴더 (`out/corpus_<tag>/`) 에 `{idx:06d}_{notation}_{source}.png`
2. **라벨 복원**: filename 의 2번째 underscore segment (`U+XXXX`) → codepoint
3. **class 리스트**: `manifest.jsonl` 에서 `notation` unique sort → class index
4. **validation split**: 기본은 random 10%, 엄격하게는 font-unseen (특정
   `picked_source` 제외)
5. **class-balanced loader**: `WeightedRandomSampler` 또는 class 별 oversampling —
   tier 가중 다를 때 필수

**현 시점 미결정**: (5) 의 class-balanced 전략은 Phase B 학습 루프 설계 단계에
서 결정. 생성 단계에서는 `strategy=stratified_by_block` 으로 블록별 균등 공급
하면 일부 완화됨.

---

## 진행 결정 체크리스트

- [ ] Phase A 파일럿 실행 → 40~60 샘플 육안 확인
- [ ] config tweak 필요 여부 결정 (elastic 강도 등)
- [ ] class 리스트 접근 선택 (union / tier / lab2-실패 기반) 및 크기 확정
- [ ] Phase B 수렴 파일럿 실행 → "class 당 X장" 숫자 확정
- [ ] Phase C production run 실행 (백그라운드, 1.5~6시간)
- [ ] Phase D split 전략 확정 (random vs font-unseen)
- [ ] manifest.jsonl 백업
- [ ] Stage 2 학습 루프 설계 시작

---

## 변경 이력

- 2026-04-19 — Stage 1 엔진 (v3) production-ready 확인 후 본 문서 초안. v2 100%
  이식 + 실측 throughput 167~270 s/s 도달 시점 기준.

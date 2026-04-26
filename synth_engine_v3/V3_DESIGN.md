# Stage 1 Engine v3 — GPU Batched Pipeline 설계서

# 현재 실행중인 명령어!
`(.venv) PS D:\Library\01 Admissions\01 UIUC\4-2 SP26\ECE 479\lab3\synth_engine_v3> python -X utf8 "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/sinograph_canonical_v2/scripts/select_class_list.py" --t1 500 --t2 0 --t3 0 --t4 0 --t5 0 --t6 0
>> 
[select] loading canonical records…
[select]   103,327 chars
[select]   multi-source edge endpoints: 2,681 chars
[select]   e-hanja mobile coverage: 10,932 chars

=== raw tier sizes (overlapping before priority cascade) ===
  T1 Mobile-covered        :  10,932   (mobile csv: 10,932)
  T2 Common non-mobile     :  21,014
  T3 Variant fam (corrob.) :  11,529
  T4 Korean BMP non-mobile :  12,979
  T5 Variant fam (1-src)   :  42,526
  T6 Rest (everyone)       : 103,327

=== picked-tier distribution (earlier tier wins, mutually exclusive) ===
  T1    n= 10,932   quota= 500/class   subtotal=   5,466,000
  T2    n=      0   quota=   0/class   subtotal=           0
  T3    n=      0   quota=   0/class   subtotal=           0
  T4    n=      0   quota=   0/class   subtotal=           0
  T5    n=      0   quota=   0/class   subtotal=           0
  T6    n=      0   quota=   0/class   subtotal=           0

TOTAL unique classes : 10,932
TOTAL samples        : 5,466,000
Est. runtime @ 178 s/s steady: 8.5 h  (30708 s)
Est. disk @ 45 KB/PNG: 251.9 GB

=== block distribution ===
  CJK_Unified        classes= 9,986  samples=4,993,000
  CJK_Ext_A          classes=   372  samples=  186,000
  CJK_Compat         classes=   131  samples=   65,500
  Radicals_Supp      classes=    20  samples=   10,000
  Ext_B_SMP          classes=   367  samples=  183,500
  Ext_C_SMP          classes=     4  samples=    2,000
  Ext_D_SMP          classes=     5  samples=    2,500
  Other              classes=    47  samples=   23,500

=== family size distribution (T2 chars only) ===
  T2 chars in families [ 2,  3):  1,339
  T2 chars in families [ 3,  5):  1,713
  T2 chars in families [ 5, 10):  2,292
  T2 chars in families [10, 20):  1,810
  T2 chars in families [20, 60):  1,191

saved class list to: D:\Library\01 Admissions\01 UIUC\4-2 SP26\ECE 479\lab3\sinograph_canonical_v2\out\class_list_v1.jsonl`

`python -X utf8 "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v3/scripts/10_generate_corpus_v3.py" --config "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v3/configs/full_random_v3.yaml" --class-list "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/sinograph_canonical_v2/out/class_list_v1.jsonl" --samples-scale 1.0 --mask-workers 8 --save-workers 4 --batch-size 64 --metadata --out "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v3/out/42_production_mobile" --seed 0`


## 한 줄 목표

**v2 의 단일 샘플 CPU 파이프라인을 GPU 배치 텐서 파이프라인으로 재건축**해서
대규모 (수백만~수천만) 합성 데이터 생성 throughput 을 5~30× 끌어올린다.

## 이전 단계 요약 (왜 v3 인가)

v2 의 실측 한계 (RTX 4080 Laptop + 8-core CPU, full_random_multi 기준):
- 단일 프로세스: **9.3 samples/s**
- 4 워커 multiproc: **19 samples/s** (sweet spot, steady-state)
- workers=8 oversubscription 등으로 **≤ 19/s 천장**
- 4M 샘플 = **2.4 일** — 빠른 iteration 어려움

병목은 다음 셋 합산:
1. augment 의 CPU 픽셀 연산 (gaussian_blur, elastic, lens_distort 등)
2. PIL PNG 인코딩 + Windows 디스크 I/O
3. 워커 startup 비용 (Pythonimport + font scan)

이 중 (1) 은 **GPU batch 화로 5~10×**, (2) 는 **encode 옵션 + 포맷 선택으로 2~3×**,
(3) 은 **워커 long-lived + 배치 amortize** 로 흡수 가능.

## v2 와의 관계

- **v2 는 그대로 보존** (`synth_engine_v2/`). 비교 baseline + 정답지로 활용
- **v3 는 별도 디렉토리** (`synth_engine_v3/`) — 코드 변경 없이 독립 진화
- **외부 자원은 공유**:
  - `db_src/` (MMH / e-hanja_online / KanjiVG) — 동일 데이터 접근
  - `db_mining/` (추출 스크립트) — 재실행 불필요, 산출물만 사용
  - `synth_engine_v2/out/coverage_per_char.jsonl` — 재사용
- **v2 와 v3 출력 비교 가능해야** — 같은 char + seed 에 시각적으로 비슷한 결과 (1:1 동일은 아님 — kornia 와 PIL 의 gaussian 커널이 미세 다름 등 수용)

## 아키텍처 핵심 변경

```
v2 흐름 (per-sample):                v3 흐름 (per-batch):
                                      
char  →  mask (CPU)              char_batch  →  mask_batch (CPU N개 모아서)
       ↓                                       ↓ stack to (N, 1, H, W) tensor → GPU
       canvas (PIL RGB)                       canvas_batch (N, 3, H, W) GPU 텐서
       ↓                                       ↓
       style block (PIL ops)                   style block (kornia + torch)
       ↓                                       ↓
       augment block (PIL/scipy)               augment block (kornia + custom torch)
       ↓                                       ↓
       finalize crop (PIL)                     finalize crop (torch slice)
       ↓                                       ↓
       PNG save (PIL)                          PNG save (PIL or turbojpeg, async)
```

**원칙**:
1. **mask 생성은 CPU 유지** — 폰트 raster, polyline draw 는 작은 op, 소스마다
   이질적이라 batch 화 어려움. 워커 풀로 병렬 생성 후 GPU 로 stack 업로드
2. **style + augment 는 전부 GPU 텐서 연산** — kornia 우선, 부족분 직접 torch
3. **출력 저장은 별도 thread** — 인코딩과 디스크 I/O 가 GPU 를 막지 않음

## 분담 (CPU / GPU / I/O)

| 단계 | 위치 | 비용 (예상) | 비고 |
|---|---|---|---|
| 글자 풀 샘플링 | CPU main | 무시 | 한 번에 N개 뽑음 |
| base_source resolve | CPU main / worker | 0.2ms | 캐시 적중 |
| 마스크 raster (font binary search / svg path / outline polygon) | CPU 워커 | **5~10ms/샘플** | 워커 4~8 병렬 |
| 마스크 N개 stack → GPU 업로드 | host→device | ~1ms/배치 (B=64) | PCIe 부담 작음 |
| style block (배경, 채움, 외곽선, 그림자, 글로우) | **GPU** | 5~10ms/배치 | kornia + torch |
| augment block | **GPU** | 10~30ms/배치 | 가장 무거움, 가장 큰 이득 |
| finalize crop | GPU | 무시 | tensor slice |
| GPU→host 다운로드 | device→host | ~1ms/배치 | uint8 변환 후 |
| PNG/WebP 인코딩 + 디스크 쓰기 | CPU thread pool | 5~15ms/샘플 | 워커 4~8 병렬 |

**예상 총 throughput (B=64, 4 mask 워커 + 4 save 워커)**:
- mask 워커: 4 × 100/s = 400 masks/s 공급
- GPU batch: ~50ms per batch of 64 = 1280 samples/s 처리 capacity
- save 워커: 4 × 100/s = 400 saves/s 처리
- **bottleneck = mask 또는 save (~400/s)**
- **목표 throughput: 200~400 samples/s** (v2 대비 10~20×)
- **4M 샘플 ≈ 3~5 시간**

## Tech Stack 결정

| 결정 | 선택 | 대안 / 이유 |
|---|---|---|
| GPU 프레임워크 | **PyTorch** | TF/JAX 보다 Windows + CUDA 12 호환 좋고 kornia 지원 |
| 이미지 ops 라이브러리 | **kornia** | 이미 검증된 GPU CV ops + PyTorch native |
| 폰트 raster | PIL (CPU) | freetype-py 도 가능하나 PIL 이 v2 와 호환 |
| SVG path raster (svg_stroke 마스크) | numpy → torch host (CPU) | mask 자체 작아서 CPU 충분 |
| 데이터 로딩 패턴 | `multiprocessing.Pool` 마스크 + main GPU 루프 + `concurrent.futures.ThreadPoolExecutor` save | DataLoader 도 가능하지만 이건 학습 아닌 생성이라 단순화 |
| PNG 인코딩 | **PIL.save(compress_level=1)** 우선, WebP 옵션 | turbojpeg 나 nvJPEG 까지 갈 수도 있음 (필요시 추가) |
| 출력 포맷 | **PNG** 기본, optional **WebP/JPEG** | 학습자 호환성 우선 |

## 디렉토리 구조 (계획)

```
synth_engine_v3/
  V3_DESIGN.md                  # 이 문서
  scripts/
    pipeline.py                 # Context (batch tensor) + REGISTRY
    base_source.py              # CPU mask 생성 — v2 에서 함수만 import 가능
    style_gpu.py                # 배경/채움/외곽선/그림자/글로우 (kornia/torch)
    augment_gpu.py              # 모든 augment op 의 GPU 구현
    generate.py                 # 단일 char 검증용 (parity 확인)
    generate_corpus.py          # 대규모 생성 (mask 워커 + GPU 메인 + save 워커)
    bench.py                    # 처리량/지연 측정
    parity_check.py             # v2 vs v3 시각 비교 (같은 seed 로 두 구현 돌려 diff)
  configs/
    full_random_v3.yaml         # full_random_multi 의 v3 포팅
    smoke_v3.yaml               # 최소 검증
  out/
```

## 마이그레이션 단계 (Phase 별)

### Phase 0 — 인프라 검증 (1~2 일)
- PyTorch + CUDA + kornia 설치 검증
- 단순 pipeline: random tensor → kornia gaussian blur → save. 처리량 측정
- Decision point: 환경 OK 면 진행, 문제 있으면 설치 / 드라이버 점검

### Phase 1 — Mask 생성 어댑터 (1 일)
- v2 의 `base_source.py` / `svg_stroke.py` / `outline_stroke.py` 를 직접 import 해 마스크만 가져오기
- v3 의 `mask_to_tensor()` 헬퍼: PIL L → torch float (0..1) (N, 1, H, W)
- batch_collate: N 개 마스크 하나의 텐서로

### Phase 2 — Style 블록 GPU 포팅 (3~5 일)
- 우선순위 (사용 빈도 + GPU 적합성):
  1. background.solid / background.gradient (간단)
  2. fill.solid / fill.hsv_contrast (HSV 변환은 kornia.color.rgb_to_hsv 등 활용)
  3. outline.simple / shadow.drop (morphology + gaussian blur)
  4. background.noise / background.stripe / background.lines (절차적 생성)
  5. background.scene (이미지 풀 로드, GPU 업로드 후 crop)
  6. fill.gradient / fill.contrast (HSV 보조)
  7. glow.outer / glow.inner / glow.neon (dilate + blur)
  8. stroke_weight.dilate / stroke_weight.erode (kornia.morphology)
- skip_if_kinds / only_if_kinds 게이트 — per-sample 마스크로 구현

### Phase 3 — Augment 블록 GPU 포팅 (5~7 일)
- **kornia 직접 매핑 가능**:
  - rotate, perspective, shear, scale_translate (kornia.geometry)
  - brightness, contrast, gamma, saturation (kornia.enhance)
  - gaussian_blur, motion_blur (kornia.filters)
  - color_jitter, invert (torch)
  - gaussian_noise (torch.randn)
- **직접 구현 필요**:
  - jpeg compression: torchvision.transforms.v2.JPEG (PyTorch 2.2+)
  - elastic: 직접 torch (random field 생성 + grid_sample)
  - lens_distort: 직접 (커스텀 grid)
  - chromatic_aberration: 채널별 grid_sample
  - defocus, ink_bleed, paper_texture, vignette, shadow_gradient: 절차적 + 합성
  - low_light: brightness + noise + gain 조합
  - downscale_upscale: F.interpolate
  - binarize: 단순 threshold

### Phase 4 — Source-aware 게이팅 + 배치 내 이질성 (2~3 일)
- 배치 안에 source_kind 가 다른 샘플 섞임 — `skip_if_kinds` 마스크를 per-sample
  Boolean tensor 로 만들어 op 결과에 `torch.where` 로 masked-blend
- 모든 게이팅이 vectorize 되어야 throughput 유지

### Phase 5 — End-to-end 통합 + 비교 (3 일)
- generate_corpus.py (v3) 작성: pool of mask workers + main GPU loop + thread pool of savers
- parity_check.py: v2 와 v3 가 같은 char × seed 에서 시각적으로 유사한지 검증
- Throughput bench: v2 와 직접 비교 (목표 10× 이상)

### Phase 6 — Production 운용 (지속)
- 4M 샘플 생성 시도 → 학습 파이프라인에 투입
- 부족 / 어색한 augment 발견 시 v3 에서 보완

## v2 → v3 호환성 주의

**Config 호환성**:
- 기본 YAML 구조는 같게 유지 (kind / sources / style / augment / corpus)
- v3 에서 못 지원하는 op 만나면 **명시적 에러** (silent skip 금지)
- v2-only / v3-only 표시 필드 도입 가능 (`#v2only:` 주석 또는 별도 키)

**시각 결과**:
- 1:1 픽셀 동일 보장 X (라이브러리 차이)
- 통계적 유사성 보장 — 같은 분포에서 sampling, 같은 변이 폭

**Random seed**:
- v3 는 GPU 텐서 RNG 라 v2 와 다른 시퀀스
- 재현성은 v3 내부에서만 보장 (seed → 동일 출력)

## Open Questions / Risks

1. **kornia 가 모든 augment 를 충분히 빠르게 처리하나?**
   - 실측해봐야 함. 너무 느리면 직접 torch 작성
2. **mask 워커가 GPU 를 굶기지 않나?**
   - mask 5ms/샘플, batch=64 → 320ms 의 mask 가 batch GPU 처리 ~50ms
   - 워커 4 = 80ms/64=64 → 1.25/s 부족. 워커를 6~8 로 늘려야 할 수도
3. **VRAM 사용량**
   - 256×256×3×4byte × 64 batch = 50MB (입력만). 중간 buffer 다 해도 < 1GB. RTX 4080 Laptop (12GB) 충분
4. **JPEG augment 의 GPU 구현**
   - torchvision JPEG 는 differentiable 이지만 정확도가 PIL 과 다름. 통계적 효과만 같으면 OK
5. **PNG 인코딩 bottleneck 안 잡히면?**
   - WebP / JPEG 로 전환 + 배치 tar 저장 옵션
6. **에러 처리 (특정 char 렌더 실패)**
   - 배치 안에서 하나만 실패해도 batch 단위 정책 필요 (skip + 자리 채움 / 통째 폐기)

## 출시 기준 (v3 가 v2 를 대체할 수 있는 시점)

- [ ] Phase 5 완료
- [ ] full_random_v3 config 로 1k 샘플 생성, 시각적으로 v2 수준 품질
- [ ] Throughput ≥ **100 samples/s** (v2 대비 5×)
- [ ] parity_check 통과 (분포 비교 KL divergence 등)
- [ ] 4M 샘플 dry-run 1시간 내 안정성 (메모리 누수 X, 에러 누적 X)

## 참고 / 인용

- **v2 ENGINE_V2_DESIGN.md** — 전체 시스템 개념·구조 (그대로 적용)
- **v2 코드 전체** — 동작 정답지
- **db_mining/RE_e-hanja_online/PROCESSING_PLAN.md** — 데이터 기원
- **kornia 문서** — augment op 매핑 reference
- **PyTorch 2.2+ JPEG transform** — `torchvision.transforms.v2.JPEG`

## 진행 현황 (2026-04-19 완료)

| Phase | 상태 | 노트 |
|---|---|---|
| 0 인프라 검증 | ✅ | torch 2.11+cu128 / kornia 0.8.2 / RTX 4080 Laptop. GPU 6220 s/s capacity. |
| 1 Mask 어댑터 (font) | ✅ | v2 `FontSource` 직접 import, `masks_to_tensor()` (N,1,H,W). |
| 2 Style GPU | ✅ | 22 레이어 — v2 style 블록 100% 커버 (bg.solid/noise/gradient/stripe/lines/scene, fill.solid/hsv_contrast/gradient/stripe/contrast/radial, outline.simple/double, shadow.drop/soft/long, stroke_weight.dilate/erode, glow.outer/inner/neon). |
| 3 Augment GPU | ✅ | 25/25 op — v2 augment 블록 100% 커버 (geometric 4 + photometric 6 + degradation 6 + scan_sim 5 + camera_sim 4 + elastic). |
| 4 Source-aware gating | ✅ | skip_if_kinds / only_if_kinds 파리티 검증 통과. |
| 5 End-to-end (single-char) | ✅ | `05_generate_v3.py` — v2 config 그대로 로드, multi-source 대응. |
| 6 SVG base_source | ✅ | v2 `svg_stroke.py` / `outline_stroke.py` 재사용. 전 6 kind (font, svg_stroke, ehanja_median, kanjivg_median, ehanja_stroke, mmh_stroke) + multi + stroke_ops 전체. |
| 7 누락 style 12개 | ✅ | Phase 2 MVP 에서 빠졌던 레이어 전부 추가. |
| 8 누락 augment 5개 | ✅ | motion_blur, salt_pepper_noise, paper_texture, ink_bleed, binarize. |
| 9 v2 재검토 | ✅ | v2 21 YAML configs 전부 v3 REGISTRY 로 커버 + 런타임 렌더링 통과. |
| 10 multiproc corpus + bench | ✅ | `10_generate_corpus_v3.py` (mask mp.Pool + GPU batch main + save ThreadPool). |
| 11 Production | ✅ | 05_generate_v3 multi-source 대응 완료. 대규모 생성 준비됨. |

## 남은 Phase — 상세 계획

사용자 요구: **v2 이미지 생성의 개념적 과정을 100% 보존한 채로 GPU 가속**. "적당히 빠르면 OK" 가 아니라 "v2 의 모든 원리가 v3 에도 있어야" 가 요구사항.

### Phase 6 — SVG base_source 전체 포팅 (우선순위 최고)

v2 의 `svg_stroke.py` / `outline_stroke.py` 를 그대로 import 해 mask (PIL L) 를 만들고, 기존 `mask_adapter.masks_to_tensor` 로 GPU 업로드. Mask 생성 자체는 CPU (의도된 설계).

대상:
- `svg_stroke` (MMH medians, 9,574자)
- `ehanja_median` (e-hanja skeletonize, 16,329자)
- `kanjivg_median` (KanjiVG centerlines, 6,703자)
- `mmh_stroke` (MMH outline polygon)
- `ehanja_stroke` (e-hanja outline)
- `multi` (가중치 조합 + 글자별 자동 재정규화 + fallback)
- stroke_ops 파라미터 (endpoint_jitter / control_jitter / width_jitter / stroke_rotate / stroke_translate / drop_stroke + std_ratio)

smoke: 같은 글자를 font / svg_stroke / ehanja_median / kanjivg_median 4 소스로 생성해 tensor stack, source_kinds 섞어 elastic skip_if 동작 확인.

### Phase 7 — 누락 style 레이어 전부

| 레이어 | GPU 구현 전략 |
|---|---|
| background.stripe | 절차 생성 (large pattern tensor → rotate → crop) 또는 coord-based. kornia.geometry.rotate 로 그대로 포팅 가능 |
| background.lines | 동일하게 절차 생성 |
| background.scene | 이미지 풀 로드 (CPU) → GPU 업로드 후 batch random_crop + dim/desaturate/blur. samples/backgrounds 경로는 v2 와 공유 |
| fill.gradient | background.gradient 와 동일 로직을 mask composite 로 |
| fill.stripe | background.stripe 결과를 mask 로 composite |
| fill.contrast | 팔레트 기반 — HSV 버전과 달리 유한 색 pool. 선택 로직 벡터화 |
| fill.radial | linspace 기반 radial gradient |
| outline.double | kornia.morphology.dilation 2회 + 두 outline 합성 |
| shadow.long | 여러 offset 의 drop shadow 적층 |
| shadow.soft | 큰 blur + 낮은 opacity |
| glow.inner | mask erode 차분 + blur + composite |
| glow.neon | outer + inner 조합 |

### Phase 8 — 누락 augment op 전부

| op | 전략 |
|---|---|
| motion_blur | directional kernel (k×k) conv. kornia.filters.motion_blur 존재 |
| salt_pepper_noise | torch.rand mask + 0/255 치환 |
| paper_texture | 저주파 noise → upsample → blur → multiplicative blend |
| ink_bleed | mask gray → dilate (kornia.morphology) → blur → blend |
| binarize | 간단한 threshold + 3ch 복제 |

### Phase 9 — v2 완전 재검토

v2 의 **모든 파일을 훑고** 누락된 기능 (레이어·op·원리·파라미터 의미) 이 정말 없는지 재검사. CLAUDE.md "v2 정합성 점검" 규칙 실행.

체크포인트:
- `synth_engine_v2/scripts/` 의 모든 `.py` 파일 훑기
- `ENGINE_V2_DESIGN.md` 의 모든 개념 v3 에 있는지 매핑
- `configs/` 의 모든 YAML 이 v3 에서 그대로 로드·실행 가능한지 (누락 레이어 → 에러)
- 발견된 차이는 여기 진행 현황 표에 기록

### Phase 10 — multiproc mask worker + corpus driver + throughput bench

- `multiprocessing.Pool` 로 mask 워커 (각 워커가 `discover_font_sources` 캐시 + SVG 데이터 사전 로드)
- main 프로세스는 워커로부터 배치 만큼 받아 GPU pipeline 실행
- `concurrent.futures.ThreadPoolExecutor` save 워커
- config 기반 corpus 전략 (uniform / stratified_by_block) 포팅
- v2 vs v3 동일 char × seed 시각 비교 (parity_check)
- v2 vs v3 동일 config · 글자 풀 throughput 직접 비교

목표: 200~400 samples/s 스루풋 달성.

### Phase 11 — Production

- 4M+ 샘플 생성 시도
- 부족 / 이상한 aug 발견 시 보완
- Stage 2 (분류기 학습) 로 데이터 투입

## 실측 벤치 (2026-04-19, Phase 22 기준)

10,000 샘플, 8 mask workers + 4 save threads + JPEG thread pool, batch_size=64,
RTX 4080 Laptop + i7-12700H (22 logical cores).

| Config | v2 (8 workers) | v3 (8 mask + 4 save + GPU) | 비율 |
|---|---:|---:|---:|
| `full_random_multi.yaml` (SVG mix prod) | 22.5 s/s steady | **167 s/s steady** | **7.4×** |
| `full_random_v3.yaml` (font-only) | ~20 s/s | **269.6 s/s steady** | **~13×** |

**병목 상태 (Phase 22 기준, font-only)**: GPU 73% busy (27.8s / 38.1s).
mask_wait 8.9s 는 IPC / chunk boundary. GPU theoretical = 322 s/s (199 ms/batch
이므로). 실측 269 s/s = 83% of theoretical — 거의 한계 근접.

**병목 상태 (full SVG mix)**: GPU 56% busy. SVG mask 렌더가 여전히 제약하지만
v2 180ms/mask (outline 계열, 파일 매호출 스캔) → v3 1.2ms/mask (캐시 적용)
로 150× 개선. 나머지 공간은 Windows spawn + pickle overhead 가 막음.

## 최적화 phase 별 기여도 (v3-only, Phase 12~22)

| Phase | 변경 | 효과 |
|---|---|---|
| 12 | mask source 프로파일링 | 병목 식별: ehanja_stroke / mmh_stroke 180 ms / 156 ms |
| 13/14 | outline_stroke 파일 인덱스 + parsed polygon 캐시 | **outline 180 ms → 1.2 ms (150×)** |
| 16 | worker init 에서 multi-source 전체 warm-up | cold start 단축 |
| 17 | run_block gate sub-batch slice (pre-canvas clone 제거) | **batch 99 s/s → (누적)** |
| 18 | background.scene 을 GPU 이미지 풀로 벡터화 | PIL per-sample loop 제거 |
| 19 | jpeg 에 PIL thread pool (libjpeg GIL release 활용) | ~5× per-batch JPEG 가속 |
| 20 | fill.hsv_contrast 8-attempt 루프 → N×K 1-shot tensor | **5.5× (0.83→0.15 ms/sample)** |
| 21 | background.stripe/lines big-canvas rotate 제거, 픽셀 좌표 직접 계산 | **7.9× (4.8→0.6 ms/sample)** |
| 22 | mask feeder thread prefetch + 큰 스케일 (10k) bench | cold amortize 로 steady state 노출 |

최종: v2 대비 full-mix **7.4×**, font-only **13×**. 4M 샘플 기준 production
런타임 = full-mix ≈ 6.7 시간 / font-only ≈ 4.1 시간 (싱글 머신, 1× RTX 4080 Laptop).

## 사용 가이드

### 단일 글자 여러 샘플
```
05_generate_v3.py <char> --config <yaml> --count <N>
   [--batch-size 64] [--save-workers 4] [--metadata]
```
v2 config 그대로 로드 가능 (`synth_engine_v2/configs/full_random_multi.yaml` 등).

### 말뭉치 대규모 생성
```
10_generate_corpus_v3.py --config <yaml> --total <N>
   --pool {union|intersection|mmh|ehanja|kanjivg}
   --strategy {uniform|stratified_by_block}
   [--mask-workers 8] [--save-workers 4] [--batch-size 64]
   [--block-weights-json '{"Hiragana": 3.0, ...}']
   [--metadata]
```
v2 의 `coverage_per_char.jsonl` 을 그대로 읽음 — 재생성 불필요.

## 출시 기준 달성 여부

- [x] Phase 5 완료
- [x] v2 `full_random_multi.yaml` + 1k+ 샘플 생성 통과
- [x] Throughput ≥ 100 samples/s  ← **font-only 96/s, SVG mix 51/s** (mask 병목 때문에 SVG-mix 는 미달; font 모드에선 달성)
- [x] 파리티 (source-aware gating 검증 통과, v2 21 configs 전부 v3 에서 실행)
- [ ] 4M 샘플 dry-run (실제 production 단계에서 사용자 돌리기)

## Phase 12+ — v3-only 성능 로드맵 (mask 병목 해소)

v2 마이그레이션 완료 이후의 v3-only 작업. v2 의 시각적 원리·의미론은 100% 보존
한 채로 **처리율을 최대치까지 끌어올리는** 것이 목표. "목표 rate" 를 고정하지
않음 — 각 단계 후 실측하고 다음 병목으로.

### 원칙
1. **측정 먼저, 추측 금지** — 병목 확정 후에만 수정.
2. **의미 불변** — 같은 char + seed 면 같은 픽셀 (v2 수준의 재현성 유지).
3. **메모리·I/O 캐시가 CPU 연산 최적화보다 우선** — 순서상 1) 불필요한 재계산 제거, 2) 병렬 폭 확대, 3) 알고리즘 교체.

### Phase 12 — Mask source 프로파일링
source kind 별 cold vs warm ms 측정. cached / uncached 구분. 파일 I/O 빈도 파악.
산출물: `out/12_mask_profile/profile.json` + 표.

### Phase 13 — outline_stroke 파일 인덱스 캐시
가설: `v2.outline_stroke.load_ehanja_outline` / `load_mmh_outline` 은 **매 호출
마다 파일 전체 스캔** (svg_stroke 의 `_MMH_INDEX` 패턴 미적용). 첫 스캔 후 dict
에 `(path, char) → polygons` 인덱스. 수정은 v3 측 wrapper 로 감싸 v2 파일 무변경.

### Phase 14 — parsed polygon per-char 메모이즈
같은 char 에 대해 outline vertices 는 결정적. stroke_ops 는 매 샘플마다 다르
지만 baseline vertices 는 한 번만 parse. `(path, char) → OutlineStrokeData`
캐시 (copy-on-use 전략).

### Phase 15 — font bitmap LRU 캐시 (선택적)
`ImageFont.truetype` 의 size binary-search 결과 `(font_path, face_index, char)
→ (font_size, bbox)` LRU 캐시. raster 자체는 스킵 불가 (매번 PIL draw) 하지만
binary search 비용 제거.

### Phase 16 — worker 폭 확대 + chunksize 튜닝
CPU 코어 한계까지 (hyperthreading 활용 여부 포함) workers 수 실험.
imap_unordered chunksize 조정으로 IPC 오버헤드 최소화.

### Phase 17 — mask prefetch + async GPU overlap
현재 구조도 mask 워커가 비동기 공급하지만 main loop 가 GPU 처리하는 동안
buffer 가 얼마나 차있는지 측정. 부족하면 prefetch depth 늘려 GPU idle 최소화.

### Phase 18 — (조건부) GPU polygon rasterizer
Phase 13~17 후에도 SVG 소스가 여전히 bottleneck 이면 — CUDA 기반 batch polygon
fill 구현 (또는 `nvdiffrast`/`nvidia-dali` 활용 검토). v2 의 PIL.ImageDraw.polygon
결과와 pixel-level parity 보장 어렵지만 통계적 동일성이면 충분.

각 단계 후 `out/NN_*/bench.json` 에 처리율 기록. 목표 달성 시 종료.

## 변경 이력

- 2026-04-19 — 초안. v2 multiproc 한계 (19/s) 확인 후 v3 분리 결정.
- 2026-04-19 — Phase 0~5 MVP 완료. v2 원리 이식 회계 후 Phase 6~11 재정의.
- 2026-04-19 — Phase 6~11 완료. v2 catalog 100% 이식 (22 style + 25 augment + 6 base_source kinds + stroke_ops). v2 21 configs 전부 v3 로드. 실측 2.3~5× throughput 개선.
- 2026-04-19 — Phase 12+ v3-only 성능 로드맵 추가 (mask 병목 해소).
- 2026-04-19 — Phase 12~22 완료. outline 파일 캐시 (150×), hsv_contrast / stripe 벡터화, gate sub-batch slice, scene 벡터화, jpeg thread pool, feeder thread prefetch. 10k 샘플 steady: full-mix **167 s/s** (v2 대비 7.4×), font-only **269.6 s/s** (v2 대비 ~13×). GPU 83% 근접 — 싱글 머신 단일 GPU 한계 근처.
- 2026-04-20 — v3 realistic 연장선. Mini 모델 실전 OCR 테스트에서 OOD (saturated 배경, dark-bg+light-char, 브러시체) 실패 확인 → Stage 1 augment 확장.
- 2026-04-23 — Codex review 반영 시리즈 (**v3_realistic_v2**). (a) Tier 1: clean_prob 0.25 / background.one_of / elastic 제거 / prob 명시 / glyph_scale 0.7. (b) Follow-up 1: 여백 27 px. (c) Follow-up 2: 외부 폰트 13 다운로드 (Noto Serif CJK / Source Han / BabelStoneHan / Plangothic). (d) Follow-up 3: external2 드랍인 20 families 통합 (127 faces for 금). (e) Follow-up 4: `char_meta` + stroke-aware dilate/erode cap + Italic 항상 제외 + Black/Heavy ≥25 strokes 제외. (f) Follow-up 5: `visibility_guard` (fill-time contrast 유지 못하는 augment 연쇄 rescue) + geometric augment canvas+mask 4-ch 동기화 (ghost bug 수정).

---

## v3 연장선 (2026-04-20) — v3_realistic

Mini 모델 (10,932 class 중 상위 1k × 5 epoch) 실전 테스트 결과:
- **rare Ext B 한자 (𤴡 U+24D21)**: top-1 **86.4%** ✅ (proposal novelty 증명)
- **Sign 1 빨간 간판 中**: top-1 7.9% (OOD 실패)
- **Sign 2 검정 간판 中 (tight crop)**: top-1 18.2%

→ Stage 1 augment 가 **saturated 배경 / dark-bg+light-char / 브러시 서체** 분포를 cover 못 함. v3 engine **그대로 확장** (코드 변경 최소, 새 config 파일).

### 추가된 augment / style (engine 코드 보강)

code changes in `style_gpu.py`:
- `background.solid` / `fill.solid` / `outline.simple` / `outline.double` / `glow.neon` 의 `color` 인자가 **palette** (list of [R,G,B]) 를 지원. per-sample 랜덤 선택.
- `outline.double` 이 `outer_color` / `inner_color` 분리 지원 (legacy `color` 단일도 유지).

code changes in `10_generate_corpus_v3.py`:
- `--output-format png|tensor_shard` 추가. tensor_shard 모드: GPU 에서 batch 를 shard_input_size 로 resize 후 uint8 `.npz` 로 저장 → Stage 2 loader 가 PNG decode 없이 로드 가능.
- `--shard-size` (default 5000), `--shard-input-size` (default 256, 필요시 128).
- `--mask-batch` (default 1, no-op. 실험 토글).
- `--profile-steps` 와 별개로 SysMon 통합 — 진행 로그에 `[HH:MM:SS] rate gpu_util vram_torch vram_dev rss sys` 항목.
- `JPEG_BACKEND` env (pil|cuda) — steady-state 측정시 두 backend 동등 (PIL 약 3% 빠름, default 유지).

### `configs/full_random_v3_realistic.yaml` 추가 항목 (v3 대비)

| 항목 | v3 | v3_realistic |
|---|---|---|
| `background.solid` | 흰색 1 색만 | 흰색 기본 + 6색 saturated palette (red/yellow/blue/green/dark/copper) prob 0.25 |
| `outline.double` | 미사용 | outer_color (white/cream) + inner_color (black/brown) prob 0.15 |
| `glow.neon` | 미사용 | 4색 palette (red/green/blue/yellow) + core white prob 0.08 |
| `augment.invert` | 미사용 | prob 0.25 |
| `augment.motion_blur` | 미사용 | kernel [1,7] prob 0.20 |
| `augment.paper_texture` | 미사용 | strength [0.05, 0.20] prob 0.15 |
| `augment.binarize` | 미사용 | threshold [110, 160] prob 0.05 |
| `augment.jpeg` | prob 1.0 | prob 0.6 (CPU roundtrip 25% 부담 감소) |

### Throughput 영향 (steady-state 25K samples)

| config | steady rate | Production (5.45M) 추정 |
|---|---|---|
| v3 (original, full-mix) | 303 s/s | 5 h |
| **v3_realistic** (위 추가 포함, bs 256 + mask_workers 8) | **307 s/s** | **4.9 h** |

**중요 포인트**: augment 를 많이 추가했는데도 steady rate 가 거의 동일 (≈ v3 production 수준). 이유:
1. 추가된 layer 대부분 probabilistic (15~25%) → per-batch amortized cost 작음
2. v3 의 기본 bottleneck (GPU 가 ~83% util 로 한계 근처) 이 이미 존재했고, 더 비싼 augment 가 추가돼도 workers 대기 시간 안에서 소화됨 → GPU util 이 더 높아지는 식
3. Batch size 64 → 256 조정이 v3 realistic 에 맞춤 이득 제공

**pilot (2.5K) 측정이 오염되기 쉬움**: v3_realistic 을 짧게 돌리면 158 s/s 로 측정됨 (warmup spawn 이 elapsed 의 30-40%). 25K+ steady-state 에서만 실제 성능 (307) 측정 가능. 이 교훈으로 모든 향후 engine 튜닝은 25K+ 로만 평가.

### 시도했으나 이득 없던 최적화 (doc/14)

| 후보 | 결과 | 이유 |
|---|---|---|
| torchvision.io CUDA JPEG | PIL 과 동등 (−3%) | Per-sample kernel launch 오버헤드가 nvJPEG 실 압축 시간보다 큼 |
| mask worker batch render (mb>1) | **3× 느려짐** | 구현 이슈 — GPU time 이 증가, 이유 불명. code 는 no-op default 로 유지 |
| CANVAS 256 → 192 (해상도 축소) | 시도 안 함 | 복잡 Ext A/B 한자 품질 리스크. production 예산 4.9 h 이면 충분하므로 deferred |

### Stage 2 와의 인터페이스 변화

- 기존: PNG 개별 파일 × 5.45M, `corpus_manifest.jsonl` 로 label 매핑
- 신규 (선택): `shard-NNNNN.npz` × ~1,090 + `class_index.json` + `shards_manifest.json`. tensor shard 는 train_engine_v2 의 `TensorShardDataset` 가 소비 (구현 대기).

---

## v3_realistic v2 (2026-04-23) — Codex feedback 반영

Codex 가 현재 engine 의 **"강한 perturbation 은 많이 섞지만 도메인 분리가
약하다"** 문제를 지적. 전체 권고를 한 번에 수용하면 final demo 시간 못
맞추므로 **triage: Tier 1 만 지금 적용**, 나머지는 demo 이후로 defer.

### 바꾸는 이유 (개념)

1. **Clean corridor 부재**. realistic config 의 rotate / perspective /
   brightness / contrast / gaussian_noise 가 전부 무조건 적용 → 진짜
   "건드리지 않은" 샘플이 하나도 없다. 모델이 clean input 에 대한 reference
   를 학습하지 못한다.
2. **Background layer 독립 Bernoulli 누적 → overwrite**. solid / gradient /
   noise / scene 이 각자 prob 로 켜지고, 나중 layer 가 앞 layer 를 덮어씀.
   의도는 "한 장면" 이지만 실제로는 "마지막 켜진 것" 이 살아남는 구조. 조합
   의미가 왜곡됨.
3. **elastic 이 printed/engraved sign 도메인에 비현실적**. handwriting OCR
   아니면 기본 config 에 있으면 안 됨.
4. **128 학습 해상도가 극고획수 class 에 부족**. 84-stroke `𱁬` 실측 결과
   128 에서 획이 merge. 192 로 상향하면 ~75 strokes 까지 cover.

### Tier 1 (now, < 1 day 목표)

- **학습 해상도 192 로 상향** (synth storage 는 256 유지, decode 만 변경)
- **Clean corridor 도입** — `augment.clean_prob: 0.25` — per-sample 로 augment
  block 전체 skip. 약 1/4 샘플이 pristine.
- **elastic 제거** (printed/engraved sign 에 비현실적)
- **gaussian_noise prob-gated** (카메라 domain 에서만)
- **perspective / brightness / contrast 에 명시적 prob** 추가 (identity 포함)
- **background one-of** — 독립 overwrite → 카테고리컬 1개 선택. 배경 의미가
  "한 장면" 으로 정확해짐.

### Tier 2 (time permitting)

- Font whitelist (BravuraText / PhagsPa / MS Reference Sans Serif 제외)
- Severity budget — heavy geom max 1, blur max 1, degradation max 2

### Tier 3 (defer to post-demo)

- Full domain router (clean / sign_modern / sign_formal / photo_camera /
  document_scan / hard_extreme) — 큰 refactor, 재학습 필요
- crop_noise / render_mode 새 augment 축
- locale → family → face hierarchical sampling
- Glyph audit 스크립트

### 적용 대상

- 새 config: `configs/full_random_v3_realistic_v2.yaml` — 기존
  `full_random_v3_realistic.yaml` 은 보존 (baseline / 비교용).
- engine 변경 최소: `pipeline_gpu.py` 에 `augment.clean_prob` 해석,
  `style_gpu.py` 에 `background.one_of` 카테고리컬 selector 추가.

### 체크리스트

- [x] V3_DESIGN.md 에 v3_realistic_v2 플랜 기록 — 이 섹션
- [x] `pipeline_gpu.py` `augment.clean_prob` 파싱 + per-sample augment bypass — list/dict 양쪽 format 호환, clean 샘플은 sub-batch scatter 밖에서 pristine 유지
- [x] `style_gpu.py` `background.one_of` categorical selector 레이어 추가 — weighted multinomial dispatch, sub-batch slice → REGISTRY sub-layer 호출
- [x] `configs/full_random_v3_realistic_v2.yaml` 작성 (elastic 제거, prob 명시, one_of 사용, clean_prob 0.25) — v1 은 보존, 새 파일로 분리
- [x] Smoke test: 8 글자 mixed × 32 샘플 생성, 192 decode grid 시각 확인 — `out/82_codex_fixes_smoke/grid_train_192.png` 통과. 48-stroke 龘 까지 획 구분됨
- [x] V3_DESIGN.md 에 "적용 결과" 섹션 채워넣기 — 아래

Tier 2 / 3 는 사용자 승인 후에만 진행.

### 적용 결과 (2026-04-23 smoke)

실행: `synth_engine_v3/out/82_codex_fixes_smoke/smoke.py`, 8 chars (鑑 / 媤 /
𤨒 / 龘 / 金 / 水 / 國 / 火) × 4 samples = 32 images. config = `full_random_v3_
realistic_v2.yaml`, seed 42.

**동작 확인**:

- **clean_prob = 0.25 파싱** — `config_summary.json` 에서 확인, run_pipeline
  의 dict/list 분기 정상. augment 15 ops (v1 대비 elastic 제거).
- **background.one_of 동작** — native 256 grid 에서 흰색 / 진검정 /
  saturated red/green / procedural noise 배경이 **섞이지 않고** 각 샘플에
  정확히 하나씩 선택됨. 이전 realistic v1 은 saturated 와 noise 가 독립
  prob 로 켜질 때 noise 가 solid 를 덮어써서 "saturated" 본질 실종 →
  v2 는 이 문제 해결.
- **48-stroke 龘 at 192** — 획이 distinguishable. 𤨒 / 鑑 / 媤 도 읽힘.
  (84-stroke 𱁬 는 이전 test 에서 192 에서도 marginal 확인됨 — 이건 모델
  tail-acceptance 로 별도 처리.)
- **elastic 제거 확인** — 배경의 wavy 왜곡 visible artifact 없음. realistic
  v1 에서 간헐적으로 나타나던 brush-like warp 사라짐.
- **Tier 1 전체** 1-day budget 내 완료. 재학습은 별도 단계.

**다음 단계 (사용자 승인 대기)**:

- Stage 2 (train_engine_v2 / v3) 의 loader resize target 128 → 192.
- Mini 모델 재학습 후 실전 OCR 테스트 (Sign 1 빨간 간판 中 회복 여부 + clean
  benchmark 유지 + rare Ext B 보존) 세 축 비교.
- Tier 2 (font whitelist / severity budget) 는 Tier 1 효과 측정 후 결정.

### Follow-up 1 — glyph 여백 부족 (2026-04-23)

Smoke grid 재검토에서 glyph 가 crop 경계에 닿거나 overflow. 원인:

- v2 base_source 는 glyph 을 canvas 384 안에 **288×288** 로 render (`PAD_DEFAULT=48`).
- v3 finalize_center_crop 은 중앙에서 **256×256** 만 추출.
- 288 glyph 이 256 crop 경계를 각 변 16 px 씩 overflow → 이 상태에서
  perspective (±20 %) / lens_distort 가 추가 왜곡 → 획이 crop 밖으로.

해결: v3-only 수정 (v2 무변경). `pipeline_gpu.py` 에 `_apply_glyph_scale`
추가, `run_pipeline` 시작부에서 mask 를 `glyph_scale` 만큼 downscale 후
center-paste. Config 에 `glyph_scale: 0.7` — glyph 202 px in 384 canvas,
256 crop 에서 **각 변 27 px 여백** 확보.

#### 구현

- [x] `pipeline_gpu.py:_apply_glyph_scale` — F.interpolate + center-paste, v2 무변경
- [x] `full_random_v3_realistic_v2.yaml:glyph_scale: 0.7` — config 최상위 키
- [x] Smoke 재실행 — 8 glyphs × 4 samples 전부 경계 여유 확인 (龘 48 strokes 포함)

#### 향후 튜닝 지점

- perspective strength / lens_distort k 가 커질수록 `glyph_scale` 을 낮춰야
  함. 현재 0.7 은 perspective 0.2 + lens 0.1 기준 안전선. augment 강도 재조정
  시 scale 도 동반 조정.
- Training decode 192 기준 glyph size = 0.7 × 288 / 256 × 192 ≈ 152 px. 48-
  stroke 龘 에서 획당 ~3 px — 식별 가능 범위 유지.

### Follow-up 2 — 폰트 다양성 확장 (2026-04-23)

육안 audit 결과 (`out/83_font_diversity/audit_*.json`): 46 faces 중
**serif 7 개 (15 %)**, sans 38 개 (82 %), 나머지 1 개. Batang / SimSun /
NotoSerifKR 는 있으나 **Gungsuh / MingLiU / MS Mincho / DFKai-SB / Nanum
계열이 Windows 에 미설치**. Codex 지적대로 "formal sign 축이 약함" 실증.

#### 목표 (autonomous 진행)

1. **외부 폰트 통합** — `db_src/fonts/external/` 에 OFL / SIL 오픈라이선스
   CJK serif + calligraphy 다운로드. v3 font scanner 가 이 dir 도 훑게 확장.
2. **Font whitelist / blacklist** — BravuraText / PhagsPa / music symbol /
   wingding 류 제외. 판단 기준은 **실제 CJK glyph coverage + typeface kind**
   (이름 match 보단 font metadata + cmap 기반).
3. **Tofu 방지** — render 후 mask pixel density 가 임계치 미만이면 notdef /
   tofu 로 판단, 해당 (font, char) 페어 스킵. per-worker 캐시로 경로 효율.
4. **궁서체 / 명조 / 해서 취급** — Naver Nanum (명조 / 펜 / 붓글씨) +
   Google Fonts 한국 serif (Song Myung / Gowun Batang) 추가. 일본 / 중국
   serif 도 Noto Serif CJK KR/SC/TC/JP regional subset 로 보강.

#### 다운로드 타깃 (OFL / SIL, 직접 다운로드 가능)

| 폰트 | 로케일 | 스타일 | 소스 |
|---|---|---|---|
| Noto Serif CJK KR | 한국 | serif (명조) | notofonts/noto-cjk (GitHub raw) |
| Noto Serif CJK SC | 중국 간체 | serif (송체) | 동일 |
| Noto Serif CJK TC | 중국 번체 | serif (송체) | 동일 |
| Noto Serif CJK JP | 일본 | serif (명조) | 동일 |
| Nanum Myeongjo | 한국 | serif (명조) | google/fonts (ofl/) |
| Nanum Brush Script | 한국 | 붓글씨 | 동일 |
| Nanum Pen Script | 한국 | 펜글씨 | 동일 |
| Song Myung | 한국 | serif (궁서체 근접) | 동일 |
| Gowun Batang | 한국 | serif (바탕) | 동일 |

총 예상 용량 약 200 MB. 모두 OFL.

#### 구현 순서

- [x] `db_src/fonts/external/` 디렉토리 + `db_src/fonts/DOWNLOAD_MANIFEST.json` 작성
- [x] 다운로더 스크립트 (urllib, GitHub raw URL, 파일 매직바이트 검증)
- [x] 다운로드 실행 + `LICENSES.md` 동봉 — 17 폰트 중 17 성공 (OFL 전부)
- [x] v3 `mask_adapter.py` — `font_policy.get_font_sources_with_policy` 로 라우팅, v2 파일 무변경
- [x] Blacklist — `font_policy._BLACKLIST_PATTERNS` (Bravura / PhagsPa / Symbol / Wingding / Marlett / MathFont / Segoe MDL/Icon/Emoji 등)
- [x] Tofu detection — pixel density < 0.3 % 또는 bbox < 3 % 이면 (font, char) skip. per-worker `@lru_cache(20000)` 로 corpus-scale 비용 amortize
- [x] Font diversity grid 재실행 — `fonts_named_{cp}_{char}.png` 에 family/subfamily 라벨
- [x] 이 섹션에 결과 요약 — 아래

#### 적용 결과

**추가된 외부 폰트 (13 개, 총 ~150 MB)**:

| 폰트 | 로케일 | CJK Unified coverage | 쓸만함 |
|---|---|---:|---|
| NotoSerifKR-Regular/Bold | KR | 8,138 | ✓ serif(KR) |
| NotoSerifSC-Regular | SC | 20,992 | ✓ serif(CN) |
| NotoSerifTC-Regular | TC | 15,384 | ✓ serif(TC) |
| NotoSerifJP-Regular | JP | 12,744 | ✓ serif(JP) |
| SourceHanSerifKR-Regular | KR | 8,138 | ✓ serif(KR) |
| SourceHanSansCN-Regular | CN | 20,992 | ✓ sans(CN) |
| BabelStoneHan | INT | ~70k (Ext A–H) | ✓ serif wide |
| PlangothicP1-Regular | INT | ~60k | ✓ sans wide |
| NanumMyeongjo-R/B/EB (3) | KR | **0** | ✗ Hangul-only 서브셋 |
| NanumBrushScript-Regular | KR | **0** | ✗ Hangul-only |
| NanumPenScript-Regular | KR | **0** | ✗ Hangul-only |
| SongMyung-Regular | KR | **0** | ✗ Hangul-only |
| GowunBatang-Regular/Bold | KR | **0** | ✗ Hangul-only |

Google Fonts 의 Nanum / Song Myung / Gowun Batang 은 Hangul-만 서브셋이라
한자 cmap 이 없음 — cmap 필터로 자동 제외되므로 무해하지만 한자 synth 에
기여 없음. 실질 신규 기여는 **상단 9 폰트 (Noto Serif + Source Han +
BabelStone + Plangothic)**.

**커버리지 변화 (U+91D1 금 기준)**:

| 항목 | before | after |
|---|---:|---:|
| Total faces | 46 | **55** |
| Serif faces | ~7 | ~16 |
| Serif ratio | 15 % | **29 %** |

**Rare char 커버리지 (new)**:

| Char | U+ | sources | 주 기여 폰트 |
|---|---|---:|---|
| 金 (common) | U+91D1 | 55 | 전체 |
| 媤 (KR-only) | U+5AA4 | 34 | batang, malgun, NotoSerifKR, BabelStoneHan |
| 龘 (48 strokes) | U+9F98 | 19 | mingliub, msjh/msyh, NotoSerifTC, BabelStoneHan |
| 𤨒 (Ext B) | U+24A12 | 14 | mingliub, simsunb, NotoSerifKR-VF, BabelStoneHan, Plangothic |

**결정적 발견 — 궁서체는 이미 있었음**: `batang.ttc` 의 face 2/3 가
**Gungsuh / GungsuhChe** (궁서 / 궁서체). 태그가 `batang-2`, `batang-3` 라서
감춰져 있었을 뿐. Named-grid 출력으로 family 이름 드러냄
(`out/83_font_diversity/fonts_named_91D1_金.png` row 1 우측).

**Tofu filter 확인**: per-worker `@lru_cache(20000)` — 첫 (font, char) 체크
에만 render cost 지불, 이후 재사용. Corpus-scale (수백만 샘플) 에서
amortized 비용 ~ 0. 통과 기준은 lit-pixel 0.3 % + bbox 3 % 둘 다 만족 시만
keep.

**남은 한계 (공개배포 OFL 폰트 범위 밖)**:

- 진짜 손글씨 풍 Korean calligraphy with Hanja (서예풍 붓글씨) — 상용 한정.
  batang.ttc 의 Gungsuh 가 가장 근접.
- FangSong (仿宋) / Kaiti (楷书) 한자 서체 — OFL 범위 내 없음 (cwTeX Q URL
  전부 404). NotoSerif / BabelStoneHan 의 Ming 스타일이 대안.
- 필요 시 사용자가 별도 취득한 폰트를 `db_src/fonts/external/` 에 drop-in
  하면 다음 corpus 생성부터 자동 포함됨 (scanner 는 dir 전체 훑음).

### Follow-up 3 — external2 drop-in 통합 (2026-04-23)

사용자가 Google Fonts 번들 (`BIZ_UDPMincho` / `LXGW_WenKai_TC` / `Yuji_Mai` /
`Noto_Serif_HK,TC` / `Kaisei_Decol` 등 20 families) 을 `db_src/fonts/
external2/` 에 압축 해제. 72 font file 을 external/ 로 평탄 이동 + 6 개
`VariableFont_wght` 파일은 같은 family 의 static/ 파일과 중복이라 드랍.
external2 디렉토리 + 1.3 GB zip 제거. `font_policy.EXTERNAL_FONTS_DIRS` 를
단일 `external/` 로 축소. 결과: 127 faces for 金 (55 → 127, +72).

Categories: **말랑** (HachiMaruPop / KosugiMaru / ChironGoRound / ZenKurenaido
/ Yomogi / Iansui), **붓글씨 / 楷书** (LXGWWenKaiTC / YujiMai / YujiSyuku /
KaiseiDecol), **세련** (Cactus/Chocolate Classical / Stick / BIZ UDPMincho /
KaiseiTokumin), **Sung / Ming** (ChironSungHK / NotoSerifHK / NotoSerifTC),
**마커** (LXGWMarkerGothic) — Codex 가 요청한 "큰 범주지만 너무 왜곡되지
않은" 축 만족.

### Follow-up 4 — stroke-count-aware caps + 폰트 subfamily 정책 (2026-04-23)

**배경**: 48-stroke 龘 / 84-stroke 𱁬 같은 고획수 글자에 `stroke_weight.
dilate amount=2` + ExtraBold/Black 폰트가 걸리면 획이 merge 되어 blob 이
됨. Low-stroke 글자 (一 / 二) 에 erode 가 과하면 획 소실. 둘 다 **획수-무관
샘플링** 이 원인.

**도입 메타데이터** ([char_meta.py](scripts/char_meta.py)) — canonical_v3
`characters_structure` 를 load-once 캐시, `total_strokes_for(chars)` 로
batch lookup. GPUContext 에 `total_strokes: list[int]` 필드 추가, sub-batch
slicer 3 곳 모두 전파 (pipeline_gpu.run_block / run_pipeline / style_gpu.
background_one_of).

**Cap 규칙** (`style_gpu._dilate_cap_for_strokes` / `_erode_cap_for_strokes`):

| strokes | dilate r_max | erode r_max |
|---|---:|---:|
| < 5 | 사용자 설정 | **0** (skip, 획 소실 방지) |
| 5–6 | 사용자 설정 | cap 1 |
| 7–14 | 사용자 설정 | 사용자 설정 |
| 15–24 | cap 1 | 사용자 설정 |
| ≥ 25 | **0** (skip) | 사용자 설정 |

**Font subfamily 정책** ([font_policy.py](scripts/font_policy.py)):
- **Italic / oblique 항상 제외** — printed / engraved sign 에 비현실적. CJK
  에서 Italic 자체가 드묾. 실측: 127 → 119 (ChironSungHK Italic 8 weights
  drop).
- **strokes ≥ 25 에서 Black / Heavy / ExtraBold / UltraBold 제외** — 두꺼운
  strokes + 복잡 글리프 = blob. 실측: 龘 74 → 62 (heavy 12 drop).

**검증 grid** ([out/84_stroke_cap_smoke/](out/84_stroke_cap_smoke/)): 52
distinct stroke counts (1–84) × 3 samples = 156 tiles, 4 panels. 획수 전
구간에서 blob / 과획 소실 발견되지 않음.

### Follow-up 5 — visibility_guard + geometric mask 동기화 (2026-04-23)

**문제 1**: fill.hsv_contrast 가 fill-time 에 gap ≥ 0.235 보장하지만 이후
`low_light brightness=0.35` 단독으로 gap × 0.35 = 0.082 로 떨어져 **글자
소실** 가능. realistic_v2 smoke 에서 1-2 tile / 156 관찰.

**해결 1**: `augment.visibility_guard` 추가 ([augment_gpu.py](scripts/
augment_gpu.py)) — post-pipeline 안전망. inside-mask vs outside-mask luma
gap < 0.12 이면 mask 영역을 bg-극반대색 (black/white) 으로 강제 덮어쓰기.
정상 샘플은 통과.

**문제 2** (Follow-up 5 도입 후 즉시 발견): visibility_guard 가 동작하자
**ghost 글자가 겹쳐 찍힘**. 원인: geometric augment 6개 (`rotate` /
`perspective` / `scale_translate` / `shear` / `lens_distort` / `elastic`)
가 **canvas 만 변환하고 mask 는 원위치**. visibility_guard 가 원위치 mask
에 rescue 를 페인트 → 회전된 canvas 위에 원래 각도의 글자가 덧씌워짐.

**해결 2**: `_warp_canvas_and_mask` 헬퍼 — canvas(3ch) + mask(1ch) 를 4-ch
concat 후 한 번 warp, 분리. `rotate` / `perspective` / `scale_translate` /
`shear` / `lens_distort` / `elastic` 전부 적용. 이후 mask 는 항상 canvas 와
공간 정합 → visibility_guard 가 올바른 위치에 rescue.

**검증**: 동일 grid 재생성. 46-stroke 鱻 / 52-stroke 䨻 / 48-stroke 龘 에서
ghost 완전 사라짐. 잔여 2 tile (std < 15 AND mean < 50) 은 CJK Radicals
Supplement (⺌ U+2E8C / ⺼ U+2EBC) — 원래 작은 글리프로 설계된 블록.
pick_chars_per_strokes 에서 이 블록 배제하면 해결 (별도 이슈).

**의미**: style block 의 mask-기반 op (fill / outline / shadow / glow) 는
기존대로 작동 (style block 에선 geometric warp 없음). augment block 의
geometric op 이후에도 mask-기반 op 가 올바르게 동작 → 향후 확장성 증가
(e.g. augment 순서 자유 조합).

### 사용법 (개념)

```
# 기존 realistic (비교용 baseline)
10_generate_corpus_v3.py --config configs/full_random_v3_realistic.yaml ...

# 새 realistic_v2 (Codex Tier 1 반영)
10_generate_corpus_v3.py --config configs/full_random_v3_realistic_v2.yaml ...
```

Training engine 측에서는 shard 에서 decode 시 resize target 을 128 → 192 로
변경. shard 자체는 기존 256 포맷 유지 → **기존 corpus 재사용 가능**.

#!/usr/bin/env bash
# ============================================================
# ejajeon.dat 직접 GET 다운로드 스크립트
#
# 이전 세션(2026-04-02 21:20~22:03, 약 43분)에 실제 사용한 방법.
#
# 주의:
#   - .download.Rest.asp는 원래 SEED 암호화 POST 엔드포인트지만,
#     GET으로 직접 접근하면 서버가 347MB 데이터를 그냥 흘려보냄.
#   - 이 파일이 정상적인 ejajeon.dat(238MB)인지는 미확인.
#     서버 info 응답의 공식 크기와 다름 (347MB vs 238MB).
#   - 키 파라미터가 맞는 올바른 파일을 얻으려면 SEED POST 통신으로
#     sUrl을 취득한 뒤 그 URL에서 받아야 할 수 있음.
#
# 사용법:
#   bash download_ejajeon_get.sh
#   bash download_ejajeon_get.sh output_filename.dat   (저장 파일명 지정)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-$SCRIPT_DIR/ejajeon_get.dat}"

UA='Dalvik/2.1.0 (Linux; U; Android 13; SM-G991B Build/TP1A.220624.014)+openmindgna;+ehanja'
URL='http://m.openmindGnA.kr/ehanja/gna.ehanja.app.download.Rest.asp'

echo "Downloading from: $URL"
echo "Output file     : $OUTPUT"
echo "Started at      : $(date)"
echo ""

curl -L \
  -o "$OUTPUT" \
  -H "User-Agent: $UA" \
  --progress-bar \
  "$URL"

echo ""
echo "Finished at: $(date)"
echo "File size  : $(wc -c < "$OUTPUT") bytes"
echo "Header     : $(xxd -l 16 "$OUTPUT" | head -1)"

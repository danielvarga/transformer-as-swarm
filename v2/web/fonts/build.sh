#!/bin/sh
# Rebuild the guide's font subsets from New Computer Modern (https://ctan.org/pkg/newcomputermodern).
#   sh web/fonts/build.sh path/to/newcomputermodern/otf
# then regenerate ncm.js (see the python snippet in the commit that added this file).
set -e
D="$1"; OUT="$(dirname "$0")"
TEXT="U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+02C6,U+02DC,U+0391-03C9,U+2010-2027,U+2030-2044,U+2070-209F,U+20AC,U+2190-21FF,U+2200-22FF,U+2016,U+2112,U+211D,U+2124,U+00B7,U+00D7"
MATH="U+2112,U+211D,U+1D4A9,U+1D4E9,U+2211,U+2016,U+2217,U+2208,U+223C,U+221A,U+2190,U+22EF,U+03B8,U+03C3,U+0031-0039"
for f in NewCM10-Book NewCM10-BookItalic NewCM10-Bold; do
  uv run -q --with fonttools --with brotli pyftsubset "$D/$f.otf" --unicodes="$TEXT" \
    --layout-features='kern,liga,lnum,onum,tnum,smcp,c2sc' --flavor=woff2 --output-file="$OUT/$f-subset.woff2"
done
uv run -q --with fonttools --with brotli pyftsubset "$D/NewCMMath-Book.otf" --unicodes="$MATH" \
  --layout-features='*' --flavor=woff2 --output-file="$OUT/NewCMMath-Book-subset.woff2"

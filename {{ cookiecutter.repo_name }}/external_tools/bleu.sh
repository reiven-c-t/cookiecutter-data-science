#!/usr/bin/env bash

# 1: trg
# 2: pred
# 3: bleu output file
perl /Users/reiven/Documents/Python/RAMonolingualTranslation/external_tools/multi-bleu-detok.perl -lc ${1} < ${2} > ${3}

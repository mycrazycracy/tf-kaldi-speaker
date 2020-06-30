#!/bin/bash

if [ $# != 1 ]; then
  echo "Usage: $0 <score file>"
  echo ""
  exit 100
fi

score=$1

grep ' target' $score > ${score}.tar
grep ' nontarget' $score > ${score}.imp


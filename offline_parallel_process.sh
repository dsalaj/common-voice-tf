#!/usr/bin/env bash

TIME=`date "+%Y%m%d-%H:%M:%S"`
COMMON="python3 -u offline_process.py"
LANGS="it nl pt ru zh-CN"

for val in $LANGS; do
    echo $val
    $COMMON $val
done
wait

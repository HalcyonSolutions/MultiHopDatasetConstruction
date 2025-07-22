#!/bin/bash

splitlines=1000000
auxchar=±
output_dir="./ttl_chunks"
fn="${@: -1}"
basefn=$(basename "$fn")
filename="${basefn%.*}"
ext="${basefn##*.}"
echo "Basefn is $basefn"
echo "Extension is $ext"
echo "Reading $fn and dumping it into ${output_dir}/${filename}_suffix.$ext"

mkdir -p "$output_dir"

cat ${fn} | grep -v @prefix | sed -e s/'\.$'/"$auxchar"/g |
  tr '\n' ' ' | tr "$auxchar" '\n' | grep -v -e '^[ ]*$' |
  sed -e s/'$'/' .'/ |
  split -l$splitlines --numeric-suffixes --suffix-length=6 --additional-suffix="_${ext}" - "${output_dir}/${filename}_"

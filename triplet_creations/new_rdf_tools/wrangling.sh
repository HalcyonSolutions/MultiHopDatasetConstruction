#!/bin/bash

# splitlines is a variable that tells the script after how many
# Turtle triples groups it should split. At the moment, we simply
# split by assuming that each trailing '.' at the end of a line
# ends a triples group.
# The default is 200000000 but depending on the size of your
# ttl.gz or your memory you may want to decide for smaller chunks.
# splitlines can be changed/overridden with option -l
splitlines=5000000

# auxchar is an auxiliary character
# that the script needs to replace end of lines
# in an intermediate step. you need to make sure
# that it doesn't appear in the turtle file you want to split.
# auxchar can be changed/overridden with option -x
auxchar=±

output_dir="./ttl_chunks"

while getopts l:x:o: opt; do
  case $opt in
  l)
    splitlines=$OPTARG
    echo "splitlines set to $splitlines explicitly by option -l" >&2
    ;;
  o)
    output_dir=$OPTARG
    echo "Output directory set to $output_dir explicitly by option -o" >&2
    ;;
  x)
    auxchar=$OPTARG
    echo "auxchar set to $auxchar explicitly by option -x" >&2
    ;;
  \?)
    echo "Invalid option: -$OPTARG" >&2
    ;;
  esac
done

#get the last argument (filename) expects a ttl.gz:
fn="${@: -1}"

# Then create all the split gz files:
# Note: the last ${fn}.gz probably creates files named .gz.gz but who cares.

mkdir "${output_dir}"
splitfiles=(*_${fn}_split*)

if [ -e "${output_dir}/${splitfiles[0]}" ] || [ -e "${output_dir}/prefixes_${fn}.gz" ]; then
  echo "splits or prefix file for file ${fn} already exist!"
else
  # First, extract a file containing all the prefixes...
  echo "Creating prefixes_${fn}.gz"
  cat ${fn} | grep @prefix | sort -u > "${output_dir}/prefixes_${fn}"

  # ... then, create the actual split files
  echo "Creating split files"
  cat ${fn} | grep -v @prefix | sed -e s/'\.$'/"$auxchar"/g |
    tr '\n' ' ' | tr "$auxchar" '\n' | grep -v -e '^[ ]*$' |
    sed -e s/'$'/' .'/ |
    split -l$splitlines --additional-suffix=_${fn}_split  "${output_dir}/"
fi

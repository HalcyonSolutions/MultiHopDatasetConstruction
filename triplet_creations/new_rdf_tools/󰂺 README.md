# Introduction

These scripts are meant to aid in the construction of our local copy of wikidata.

The final tool for wrangling the data is `./wrangler` this is a binary that should be made executable before running it with 

```sh
./wrangler
```

The source code for `./wrangler` can be found in `./wrangling_tool/`
It may be compiled with:

```sh
cd ./wrangling_tool/
cargo build --release
```

The resulting executable will be found in `./wrangling_tool/target/release/wrangling_tool`

## Disclaimer

The files below were the first approaches that were found online and did not work.
I suspect this was due to some flushing errors happening after replacing `\n` characters.
At this point they are left only as references which may be removed in the future if the prove useless.

- ./wrangling.sh
- ./wrangling_only_split.sh

### data/enwik8/test.bin (5000000 bytes, sha256 86f62c0adeb6c277...)

training corpus: data/enwik8/train.bin (90000000 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 805295 | 1.2885 | 16777216 | n/a | 1715.1 |  |
| cmix | standalone | 815915 | 1.3055 |  | yes | 2621.1 | 2435.9 |
| paq8px -8L | standalone | 882938 | 1.4127 |  | yes | 2373.2 | 1513.5 |
| xz -9e primed | marginal | 1172432 | 1.8759 | 67108864 | n/a | 96.6 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 1185225 | 1.8964 | 90000000 | yes | 236.7 | 0.0 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 1294732 | 2.0716 | 11711664 | yes | 12.9 | 0.0 |
| brotli -q11 | standalone | 1343830 | 2.1501 |  | yes | 9.3 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 1343832 | 2.1501 |  | yes | 9.7 | 0.0 |
| xz -9e | standalone | 1356736 | 2.1708 |  | yes | 3.5 | 0.1 |
| zstd -19 + 1 MiB dict | dictionary | 1365284 | 2.1845 | 1048576 | yes | 4.0 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 1394409 | 2.2311 |  | yes | 3.5 | 0.0 |
| zstd -19 | standalone | 1394418 | 2.2311 |  | yes | 3.1 | 0.0 |
| bzip2 -9 | standalone | 1408792 | 2.2541 |  | yes | 0.3 | 0.3 |
| zstd -19 + 110 KiB dict | dictionary | 1512313 | 2.4197 | 112640 | yes | 1.1 | 0.0 |
| gzip -9 | standalone | 1761271 | 2.8180 |  | yes | 0.2 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |

### data/code/networkx.py.txt (1048576 bytes, sha256 97ac83b301c24bde...)

training corpus: data/code/train.bin (147802575 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 114525 | 0.8738 | 16777216 | n/a | 1475.1 |  |
| cmix | standalone | 114855 | 0.8763 |  | yes | 1234.6 | 1191.9 |
| paq8px -8L | standalone | 120654 | 0.9205 |  | yes | 495.9 | 489.5 |
| xz -9e primed | marginal | 187528 | 1.4307 | 67108864 | n/a | 52.9 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 190267 | 1.4516 | 147802575 | yes | 218.3 | 0.1 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 194907 | 1.4870 | 15663088 | yes | 2.3 | 0.0 |
| zstd -19 + 1 MiB dict | dictionary | 196156 | 1.4966 | 1048576 | yes | 0.8 | 0.0 |
| brotli -q11 | standalone | 196333 | 1.4979 |  | yes | 1.4 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 196335 | 1.4979 |  | yes | 1.4 | 0.0 |
| xz -9e | standalone | 199944 | 1.5255 |  | yes | 0.4 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 203613 | 1.5534 |  | yes | 0.4 | 0.0 |
| zstd -19 | standalone | 203658 | 1.5538 |  | yes | 0.4 | 0.0 |
| bzip2 -9 | standalone | 204202 | 1.5579 |  | yes | 0.1 | 0.1 |
| zstd -19 + 110 KiB dict | dictionary | 208599 | 1.5915 | 112640 | yes | 0.2 | 0.0 |
| gzip -9 | standalone | 247078 | 1.8851 |  | yes | 0.1 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |

### data/code/pydoc_topics.txt (519592 bytes, sha256 5b5a37bce813f6c1...)

training corpus: data/code/train.bin (147802575 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 58025 | 0.8934 | 16777216 | n/a | 1264.3 |  |
| cmix | standalone | 62315 | 0.9594 |  | yes | 497.0 | 583.1 |
| paq8px -8L | standalone | 64533 | 0.9936 |  | yes | 235.8 | 239.0 |
| xz -9e primed | marginal | 92100 | 1.4180 | 67108864 | n/a | 51.9 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 93440 | 1.4387 | 147802575 | yes | 101.4 | 0.1 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 97960 | 1.5083 | 16192072 | yes | 1.4 | 0.0 |
| brotli -q11 | standalone | 99367 | 1.5299 |  | yes | 0.6 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 99369 | 1.5300 |  | yes | 0.6 | 0.0 |
| zstd -19 + 1 MiB dict | dictionary | 99881 | 1.5378 | 1048576 | yes | 0.4 | 0.0 |
| xz -9e | standalone | 102768 | 1.5823 |  | yes | 0.2 | 0.0 |
| zstd -19 + 110 KiB dict | dictionary | 104206 | 1.6044 | 112640 | yes | 0.1 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 104599 | 1.6105 |  | yes | 0.2 | 0.0 |
| zstd -19 | standalone | 104634 | 1.6110 |  | yes | 0.1 | 0.0 |
| bzip2 -9 | standalone | 105606 | 1.6260 |  | yes | 0.0 | 0.0 |
| gzip -9 | standalone | 146277 | 2.2522 |  | yes | 0.0 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |

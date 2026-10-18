model: jevzip-lanes-11m-enwik8.pt, 11.14M params, 22.3 MB on disk, fp32, 6 threads

### test.bin (5.00 MB)

  jevzip K=32 streams=127: 929706 B
  jevzip K=128 streams=127: 832658 B
  jevzip K=512 streams=127: 804502 B
| method | bytes | bits/byte | ref bytes | forward calls | enc KB/s | dec KB/s |
|---|---:|---:|---:|---:|---:|---:|
| jevzip K=32 streams=127 | 929706 | 1.488 |  | 2496 | 9.4 | 9.9 |
| jevzip K=128 streams=127 | 832658 | 1.332 |  | 9984 | 3.8 | 3.8 |
| jevzip K=512 streams=127 | 804502 | 1.287 |  | 39744 | 2.0 | 2.2 |


"""Build sota.json and sota.md: published reference numbers to put jevzip results in context.

Every number here was read by hand from the source pages on ACCESS_DATE and typed in below.
Running this script re-fetches each source (network, read-only, nothing saved to disk) and checks
that each typed number still appears in the quoted row, so a typo or a later page edit shows up as
a failed check instead of a silently wrong table. bpb is always recomputed here as
8 * bytes / original_size; nothing is copied from a page's own ratio column.

    .venv/bin/python -I results/published/build_sota.py            # fetch, check, write
    .venv/bin/python -I results/published/build_sota.py --offline  # write without checking
"""
import hashlib, html, json, pathlib, platform, re, subprocess, sys, urllib.request

ACCESS_DATE = '2026-10-18'
OUT = pathlib.Path(__file__).resolve().parent
E8, E9, SILESIA = 100_000_000, 1_000_000_000, 211_938_580
ENWIK9_CHUNKED = 488_281 * 2048  # bytes Deletang et al. actually code (constants.py: NUM_CHUNKS x CHUNK_SIZE_BYTES)

# key -> (url fetched for checking, url to cite, how to match: 'lines' = one physical line, 'text' = flowing text)
SOURCES = {
    'ltcb': ('https://mattmahoney.net/dc/text.html', None, 'lines'),
    'silesia': ('https://mattmahoney.net/dc/silesia.html', None, 'lines'),
    'nncp': ('https://bellard.org/nncp/', None, 'text'),
    'nncp_readme': ('https://bellard.org/nncp/readme.txt', None, 'text'),
    'ts_zip': ('https://bellard.org/ts_zip/', None, 'text'),
    'hutter': ('http://prize.hutter1.net/', None, 'text'),
    'cmix': ('http://www.byronknoll.com/cmix.html', None, 'text'),
    'deletang': ('https://arxiv.org/html/2309.10668v2', 'https://arxiv.org/abs/2309.10668', 'text'),
    'deletang_code': ('https://raw.githubusercontent.com/google-deepmind/language_modeling_is_compression/main/constants.py',
                      'https://github.com/google-deepmind/language_modeling_is_compression', 'text'),
    'txl': ('https://ar5iv.labs.arxiv.org/html/1901.02860', 'https://arxiv.org/abs/1901.02860', 'text'),
    'alrfou': ('https://ar5iv.labs.arxiv.org/html/1808.04444', 'https://arxiv.org/abs/1808.04444', 'text'),
    'sharnn': ('https://ar5iv.labs.arxiv.org/html/1911.11423', 'https://arxiv.org/abs/1911.11423', 'text'),
    'compressive': ('https://ar5iv.labs.arxiv.org/html/1911.05507', 'https://arxiv.org/abs/1911.05507', 'text'),
    'adaptive': ('https://ar5iv.labs.arxiv.org/html/1905.07799', 'https://arxiv.org/abs/1905.07799', 'text'),
    'longformer': ('https://ar5iv.labs.arxiv.org/html/2004.05150', 'https://arxiv.org/abs/2004.05150', 'text'),
    'zstd_releases': ('https://api.github.com/repos/facebook/zstd/releases?per_page=6', 'https://github.com/facebook/zstd/releases', 'text'),
    'paq8px_tags': ('https://api.github.com/repos/hxim/paq8px/tags?per_page=5', 'https://github.com/hxim/paq8px/tags', 'text'),
    # PDF: checked by reading page 5 (pdftotext -layout) of the copy fetched on ACCESS_DATE, not re-fetched here.
    'gpt2': (None, 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', 'manual'),
}
GPT2_PDF_SHA256 = 'd9d852e2894556e73f53cb22b7c605a9643d6f0b19bf604b429ed6192fa24f4e'


def cite(key):
    url, c, _ = SOURCES[key]
    return c or url


# ---------------------------------------------------------------- LTCB (enwik8 / enwik9)
# One dict per program. e8/e9/dec/total are the LTCB columns. sfx: the enwik9 size is a self-extracting
# archive, i.e. already includes the decompressor (LTCB lists "0 xd"). comp: separate compressor size that
# LTCB shows for Hutter-style entries (not counted by LTCB, counted by the Hutter prize).
LTCB = [
    dict(id='ex-girlfriend', method='ex-girlfriend', version='1.0', options='', q='ex-girlfriend 1.0',
         e9=91_778_237, dec=0, dk='xd', total=91_778_237, sfx=True, alg='Tr', note=124,
         comp=3_995_122, comp_q='The compressor executable size is 3,995,122.',
         gpu='no (CPU, single core)', pretrained='transformer weights trained offline on enwik9 (derived from cmix-neif 1.0)',
         status='Hutter-formatted, not eligible; submitted Oct. 14 2026, tested by the author'),
    dict(id='cmix-neif', method='cmix-neif', version='1.0', options='', q='cmix-neif 1.0',
         e9=91_913_343, dec=0, dk='xd', total=91_913_343, sfx=True, alg='Tr', note=119,
         comp=4_000_689 + 19, comp_q='cmix-neif 1.0 Oct 5 2026 91,913,343 4,000,689+19 95,914,051',
         hutter_total=95_914_051, gpu='no', pretrained='transformer derived from fx2-cmix-transformer (trained offline on enwik9)',
         status='Hutter prize submission, pending (Oct. 5 2026)'),
    dict(id='makimono', method='makimono', version='1.0', options='', q='makimono 1.0',
         e9=95_793_872, dec=0, dk='xd', total=95_793_872, sfx=True, alg='Tr', note=123,
         comp=3_984_978, comp_q='makimono 1.0 95,793,872 3,984,978 99,788,850', hutter_total=99_788_850,
         gpu='no', pretrained='zmix with retrained transformer weights', status='Hutter-formatted, not eligible'),
    dict(id='cmix-lex-transformer', method='cmix-lex-transformer', version='lexth11c', options='',
         q='cmix-lex-transformer lexth11c', e9=95_836_613, dec=0, dk='xd', total=95_836_613, sfx=True, alg='Tr', note=117,
         comp=3_477_137, comp_q='cmix-lex-transformer lexth11c 95,836,613 3,477,137 99,313,750', hutter_total=99_313_750,
         gpu='no', pretrained='transformer weights retrained externally', status='Hutter prize entry, pending'),
    dict(id='zmix', method='zmix', version='1.0', options='', q='zmix 1.0',
         e9=96_096_261, dec=0, dk='xd', total=96_096_261, sfx=True, alg='Tr', note=115,
         comp=3_216_163, comp_q='zmix 1.0 96,096,261 3,216,163 99,312,424', hutter_total=99_312_424,
         params='~6M (fx2-cmix-transformer architecture, 4-bit weights)', gpu='no (single CPU core)',
         pretrained='transformer retrained on enwik9', status='Hutter prize entry, pending'),
    dict(id='rata-cmix', method='rata-cmix', version='1.0.0', options='', q='rata-cmix 1.0.0',
         e9=96_849_689, dec=0, dk='xd', total=96_849_689, sfx=True, alg='Tr', note=113,
         comp=3_369_432, comp_q='RATA-CMIX 1.0.0 96,849,689 3,369,432', gpu='no',
         pretrained='fx2-cmix-transformer lineage', status='not a Hutter candidate (Hutter-style rules)'),
    dict(id='fx2-cmix-t-ft8', method='fx2-cmix-t-ft8', version='Oct 10 2026', options='', q='fx2-cmix-t-ft8',
         e9=96_945_005, dec=0, dk='xd', total=96_945_005, sfx=True, alg='Tr', note=121,
         comp=3_422_266, comp_q='fx2-cmix-t-ft8 96,945,005 3,422,266 100,367,271', hutter_total=100_367_271,
         params='~6M', gpu='no', pretrained='fx2-cmix-transformer weights + 8 more epochs', status='Hutter-formatted, not eligible'),
    dict(id='fx2-cmix-transformer', method='fx2-cmix-transformer', version='24 Jul 2026', options='', q='fx2-cmix-transformer',
         e9=96_996_198, dec=0, dk='xd', total=96_996_198, sfx=True, alg='Tr', note=108,
         comp=3_428_474, comp_q='24 Jul 2026 96,996,198 3,428,474 100,424,672', hutter_total=100_424_672,
         params='6M (transformer, pre-trained on enwik9 on 8 RTX 5090 GPUs for 26 h)', gpu='no (runs on one CPU core)',
         pretrained='enwik9', status='Hutter prize awarded Sept. 30 2026 (current record)'),
    dict(id='fractal-zip', method='fractal-zip', version='hutter-2026-09', options='', q='fractal-zip',
         e8=12_303_845, e9=97_617_144, dec=0, dk='xd', total=97_617_144, sfx=True, alg='Tr', note=118,
         comp=3_836_817, comp_q='fractal-zip 12,303,845 97,617,144 3,836,817 101,453,961', hutter_total=101_453_961,
         gpu='no', pretrained='derived from cmix-lex-transformer (weights trained on enwik9, which contains enwik8)',
         status='Hutter-formatted, not eligible',
         e8_note='enwik8 is the first 10% of enwik9, so a model pretrained on enwik9 has seen it: not comparable to from-scratch enwik8 results'),
    dict(id='tufazip', method='tufazip', version='v2', options='', q='tufazip v2',
         e9=100_958_324, dec=4_430_029, dk='sd', total=105_388_353, sfx=False, alg='Tr', note=112,
         gpu='yes: 2x NVIDIA B200 (7.5 days per direction for v0.1.0)', pretrained='transformer weights trained offline, quantized'),
    dict(id='altxs', method='altxs', version='1.0.0', options='', q='altxs 1.0.0',
         e9=93_434_410, dec=13_490_401, dk='xd', total=106_924_811, sfx=False, alg='Tr', note=110,
         gpu='yes: CUDA GPU (GH100 94 GB; ~80 GB GPU memory; 63 h compress, 330 h decompress)',
         pretrained='transformer weights trained on enwik9 (shipped in the 13.5 MB decompressor)'),
    dict(id='nncp', method='nncp', version='v3.2 (2023-10-21)', options='', q='nncp v3.2',
         e8=14_915_298, e9=106_632_363, dec=628_955, dk='xd', total=107_261_318, sfx=False, alg='Tr', note=88,
         params='not stated for v3.2 (LTCB: v3 transformer ~199M parameters; v3.2 "larger")',
         gpu='yes for usable speed (LTCB test: RTX 3090; readme: RTX 4090, enwik9 2.8 days)',
         pretrained='no: trained online during (de)compression'),
    dict(id='cmix', method='cmix', version='v21', options='-t', q='cmix v21',
         e8=14_623_723, e9=107_963_380, dec=281_387, dk='sd', total=108_244_767, sfx=False, alg='CM', note=83,
         gpu='no (CPU, ~31 GB RAM, ~7 days on enwik9)', pretrained='no (dictionary + article reordering preprocessing)'),
    dict(id='paq8px', method='paq8px', version='v206fix1', options='-12L', q='paq8px_v206fix1 -12L',
         e8=15_849_084, e9=124_696_410, dec=402_949, dk='s', total=125_099_359, sfx=False, alg='CM', note=93,
         gpu='no', pretrained='no', status='latest paq8px listed on LTCB (Silesia page lists v217)'),
    dict(id='paq8px-12LT', method='paq8px', version='v206fix1', options='-12LT', q='paq8px_v206fix1 -12LT 15,799,749',
         e8=15_799_749, e9=124_619_348, dec=512_427, dk='sd', total=125_131_775, sfx=False, alg='CM', note=92, main=False,
         gpu='no', pretrained='pre-trains models on shipped English dictionaries (counted in decompressor size)'),
    dict(id='durilca', method="durilca'kingsize", version='', options='-m13000 -o41 -t2', q="durilca'kingsize -m13000",
         e8=16_209_219, e9=127_376_595, dec=406_700, dk='xd', total=127_783_295, sfx=False, alg='PPM', note=109,
         status='best PPM-class entry on LTCB'),
    dict(id='ppmonstr', method='ppmonstr', version='J', options='-m1700 -o16', q='ppmonstr J -m1700 -o16',
         e8=19_055_092, e9=157_007_383, dec=42_019, dk='x', total=157_049_402, sfx=False, alg='PPM', note=None),
    dict(id='lstm-compress', method='lstm-compress', version='v3', options='', q='lstm-compress v3',
         e8=20_318_653, e9=173_874_407, dec=144_567, dk='s', total=174_018_974, sfx=False, alg='LSTM', note=83,
         pretrained='no (online LSTM)'),
    dict(id='7zip-ppmd-4.46a', method='7-Zip PPMd (var. H based)', version='4.46a', options='-m0=ppmd:mem=1630m:o=10 -sfx7xCon.sfx',
         q='7zip 4.46a -m0=ppmd:mem=1630m:o=10', e8=21_197_559, e9=178_965_454, dec=0, dk='xd', total=178_965_454,
         sfx=True, alg='PPM', note=23),
    dict(id='7zip-ppmd-9.12b', method='7-Zip PPMd (var. H based)', version='9.12b', options='-m0=ppmd:mem=2048m:o=10',
         q='7zip 9.12b -m0=ppmd:mem=2048m:o=10', e8=21_060_863, e9=177_187_967, sfx=False, alg='PPM', note=42, main=False,
         status='no decompressor size listed, so not in the ranking'),
    dict(id='ppmd-J1', method='ppmd', version='J1 (var. J)', options='-m256 -o10 -r1', q='ppmd J1 -m256 -o10 -r1',
         e8=21_388_296, e9=183_964_915, dec=11_099, dk='s', total=183_976_014, sfx=False, alg='PPM', note=None),
    dict(id='winrar-ppm', method='WinRAR PPM mode (PPMd var. I based)', version='3.60b3', options='-mc7:128t+ -sfxWinCon.sfx',
         q='WinRAR 3.60b3 -mc7:128t+', e8=22_713_569, e9=198_454_545, dec=0, dk='xd', total=198_454_545, sfx=True, alg='PPM',
         note=None, status='LTCB: "ppmd var. I, an earlier version of ppmd J" with text optimizations; no plain var. I row exists'),
    dict(id='xz', method='xz', version='5.2.1', options='--lzma2=preset=9e,dict=1GiB,lc=4,pb=0', q='xz 5.2.1',
         e8=24_703_772, e9=197_331_816, dec=36_752, dk='xd', total=197_368_568, sfx=False, alg='LZ77', note=73),
    dict(id='xz-9e', method='xz', version='5.0.1', options='-9 -e', q='xz 5.0.1 -9 -e 24,831,648',
         e8=24_831_648, alg='LZ77', note=26, main=False),
    dict(id='xz-9', method='xz', version='5.0.1', options='-9', q='-9 24,865,244 2600',
         e8=24_865_244, alg='LZ77', note=26, main=False, status='the xz enwik8 figure Bellard quotes on the nncp and ts_zip pages'),
    dict(id='7zip-lzma', method='7-Zip LZMA', version='4.42', options='-mx=9 -sfx7zCon.sfx', q='7zip 4.42 -mx=9 -sfx7zCon.sfx',
         e8=24_996_113, e9=213_490_979, dec=0, dk='xd', total=213_490_979, sfx=True, alg='LZMA', note=None, main=False),
    dict(id='zstd', method='zstd', version='0.6.0', options='-22 --ultra', q='zstd 0.6.0 -22 --ultra',
         e8=25_405_601, e9=215_674_670, dec=69_687, dk='s', total=215_744_357, sfx=False, alg='LZ77', note=76,
         status='strongest zstd on LTCB; no --long result and no zstd >= 1.0 result is listed'),
    dict(id='brotli', method='brotli', version='18-Feb-2016', options='-q 11 -w 24', q='brotli 18-Feb-2016 -q 11 -w 24',
         e8=25_764_698, e9=223_597_884, dec=542_385, dk='s', total=224_140_269, sfx=False, alg='LZ77', note=48),
    dict(id='bzip2', method='bzip2', version='1.0.2', options='-9', q='bzip2 1.0.2 -9',
         e8=29_008_736, e9=253_977_839, dec=30_036, dk='x', total=254_007_875, sfx=False, alg='BWT', note=None),
    dict(id='gzip', method='gzip', version='1.3.5', options='-9', q='gzip 1.3.5 -9',
         e8=36_445_248, e9=322_591_995, dec=38_801, dk='x', total=322_630_796, sfx=False, alg='LZ77', note=None),
]
LTCB_NOTES = {  # LTCB "Note" column -> who ran the test (from the notes list on the page); None = Matt Mahoney
    124: 'Ryan Landay (author), Threadripper 3970X', 119: 'Marcus Woelk (author), GCP c2d-highcpu-8',
    123: 'Aleksandr Sklyarenko (author), EPYC 9755 VM', 117: 'Neel Patel (author), Modal 2 vCPU Xeon',
    115: 'James Byrne (author), Ryzen 9 5900X', 113: 'Dharmesh Patel (author), GCP n4d-highmem-2',
    121: 'Jesus Mejuto Gonzalez (author), Ryzen 9 5900X', 108: 'James Bowery, Ryzen 9 5900X',
    118: 'Scott Brooks (author), Core i7-11850H', 112: 'Tommy He (author), DGX B200',
    110: 'Conrad Lippert-Zajaczkowski (author), EPYC 9V84 + GH100', 88: 'Fabrice Bellard (author), Xeon E3-1230 v6 + RTX 3090',
    83: 'Byron Knoll (author), Core i7-7700K', 93: 'Sportman, Core i9 12900KS', 92: 'Darek, Core i9-8950HK',
    109: 'Zeelin, 2x Xeon Gold 6342 + Wine', 73: 'Andreas M. Nilsson, Core i7 (macOS)', 76: 'Yann Collet (author), Core i7-3930K',
    48: 'Matt Mahoney, Core i7 M620 (Linux)', 26: 'Matt Mahoney, Pentium T3200', 23: 'Matt Mahoney (note 23 gives the options)',
    42: 'Michal Hajicek, AMD Phenom II 965', None: 'Matt Mahoney (maintainer)',
}

# ---------------------------------------------------------------- Silesia (12 files compressed separately, data only)
SILESIA_ROWS = [  # (method, version/options as printed, total bytes, quote)
    ('paq8px', 'v217 -12L', 27_682_006, 'paq8px_v217 -12L'),
    ('paq8px', 'v206 -12TL', 28_241_197, 'paq8px_v206 -12TL'),
    ('cmix (after precomp v0.4.7 -cn)', 'v21', 28_261_094, 'precomp v0.4.7 -cn | cmix v21'),
    ('ppmd', '-m256 -o16 -r1 (version not printed)', 47_021_495, 'ppmd -m256 -o16 -r1'),
    ('7-Zip (LZMA)', '-mx=9', 48_792_760, '7zip -mx=9'),
    ('brotli', '18-Feb-2016 -q 11 -w 24', 50_297_179, 'brotli 18-Feb-2016 -q 11 -w 24'),
    ('zstd', '0.6.0 -22', 53_321_756, 'zstd 0.6.0 -22'),
    ('zstd', '"5.1.0" -21 (version string as printed)', 53_862_053, 'zstd 5.1.0 -21'),
    ('bzip2', '(no options)', 54_506_769, ' bzip2'),
    ('gzip', '-9', 67_631_990, 'gzip -9'),
]
CMIX_SILESIA_FILES = {  # exact bytes from byronknoll.com/cmix.html (version not printed there; totals match v21)
    'dickens': 1_802_071, 'mozilla': 6_634_210, 'mr': 1_828_423, 'nci': 781_325, 'ooffice': 1_221_977, 'osdb': 1_963_597,
    'reymont': 704_817, 'samba': 1_588_875, 'sao': 3_726_502, 'webster': 4_271_915, 'xml': 233_696, 'x-ray': 3_503_686,
}

# ---------------------------------------------------------------- language-model bpc (enwik8 / text8 test = last 5M chars)
# (benchmark, method, params, bpc, source key, quote that must appear verbatim, primary?, notes)
LM = [
    ('enwik8-test', 'Transformer-XL 24L', '277M', 0.99, 'txl', 'Ours - 24L Transformer-XL 277M 0.99', True, 'Dai et al. 2019, Table 2'),
    ('enwik8-test', 'Transformer-XL 18L', '88M', 1.03, 'txl', 'Ours - 18L Transformer-XL 88M 1.03', True, 'Dai et al. 2019, Table 2'),
    ('enwik8-test', 'Transformer-XL 12L', '41M', 1.06, 'txl', 'Ours - 12L Transformer-XL 41M 1.06', True, 'Dai et al. 2019, Table 2'),
    ('enwik8-test', 'Al-Rfou et al. T64 (64L, ctx 512)', '235M train / 219M inference', 1.06, 'alrfou',
     'T12 (ours) 44 41 1.11 T64 (ours) 235 219 1.06', True, 'Al-Rfou et al. 2018 (AAAI 2019), Table 3'),
    ('enwik8-test', 'Al-Rfou et al. T12 (12L, ctx 512)', '44M train / 41M inference', 1.11, 'alrfou',
     'T12 (ours) 44 41 1.11 T64 (ours) 235 219 1.06', True, 'Al-Rfou et al. 2018, Table 3'),
    ('enwik8-test', 'SHA-RNN (4-layer SHA-LSTM, attention head per layer)', '54M', 1.068, 'sharnn',
     'attention head per layer) 4 1.096 1.068 54M', True, 'Merity 2019, Table 1 (valid 1.096)'),
    ('enwik8-test', 'SHA-RNN (4-layer SHA-LSTM, single attention head)', '52M', 1.076, 'sharnn',
     'single attention head) 1 1.100 1.076 52M', True, 'Merity 2019, Table 1 (valid 1.100)'),
    ('enwik8-test', 'Compressive Transformer 24L', 'not stated in paper (277M per Longformer Table 3, secondary)', 0.97,
     'compressive', '24L Compressive Transformer 0.97', True, 'Rae et al. 2019 (ICLR 2020), Table 4'),
    ('enwik8-test', 'Adaptive-Span 24L', '209M', 0.98, 'adaptive', 'Adaptive 24 209M 181M 1.00 / 0.98', True,
     'Sukhbaatar et al. 2019, Table 2 (dev 1.00)'),
    ('enwik8-test', 'Adaptive-Span 12L (small)', '39M', 1.02, 'adaptive', 'Adaptive 12 39M 41M 1.04 / 1.02', True,
     'Sukhbaatar et al. 2019, Table 2 (dev 1.04)'),
    ('enwik8-test', 'Longformer (small)', '41M', 1.00, 'longformer', 'Our Longformer 41M 1.02 1.00', True,
     'Beltagy et al. 2020, Table 2 (dev 1.02)'),
    ('enwik8-test', 'Longformer (large)', '102M', 0.99, 'longformer', 'Our Longformer 102M 0.99', True, 'Beltagy et al. 2020, Table 3'),
    ('enwik8-test', 'mLSTM + dynamic evaluation (Krause et al. 2017)', '46M', 1.08, 'alrfou',
     'mLSTM + dynamic eval [ 2017 ] 46 - 1.08', False, 'secondary: quoted in Al-Rfou et al. Table 3; primary paper not checked'),
    ('enwik8-test', 'GPT-2 1542M, zero-shot (trained on WebText)', '1542M', 0.93, 'gpt2', '1542M ... 0.93 0.98', True,
     'Radford et al. 2019, Table 3; zero-shot with invertible de-tokenizers; split not restated in the table'),
    ('enwik8-test', 'GPT-2 117M, zero-shot (trained on WebText)', '117M', 1.16, 'gpt2', '117M ... 1.16 1.17', True,
     'Radford et al. 2019, Table 3'),
    ('text8-test', 'Transformer-XL 24L', '277M', 1.08, 'txl', 'Ours - 24L Transformer-XL 277M 1.08', True, 'Dai et al. 2019, Table 3'),
    ('text8-test', 'Al-Rfou et al. T64', '235M train / 219M inference', 1.13, 'alrfou',
     'T12 (ours) 44 41 1.18 T64 (ours) 235 219 1.13', True, 'Al-Rfou et al. 2018, Table 1'),
    ('text8-test', 'Al-Rfou et al. T12', '44M train / 41M inference', 1.18, 'alrfou',
     'T12 (ours) 44 41 1.18 T64 (ours) 235 219 1.13', True, 'Al-Rfou et al. 2018, Table 1'),
    ('text8-test', 'Adaptive-Span 24L', '209M', 1.07, 'adaptive', '24 245 209M 179M 1.01 1.07', True,
     'Sukhbaatar et al. 2019, Table 1 (dev 1.01)'),
    ('text8-test', 'Adaptive-Span 12L (small)', '38M', 1.11, 'adaptive', '12 314 38M 42M 1.05 1.11', True,
     'Sukhbaatar et al. 2019, Table 1 (dev 1.05)'),
    ('text8-test', 'Longformer (small)', '41M', 1.10, 'longformer', 'Our Longformer 41M 1.04 1.10', True,
     'Beltagy et al. 2020, Table 2 (dev 1.04)'),
    ('text8-test', 'GPT-2 1542M, zero-shot', '1542M', 0.98, 'gpt2', '1542M ... 0.93 0.98', True, 'Radford et al. 2019, Table 3'),
]

# ---------------------------------------------------------------- Deletang et al., Table 1 (enwik9, 2048-byte chunks)
# (method, params, raw %, adjusted %, quote) ; adjusted adds 2 bytes/parameter (float16) to the compressed size.
DELETANG = [
    ('Chinchilla 70B', '70B', 8.3, 14008.3, 'Chinchilla 70B 8.3 48.0 21.0 100.8 14008.3'),
    ('Chinchilla 7B', '7B', 10.2, 1410.2, 'Chinchilla 7B 10.2 54.7 23.6 101.6 1410.2'),
    ('Chinchilla 1B', '1B', 11.3, 211.3, 'Chinchilla 1B 11.3 62.2 24.9 108.8 211.3'),
    ('Llama 2 (7B)', '7B', 8.9, 1408.9, 'Llama 2 (7B) 8.9 53.4 23.1 103.2 1408.9'),
    ('Transformer 3.2M (trained on enwik8)', '3.2M', 17.0, 17.7, 'Transformer 3.2M 17.0 215.8 228.2 224.0 17.7'),
    ('Transformer 800K (trained on enwik8)', '800K', 21.7, 21.9, 'Transformer 800K 21.7 185.1 131.1 200.1 21.9'),
    ('Transformer 200K (trained on enwik8)', '200K', 30.9, 30.9, 'Transformer 200K 30.9 194.0 146.6 195.5 30.9'),
    ('gzip, 2048-byte chunks', None, 48.1, 48.1, '2048 2048 gzip 48.1 68.6 38.5 100.1 48.1'),
    ('LZMA2, 2048-byte chunks', None, 50.0, 50.0, 'LZMA2 50.0 62.4 38.2 100.0 50.0'),
    ('gzip, unchunked', None, 32.3, 32.3, 'gzip 32.3 70.7 36.4 100.0 32.3'),
    ('LZMA2, unchunked', None, 23.0, 23.0, 'LZMA2 23.0 57.9 29.9 100.0 23.0'),
]

# ---------------------------------------------------------------- Hutter prize (enwik9, S = compressor S1 + self-extracting archive S2)
HUTTER = [  # (holder, date, program, total S, quote, status)
    ('Vladimir Ivanov', '24 Jul 2026', 'fx2-cmix-T (fx2-cmix-transformer)', 100_424_672,
     "Vladimir Ivanov 24.Jul 2026 fx2-cmix-T 100'424'672 9.96 | ~10GB | ~50h 7.46%| 37'300", 'current record; award 7.46% = 37,300 EUR'),
    ('David Freelan', '19 Jul 2026', 'cmix-obias', 108_521_870,
     "David Freelan 19.Jul 2026 cmix-obias 108'521'870", 'previous record'),
    ('Ibrahim Marcouch & Kaido Orav', '26 Jun 2026', 'cmix-lex', 109_671_639, "26.Jun 2026 cmix-lex 109'671'639", 'previous record'),
    ('Kaido Orav & Byron Knoll', '3 Sep 2024', 'fx2-cmix', 110_793_128, "3.Sep 2024 fx2-cmix 110'793'128", 'previous record'),
]
HUTTER_NEXT = (99_420_425, "<99'420'425")
JEVZIP_PARAMS = 11_140_000  # weights/jevzip-lanes-11m-*.pt, "11.14M params" per the project brief (rounded)
EXTRA_CHECKS = [  # statements the markdown relies on
    ('ltcb', 'Last update: Oct. 14, 2026.'), ('ltcb', 'I have not verified results submitted by others.'),
    ('ltcb', 'pre-trained on enwik9 on 8 RTX 5090 GPUs for 26 hours'), ('ltcb', 'cmix v13 15,323,969 120,480,684'),
    ('ltcb', 'equivalent to ppmd var H (with minor changes)'), ('ltcb', '(actually ppmd var. I, an earlier version of ppmd J)'),
    ('silesia', 'Programs listed are the latest version available as of Oct. 10, 2026.'),
    ('alrfou', 'we split the data into 90M, 5M and 5M for training, dev and test respectively'),
    ('deletang', 'In practice, the large models had only access to the top-k next token log-probabilities'),
    ('ts_zip', 'It is experimental so no backward compability should be expected'),
    ('zstd_releases', '"tag_name": "v1.5.7"'), ('zstd_releases', '"published_at": "2025-02-19T21:50:11Z"'),
    ('zstd_releases', '"tag_name": "v1.5.1"'), ('paq8px_tags', '"name": "v217"'),
]
HUTTER_RULE = 'Must run in ≲50 hours using a single CPU core and <10GB RAM and <100GB HDD'


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/8.7.1'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def norm(s):
    return re.sub(r'\s+', ' ', s)


def page_views(raw):
    """(lines, text): tag-stripped, entity-decoded, whitespace-collapsed; lines keeps physical lines."""
    s = raw.decode('utf-8', 'replace')
    s = re.sub(r'(?is)<(script|style)\b.*?</\1>', ' ', s)
    s = html.unescape(re.sub(r'<[^>]+>', ' ', s))
    return [norm(x).strip() for x in s.splitlines()], norm(s)


def as_printed(value, quote):
    """The value as the source prints it (1.00 stays 1.00), i.e. the longest of its decimal forms in the quote."""
    forms = [f for f in (str(value), f'{value:.2f}', f'{value:.3f}') if f in quote]
    return max(forms, key=len) if forms else str(value)


def number_forms(n):
    c = f'{n:,}'
    return {c, str(n), c.replace(',', "'"), c.replace(',', ' ')}


class Checker:
    def __init__(self, offline):
        self.offline, self.pages, self.sha, self.failures = offline, {}, {}, []

    def view(self, key):
        if key not in self.pages:
            raw = fetch(SOURCES[key][0])
            self.sha[key] = hashlib.sha256(raw).hexdigest()
            self.pages[key] = page_views(raw)
        return self.pages[key]

    def ltcb_main(self):
        """Lines of the LTCB main ranking table, in rank order."""
        raw = fetch(SOURCES['ltcb'][0]).decode('utf-8', 'replace')
        body = raw[raw.index('<pre><ol>'):raw.index('</ol></pre>')]
        return [page_views(x.encode())[1].strip() for x in body.split('<li>')[1:]]

    def typed(self, value, quote):
        """The typed value must be part of the quoted row itself (catches a value edited without its quote)."""
        forms = number_forms(value) if isinstance(value, int) else {str(value), f'{value:.2f}', f'{value:.3f}'}
        if not any(f in quote for f in forms):
            self.failures.append(f'typed value {value} not in its own quote {quote!r}')
            return False
        return True

    def check(self, key, quote, numbers=(), scope=None):
        """True if quote occurs in source key and every number appears (in some digit grouping) on that line /
        within 200 chars after it. None when offline or the source is a manual one."""
        if self.offline or SOURCES[key][2] == 'manual':
            return None
        lines, text = self.view(key)
        hay = scope if scope is not None else (lines if SOURCES[key][2] == 'lines' else None)
        if hay is not None:
            cands = [ln for ln in hay if quote in ln]
        else:
            cands = [text[m.start():m.end() + 200] for m in re.finditer(re.escape(quote), text)]
        good = [c for c in cands if all(any(f in c for f in number_forms(n)) if isinstance(n, int) else n in c for n in numbers)]
        self.match = good[0].strip() if good and hay is not None else None  # the full matched row, kept as the quote
        if not good:
            self.failures.append(f'{key}: {quote!r} {list(numbers)}')
        return bool(good)


def bpb(nbytes, orig):
    return round(8 * nbytes / orig, 6), f'8 * {nbytes:,} / {orig:,} = {8 * nbytes / orig:.6f}'


def build(offline):
    ck = Checker(offline)
    recs = []
    main = None if offline else ck.ltcb_main()
    for e in LTCB:
        rank = None
        nums = [e[k] for k in ('e8', 'e9', 'dec', 'total') if e.get(k) is not None]
        if main is not None and e.get('main', True):
            hits = [i for i, ln in enumerate(main) if ln.startswith(e['q'])]
            rank = hits[0] + 1 if hits else None
            ok = ck.check('ltcb', e['q'], nums, scope=main)
        else:
            ok = ck.check('ltcb', e['q'], nums)
        row = ck.match if ok else None
        if e.get('comp') and ok is not None:  # the compressor size is quoted from the program's detail section
            ok = ck.check('ltcb', e['comp_q']) and ok
        tested = LTCB_NOTES.get(e.get('note'), f"see LTCB note {e.get('note')}")
        common = dict(benchmark='LTCB', method=e['method'], version=e['version'], options=e['options'],
                      algorithm=e['alg'], ltcb_rank=rank, ltcb_note=e.get('note'), tested_by=tested,
                      params=e.get('params'), gpu=e.get('gpu'), pretrained=e.get('pretrained'), status=e.get('status'),
                      source_url=cite('ltcb'), source_quote=row or e['q'],
                      source_quote_compressor=e.get('comp_q'), verified=bool(ok) if ok is not None else None,
                      verification='typed numbers found on the quoted row of the live page' if ok else
                                   ('not re-checked (offline build)' if ok is None else 'CHECK FAILED'))
        if e.get('e8') is not None:
            b, a = bpb(e['e8'], E8)
            if e['id'] == 'fractal-zip':  # enwik8 shown only in its detail table; archive form not stated
                cd, kind = None, 'unknown (archive form not stated for enwik8)'
            else:  # SFX options (7-Zip, WinRAR) apply to the enwik8 run too
                cd, kind = bool(e.get('sfx')), 'self-extracting archive' if e.get('sfx') else 'compressed data only'
            recs.append(dict(common, id=f"ltcb-enwik8-{e['id']}", dataset='enwik8', original_size=E8, size_bytes=e['e8'],
                             counts_decompressor=cd, size_kind=kind, bpb=b, bpb_arithmetic=a, notes=e.get('e8_note')))
        if e.get('e9') is not None:
            b, a = bpb(e['e9'], E9)
            r = dict(common, id=f"ltcb-enwik9-{e['id']}", dataset='enwik9', original_size=E9, size_bytes=e['e9'],
                     counts_decompressor=bool(e.get('sfx')),
                     size_kind='self-extracting archive (decompressor inside)' if e.get('sfx') else 'compressed data only',
                     decompressor_bytes=e.get('dec'), decompressor_kind=e.get('dk'), total_bytes=e.get('total'),
                     bpb=b, bpb_arithmetic=a)
            if e.get('total') is not None:
                r['bpb_total'], r['bpb_total_arithmetic'] = bpb(e['total'], E9)
            if e.get('comp'):
                r['compressor_bytes'] = e['comp']
                ht = e.get('hutter_total') or e['e9'] + e['comp']
                r['hutter_style_total'] = ht
                r['hutter_style_total_note'] = ('as printed on LTCB' if e.get('hutter_total') else
                                                f"computed here: {e['e9']:,} + {e['comp']:,} = {ht:,}")
                r['bpb_hutter_style'], r['bpb_hutter_style_arithmetic'] = bpb(ht, E9)
            recs.append(r)

    # nncp + ts_zip pages (Bellard)
    ok8 = ck.check('nncp', 'NNCP (2023-10-21) 14 915 298 1.19', [])
    ok9 = ck.check('nncp', 'NNCP (2023-10-21) 106 632 363 0.853 628 955 107 261 318', [])
    okr = ck.check('nncp_readme', 'Result: 106632363 bytes (2.8 days)', []) and ck.check('nncp_readme', 'Result: 14915298 bytes (13.2 hours)', [])
    gpu_nncp = 'GPU needed for acceptable speed with large models (readme: CUDA 11/12, Ampere/Ada/Hopper; test system Ryzen 9 3900 + RTX 4090; CPU 1 GB, GPU 6.6 GB)'
    for ds, orig, size, total, quote, ok in (('enwik8', E8, 14_915_298, None, 'NNCP (2023-10-21) 14 915 298 1.19', ok8),
                                            ('enwik9', E9, 106_632_363, 107_261_318, 'NNCP (2023-10-21) 106 632 363 0.853 628 955 107 261 318', ok9)):
        b, a = bpb(size, orig)
        ok = ok and ck.typed(size, quote) and (total is None or ck.typed(total, quote)) if ok is not None else None
        r = dict(id=f'nncp-page-{ds}', benchmark='bellard.org/nncp', dataset=ds, original_size=orig, method='nncp',
                 version='v3.2 results (2023-10-21); v3.3 (2024-06-05) is the current download (Windows/CUDA packaging only)',
                 options='--cuda --profile ' + ds + (' --preprocess 4096,512' if ds == 'enwik8' else ' --preprocess 16384,512'),
                 size_bytes=size, counts_decompressor=False, size_kind='compressed data only', bpb=b, bpb_arithmetic=a,
                 params='not stated on page', gpu=gpu_nncp, pretrained='no (online training; model size never counted)',
                 source_url=cite('nncp'), source_quote=quote, verified=(ok and okr) if ok is not None else None,
                 notes='same bytes as the LTCB nncp v3.2 row; readme gives the exact command lines and timings (enwik8 13.2 h, enwik9 2.8 days)')
        if total:
            r.update(decompressor_bytes=628_955, total_bytes=total)
            r['bpb_total'], r['bpb_total_arithmetic'] = bpb(total, orig)
        recs.append(r)
    tsq = {'enwik8': 'enwik8 100000000 24865244 1.989 13825741 1.106', 'enwik9': 'enwik9 1000000000 213370900 1.707 135443237 1.084'}
    okm = ck.check('ts_zip', 'the RWKV 169M v4 language model', []) and ck.check('ts_zip', 'A GPU is necessary to get a reasonable speed. 4 GB of RAM is required.', [])
    for ds, orig, size, xz in (('enwik8', E8, 13_825_741, 24_865_244), ('enwik9', E9, 135_443_237, 213_370_900)):
        ok = ck.check('ts_zip', tsq[ds], [])
        ok = ok and ck.typed(size, tsq[ds]) and ck.typed(xz, tsq[ds]) if ok is not None else None
        b, a = bpb(size, orig)
        recs.append(dict(id=f'ts_zip-{ds}', benchmark='bellard.org/ts_zip', dataset=ds, original_size=orig, method='ts_zip',
                         version='2024-03-02', options='', size_bytes=size, counts_decompressor=False,
                         size_kind='compressed data only (169M-parameter model not counted)', bpb=b, bpb_arithmetic=a,
                         params='169M (RWKV 169M v4, quantized to 8 bits/parameter, evaluated in BF16)',
                         gpu='"A GPU is necessary to get a reasonable speed"; 4 GB RAM; up to 1 MB/s on RTX 4090',
                         pretrained='yes: RWKV 169M v4 trained mostly on English text (not on enwik specifically)',
                         source_url=cite('ts_zip'), source_quote=tsq[ds], verified=(ok and okm) if ok is not None else None,
                         notes=(f'page also lists xz at {xz:,} bytes for comparison (options not stated; LTCB xz 5.2.1 gives 197,331,816 on enwik9). '
                                'If the 8-bit model (~169e6 bytes) were counted it would add ~13.5 bpb on enwik8 / ~1.35 bpb on enwik9 '
                                '(estimate: 8*169e6/1e8 and 8*169e6/1e9).')))

    # Hutter prize
    okr = ck.check('hutter', HUTTER_NEXT[1]) and ck.check('hutter', HUTTER_RULE)
    for i, (who, date, prog, s, quote, status) in enumerate(HUTTER):
        ok = ck.check('hutter', quote)
        ok = ok and ck.typed(s, quote) if ok is not None else None
        b, a = bpb(s, E9)
        recs.append(dict(id=f'hutter-{i}', benchmark='Hutter Prize (enwik9)', dataset='enwik9', original_size=E9, method=prog,
                         version=date, options='', holder=who, size_bytes=s, counts_decompressor=True,
                         size_kind='S = S1 (compressor comp.exe) + S2 (self-extracting archive.exe)', bpb=b, bpb_arithmetic=a,
                         gpu='not allowed: ' + HUTTER_RULE,
                         status=status, source_url=cite('hutter'), source_quote=quote,
                         verified=(ok and okr) if ok is not None else None,
                         notes=f'next claim must beat L - 1%: total < {HUTTER_NEXT[0]:,} (as printed: "{HUTTER_NEXT[1]}")'))

    # Deletang et al.
    okc = ck.check('deletang_code', 'NUM_CHUNKS = 488281', []) and ck.check('deletang_code', 'CHUNK_SIZE_BYTES = 2048', [])
    okp = ck.check('deletang', 'Therefore, we chunk all datasets into sequences of 2048', []) and \
        ck.check('deletang', 'We encode each neural network parameter with 2 bytes', [])
    for name, params, raw, adj, quote in DELETANG:
        ok = ck.check('deletang', quote, [])
        ok = ok and ck.typed(raw, quote) and ck.typed(adj, quote) if ok is not None else None
        recs.append(dict(id='deletang-' + re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-'),
                         benchmark='Deletang et al. 2024 (ICLR), Table 1', dataset='enwik9 (2048-byte chunks, each coded with no history)'
                         if 'unchunked' not in name else 'enwik9 (whole file)',
                         original_size=ENWIK9_CHUNKED if 'unchunked' not in name else E9, method=name, version='', options='',
                         size_bytes=None, raw_rate_percent=raw, adjusted_rate_percent=adj,
                         counts_decompressor=False, size_kind='raw rate: data only; adjusted rate: data + 2 bytes/parameter (float16)',
                         bpb=round(8 * raw / 100, 4), bpb_arithmetic=f'8 * {raw}% = {8 * raw / 100:.3f} (rate printed to 0.1%, so +-0.004 bpb)',
                         bpb_adjusted=round(8 * adj / 100, 4), bpb_adjusted_arithmetic=f'8 * {adj}% = {8 * adj / 100:.3f}',
                         params=params,
                         source_url=cite('deletang'), source_quote=quote,
                         verified=(ok and okc and okp) if ok is not None else None,
                         notes=('LLMs see only the top-100 token log-probs (renormalized), 2048 ASCII chars per chunk, bytes >127 '
                                'have the MSB zeroed and the lost bit appended; small Transformers are byte-level, trained on enwik8 '
                                '(enwik9 is partly out-of-distribution). Classical compressors count their header once.'
                                if 'unchunked' not in name else 'classical compressor on the whole 1 GB file')))

    # LM bpc
    for bench, meth, params, val, key, quote, primary, notes in LM:
        if SOURCES[key][2] == 'manual':
            ok = True  # read from the PDF (sha256 GPT2_PDF_SHA256) with pdftotext -layout, page 5
        else:
            ok = ck.check(key, quote, [])
        ok = ok and ck.typed(val, quote) if ok is not None else None
        orig = 5_000_000
        recs.append(dict(id='lm-' + bench + '-' + re.sub(r'[^a-z0-9]+', '-', meth.lower()).strip('-'), benchmark=bench + ' (LM bpc)',
                         dataset=bench.replace('-test', '') + ' test split (last 5,000,000 chars of a 90M/5M/5M split)',
                         original_size=orig, method=meth, version='', options='', size_bytes=None,
                         counts_decompressor=False, size_kind='cross-entropy (no coder, no model size)',
                         bpb=val, reported=as_printed(val, quote),
                         bpb_arithmetic='reported directly as bits per character (enwik8: 1 char = 1 byte)',
                         params=params, source_url=cite(key), checked_via=SOURCES[key][0] or f'PDF sha256 {GPT2_PDF_SHA256}',
                         source_quote=quote, verified=(bool(ok) and primary) if ok is not None else None, notes=notes))

    # Silesia
    sil = None if offline else ck.view('silesia')[0]
    for meth, ver, total, quote in SILESIA_ROWS:
        per_file = None
        ok = ck.check('silesia', quote, [total])
        row = ck.match if ok else None
        if sil is not None:
            for ln in sil:
                t = ln.split()
                if quote.strip() in ln and t and t[0] == str(total):
                    names = ['dickens', 'mozilla', 'mr', 'nci', 'ooffice', 'osdb', 'reymont', 'samba', 'sao', 'webster', 'x-ray', 'xml']
                    per_file = dict(zip(names, map(int, t[1:13])))
        note = None
        if per_file:  # each column is floor(bytes / 1000), so the 12 columns must sum to (total/1000 - 12, total/1000]
            kb = sum(per_file.values())
            if not total / 1000 - 12 < kb <= total / 1000:
                note = (f'per-file KB columns sum to {kb:,} KB, inconsistent with the total {total:,} bytes '
                        f'(allowed ({total / 1000 - 12:,.0f}, {total // 1000:,}]): a typo in a per-file column on the page')
                if 'cmix v21' in quote:
                    note += f"; dickens is printed 1892 but cmix's own page gives {CMIX_SILESIA_FILES['dickens']:,} bytes (1802 KB)"
        b, a = bpb(total, SILESIA)
        recs.append(dict(id='silesia-' + re.sub(r'[^a-z0-9]+', '-', (meth + ' ' + ver).lower()).strip('-'), benchmark='Silesia', notes=note,
                         dataset='Silesia corpus (12 files, each compressed separately)', original_size=SILESIA, method=meth,
                         version=ver, options='', size_bytes=total, counts_decompressor=False, size_kind='sum of compressed files',
                         bpb=b, bpb_arithmetic=a, per_file_kb_truncated=per_file, source_url=cite('silesia'),
                         source_quote=row or quote.strip(), verified=bool(ok and per_file) if ok is not None else None))
    ok = ck.check('cmix', 'silesia 211938580 28261094', [])
    okf = None
    if not offline:  # each "<file> <original> <compressed>" triple of the per-file table
        text = ck.view('cmix')[1]
        bad = [k for k, v in CMIX_SILESIA_FILES.items() if not re.search(rf'(?<![\w-]){re.escape(k)} \d+ {v}(?!\d)', text)]
        ck.failures += [f'cmix: per-file {k} {CMIX_SILESIA_FILES[k]} not found' for k in bad]
        okf = not bad
    s = sum(CMIX_SILESIA_FILES.values())
    b, a = bpb(s, SILESIA)
    recs.append(dict(id='silesia-cmix-page', benchmark='Silesia', dataset='Silesia corpus (12 files, each compressed separately)',
                     original_size=SILESIA, method='cmix (exact per-file bytes, byronknoll.com)', version='not printed (page lists v21 downloads; its enwik8/enwik9 rows equal v21)',
                     options='', size_bytes=s, counts_decompressor=False, size_kind='sum of compressed files', bpb=b,
                     bpb_arithmetic=a, per_file_bytes=CMIX_SILESIA_FILES, source_url=cite('cmix'),
                     source_quote='silesia 211938580 28261094', verified=(ok and okf and s == 28_261_094) if ok is not None else None,
                     notes=f'per-file bytes sum to {s:,} (page total 28,261,094); same total as the Silesia-benchmark row "precomp v0.4.7 -cn | cmix v21"'))
    for key, quote in EXTRA_CHECKS:
        ck.check(key, quote)
    for r in recs:  # every record carries the same core fields
        for k in REQUIRED:
            r.setdefault(k, None)
        r['accessed'] = ACCESS_DATE
    return recs, ck


REQUIRED = ('benchmark', 'dataset', 'original_size', 'method', 'version', 'size_bytes', 'bpb', 'bpb_arithmetic',
            'counts_decompressor', 'params', 'source_url', 'source_quote', 'verified')


def md(recs, ck):
    by = {r['id']: r for r in recs}
    L = []
    w = L.append
    w('# Published reference numbers (state of the art and baselines)\n')
    w(f'Collected {ACCESS_DATE} by reading each source page; machine-readable copy: `sota.json` (built and checked by '
      '`build_sota.py`). Research only: nothing here was run on this machine.\n')
    w('**Conventions.** bpb = 8 x bytes / original size, recomputed here (arithmetic for every number is in `sota.json`). '
      'enwik8 = 100,000,000 bytes, enwik9 = 1,000,000,000 bytes, Silesia = 211,938,580 bytes. '
      '"+dec" means the size includes the decompressor (and, for model-based entries, whatever weights it ships). '
      '**verified** = the number was read from the primary source on the access date and `build_sota.py` found it again on the '
      'quoted row of the live page. LTCB itself warns: "I have not verified results submitted by others", so the *tested by* '
      'column matters.\n')

    w('## 1. Large Text Compression Benchmark (LTCB)\n')
    w(f'Source: <{cite("ltcb")}> (page "Last update: Oct. 14, 2026"). LTCB ranks by enwik9 compressed size + zipped decompressor; '
      'the enwik8 column is informational. "SFX" = self-extracting archive (decompressor inside the archive, LTCB lists 0).\n')
    w('| LTCB rank | Program | Options | enwik8 bytes | enwik8 bpb | enwik9 bytes | Decompressor | enwik9 total (+dec) | bpb (enwik9 total) | Tested by | Verified |')
    w('|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|')
    for e in LTCB:
        r9, r8 = by.get(f"ltcb-enwik9-{e['id']}"), by.get(f"ltcb-enwik8-{e['id']}")
        r = r9 or r8
        dec = '' if e.get('dec') is None else ('SFX' if e.get('sfx') else f"{e['dec']:,} {e['dk']}")
        w(f"| {r.get('ltcb_rank') or '-'} | {(e['method'] + ' ' + e['version']).strip()} | {e['options'] or ''} | "
          f"{'' if r8 is None else format(r8['size_bytes'], ',')} | {'' if r8 is None else format(r8['bpb'], '.4f')} | "
          f"{'' if r9 is None else format(r9['size_bytes'], ',')} | {dec} | "
          f"{'' if r9 is None or r9.get('total_bytes') is None else format(r9['total_bytes'], ',')} | "
          f"{'' if r9 is None or r9.get('bpb_total') is None else format(r9['bpb_total'], '.4f')} | {r['tested_by']} | {r['verified']} |")
    w('\nRank "-" = row from a program\'s detail section, not the ranking table. '
      'Worked example: cmix v21 enwik8 = 8 x 14,623,723 / 100,000,000 = 1.1699 bpb; enwik9 total = 8 x 108,244,767 / 10^9 = 0.8660 bpb.\n')
    w('What the 2026 top of the table is: ranks 1-9 are Hutter-prize-style **self-extracting archives built around a small transformer '
      'trained offline on enwik9** (fx2-cmix-transformer: 6M parameters, 4-bit weights, ~2.9 MB, "pre-trained on enwik9 on 8 RTX 5090 GPUs '
      'for 26 hours"), mixed with cmix/paq context models and built for the Hutter limits (one CPU core, about 10 GB RAM; LTCB lists '
      '31-67 h per direction on the testers\' machines). The weights travel inside the '
      'archive, so they *are* counted. Their separate compressor executable (which also holds the weights) is not counted by LTCB but '
      'is counted by the Hutter prize:\n')
    w('| Program | archive9 (SFX) | compressor S1 | Hutter-style S1+S2 | bpb (S1+S2) | how S1+S2 was obtained |')
    w('|---|---:|---:|---:|---:|---|')
    for e in LTCB:
        r = by.get(f"ltcb-enwik9-{e['id']}")
        if r and r.get('compressor_bytes'):
            w(f"| {(e['method'] + ' ' + e['version']).strip()} | {r['size_bytes']:,} | {r['compressor_bytes']:,} | {r['hutter_style_total']:,} | "
              f"{r['bpb_hutter_style']:.4f} | {r['hutter_style_total_note']} |")
    w('\nGPU-bound entries: **altxs 1.0.0** needs a CUDA GPU (GH100, ~80 GB GPU memory, 330 h to decompress; its 13.5 MB decompressor '
      'holds the weights) and **tufazip v2** ran on 2x B200; **nncp v3.2** needs a GPU for acceptable speed but trains online, so it has '
      'no pretrained weights to count. Best results without any offline-trained model: cmix v21 (enwik8 14,623,723 = 1.1699 bpb, '
      'the smallest enwik8 on LTCB apart from fractal-zip, whose enwik9-trained weights have seen enwik8) and nncp v3.2 '
      '(enwik9 total 107,261,318 = 0.8581 bpb).\n')
    w('Baseline notes: LTCB has no plain PPMd var. I row; WinRAR\'s PPM mode is described there as "ppmd var. I, an earlier version of ppmd J". '
      '7-Zip PPMd is described as equivalent to "ppmd var H (with minor changes)". The strongest zstd on LTCB is 0.6.0 (2016) '
      '`-22 --ultra`; **no `--long` or zstd >= 1.0 result is published there** (none found elsewhere either: measure locally). '
      'gzip is 1.3.5 -9 and brotli is the Feb-2016 build.\n')

    w('## 2. Bellard: nncp and ts_zip\n')
    w('| Page | Program | Dataset | Bytes | bpb | Counts model / decompressor? | Model | GPU | Verified |')
    w('|---|---|---|---:|---:|---|---|---|---|')
    for i in ('nncp-page-enwik8', 'nncp-page-enwik9', 'ts_zip-enwik8', 'ts_zip-enwik9'):
        r = by[i]
        tot = f" (+dec {r['total_bytes']:,} = {r['bpb_total']:.4f})" if r.get('total_bytes') else ''
        w(f"| <{r['source_url']}> | {r['method']} | {r['dataset']} | {r['size_bytes']:,} | {r['bpb']:.4f}{tot} | "
          f"{'no (online-trained; LTCB adds the 628,955-byte program)' if r['method'] == 'nncp' else 'no: the 169M-parameter model is not counted'} | "
          f"{'online transformer' if r['method'] == 'nncp' else 'RWKV 169M v4, 8-bit weights, BF16 eval'} | "
          f"{'yes for speed (RTX 4090 in readme)' if r['method'] == 'nncp' else 'yes for reasonable speed (up to 1 MB/s on RTX 4090)'} | {r['verified']} |")
    w('\nts_zip arithmetic: 8 x 13,825,741 / 10^8 = 1.1061 bpb (page: 1.106); 8 x 135,443,237 / 10^9 = 1.0835 (page: 1.084). '
      'nncp readme commands: `./nncp --cuda --profile enwik8 --preprocess 4096,512 c enwik8 out.bin` (14,915,298 bytes, 13.2 h) and '
      '`--profile enwik9 --preprocess 16384,512` (106,632,363 bytes, 2.8 days); test system Ryzen 9 3900 + RTX 4090, CPU 1 GB, GPU 6.6 GB. '
      'ts_zip says it is experimental, with no backward compatibility between versions, and that its output does not depend on the '
      'GPU/CPU model or thread count (deterministic evaluation), which jevzip\'s codec does not guarantee (decode needs the same torch '
      'build, kernels, threads and bf16 flag as encode).\n')

    w('## 3. Hutter Prize (enwik9, total = compressor + self-extracting archive)\n')
    w(f'Source: <{cite("hutter")}>. Rules on the page: S := S1 + S2 (compressor comp.exe + archive.exe that rebuilds enwik9 with no input), '
      f'"{HUTTER_RULE}" on the committee\'s test machine, so GPU models are out.\n')
    w('| Date | Holder | Program | Total S (bytes) | bpb | Status | Verified |')
    w('|---|---|---|---:|---:|---|---|')
    for i in range(len(HUTTER)):
        r = by[f'hutter-{i}']
        w(f"| {r['version']} | {r['holder']} | {r['method']} | {r['size_bytes']:,} | {r['bpb']:.4f} | {r['status']} | {r['verified']} |")
    w(f'\nRecord arithmetic: 8 x 100,424,672 / 10^9 = 0.8034 bpb (= LTCB fx2-cmix-transformer archive 96,996,198 + compressor 3,428,474). '
      f'Next award needs total < {HUTTER_NEXT[0]:,}. Pending Hutter-style claims below that line on LTCB: cmix-neif 1.0 (95,914,051), '
      'zmix 1.0 (99,312,424), cmix-lex-transformer lexth11c (99,313,750); not verified by the committee yet.\n')

    w('## 4. Deletang et al., "Language Modeling Is Compression" (ICLR 2024)\n')
    w(f'Source: <{cite("deletang")}> (v2, Table 1), code constants <{cite("deletang_code")}>. Protocol: enwik9 cut into 2048-byte '
      f'chunks, each compressed with **no context from earlier chunks** (constants.py: 488,281 chunks x 2048 = {ENWIK9_CHUNKED:,} bytes). '
      'Raw rate = compressed/raw; adjusted rate adds 2 bytes per parameter (float16). LLMs get 2048 ASCII characters per chunk '
      '(bytes >127 have their top bit zeroed and the bit appended to the output) and only top-100 renormalized token probabilities. '
      'The small Transformers are byte-level and trained on enwik8, so enwik8 is in-distribution and the rest of enwik9 partly not.\n')
    w('| Model / compressor | Params | Raw rate | bpb (raw) | Adjusted rate | bpb (adjusted) | Verified |')
    w('|---|---:|---:|---:|---:|---:|---|')
    for name, params, raw, adj, _ in DELETANG:
        r = by['deletang-' + re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')]
        w(f"| {name} | {'' if params is None else r['params']} | {raw}% | {r['bpb']:.3f} | {adj}% | {r['bpb_adjusted']:.3f} | {r['verified']} |")
    w('\nArithmetic: bpb = 8 x rate; e.g. Chinchilla 70B 8 x 0.083 = 0.664 bpb raw, 8 x 140.083 = 1120.66 bpb adjusted. '
      'Rates are printed to 0.1%, so bpb is +-0.004. These chunked numbers are **not comparable to whole-file enwik9 results** '
      '(gzip goes from 32.3% unchunked to 48.1% chunked).\n')

    w('## 5. Language-model bpc on the standard enwik8 / text8 test split\n')
    w('Split: 90M / 5M / 5M characters; test = last 5,000,000 bytes (Al-Rfou et al.: "we split the data into 90M, 5M and 5M for training, '
      'dev and test respectively"). These are model cross-entropies, not compressed files: no arithmetic-coder overhead, '
      'no model size, and Transformer-XL-style models carry memory across segments. LM papers also list "cmix v13 1.23" for enwik8; '
      'that matches cmix v13\'s LTCB result on the **whole** 100 MB file learned from scratch (8 x 15,323,969 / 10^8 = 1.2259), '
      'so it is most likely not a test-split number.\n')
    w('| Benchmark | Model | Params | Test bpc | Source | Verified |')
    w('|---|---|---|---:|---|---|')
    for bench, meth, params, val, key, quote, primary, notes in LM:
        r = by['lm-' + bench + '-' + re.sub(r'[^a-z0-9]+', '-', meth.lower()).strip('-')]
        w(f"| {bench} | {meth} | {params} | {r['reported']} | [{notes}]({r['source_url']}) | {r['verified']} |")
    w('\nSmall-model reference points (< 50M parameters) on enwik8 test: Longformer small 41M 1.00, Adaptive-Span 39M 1.02, '
      'Transformer-XL 12L 41M 1.06, Al-Rfou T12 44M 1.11; SHA-RNN is just above at 52-54M (1.068-1.076). '
      'GPT-2 is zero-shot (trained on WebText, evaluated with invertible de-tokenizers).\n')

    w('## 6. Silesia corpus\n')
    w(f'Source: <{cite("silesia")}> ("Programs listed are the latest version available as of Oct. 10, 2026"; open-source programs only; '
      f'files compressed individually; data only, no decompressor), plus the cmix page <{cite("cmix")}> for exact per-file bytes.\n')
    w('| Program | Version / options | Total bytes | bpb | Verified |')
    w('|---|---|---:|---:|---|')
    for r in recs:
        if r['benchmark'] == 'Silesia':
            w(f"| {r['method']} | {r['version']} | {r['size_bytes']:,} | {r['bpb']:.4f} | {r['verified']} |")
    for r in recs:
        if r['benchmark'] == 'Silesia' and r.get('notes') and r['id'] != 'silesia-cmix-page':
            w(f"\nPage inconsistency, {r['method']} {r['version']}: {r['notes']}. The total is unaffected.")
    w('\nArithmetic: 8 x 27,682,006 / 211,938,580 = 1.0449 bpb (paq8px_v217 -12L, current #1). xz is not listed on the Silesia page '
      '(7-Zip -mx=9, i.e. LZMA, is the closest). The zstd row printed as "5.1.0" is reproduced verbatim; GitHub\'s newest zstd release is '
      'v1.5.7 (2025-02-19) and there is no 5.x, so it is probably v1.5.1 (unconfirmed). GitHub\'s newest paq8px tag is v217, the version '
      'leading this table (LTCB\'s newest paq8px is v206fix1). Per-file sizes on the benchmark page are KB (truncated); exact cmix per-file bytes '
      'are in `sota.json` (`per_file_bytes`).\n')

    w('## 7. Reading jevzip numbers against these\n')
    p = JEVZIP_PARAMS
    w(f'- **enwik8 test split (last 5M bytes)** is the only protocol-matched comparison for a model trained on '
      f'`data/enwik8/train.bin` (first 90M bytes): it lines up with section 5, except that jevzip reports real coded bytes '
      f'(coder and framing overhead included) with a 512-byte context and no cross-segment memory. Nearest published sizes: '
      f'Al-Rfou T12 (44M, context 512) 1.11, Transformer-XL 12L (41M) 1.06, Adaptive-Span 12L (39M) 1.02, Longformer small (41M) 1.00. '
      f'jevzip-lanes-11m has {p / 1e6:.2f}M parameters, about 4x fewer.')
    w('- **Whole-file enwik8 / enwik9** with that model is not comparable to LTCB, Hutter, cmix or nncp: 90% of enwik8 is its '
      'training data (the same caveat as fractal-zip above), and its weights are not counted, while every LTCB "SFX" entry counts them.')
    w(f'- If the weights were counted (parameter count rounded to {p / 1e6:.2f}M): x 4 bytes (fp32) = about {4 * p:,} bytes, adding 8 x {4 * p:,} / 10^8 = '
      f'{32 * p / 1e8:.3f} bpb on enwik8 and {32 * p / 1e9:.3f} bpb on enwik9; at 2 bytes/parameter (Deletang et al.\'s convention) '
      f'{16 * p / 1e8:.3f} and {16 * p / 1e9:.3f}. For scale, fx2-cmix-transformer ships its 6M parameters in about 2.9 MB (4-bit).')
    w('- The closest published analogue (pretrained LM + arithmetic coding, model not counted) is ts_zip: 1.1061 bpb on enwik8, '
      '1.0835 on enwik9, with a 169M RWKV trained on general text rather than on enwik.')
    w('- The Deletang et al. numbers only apply if jevzip codes independent 2048-byte chunks. No published numbers exist for the '
      'Python-source test sets in `data/code/`: baselines there (xz, zstd, brotli, cmix, paq8px) have to be measured locally.\n')

    w('## 8. Not found / unverified\n')
    unver = [r for r in recs if r.get('verified') is not True]
    w('- zstd `--ultra -22 --long` on enwik8/enwik9: no published number found (LTCB stops at zstd 0.6.0). Measure locally.')
    w('- PPMd var. I as a standalone program: not on LTCB or Silesia; only WinRAR\'s var.-I-based PPM mode (enwik9 198,454,545 SFX).')
    w('- nncp v3.2 parameter count: not stated (LTCB says v3 is ~199M; v3.2 uses "a larger transformer model").')
    w('- Compressive Transformer parameter count: not stated in the paper; 277M comes from Longformer Table 3 (secondary).')
    w('- GPT-2 rows were read from the PDF (pdftotext -layout, page 5), not re-checked by the script.')
    for r in unver:
        w(f"- `{r['id']}`: verified = {r['verified']} ({r.get('notes') or r.get('verification')})")
    if ck.failures:
        w('\n**Check failures in this build:** ' + '; '.join(ck.failures))
    w('\n## Sources (accessed ' + ACCESS_DATE + ')\n')
    for k, (u, c, mode) in SOURCES.items():
        sha = ck.sha.get(k)
        w(f"- {k}: <{c or u}>" + (f" (checked via {u})" if c and u else '') +
          (f", sha256 of fetched body `{sha}`" if sha else (f', PDF sha256 `{GPT2_PDF_SHA256}`' if k == 'gpt2' else '')))
    return '\n'.join(L) + '\n'


def main():
    offline = '--offline' in sys.argv
    recs, ck = build(offline)
    try:
        curl = subprocess.run(['curl', '--version'], capture_output=True, text=True).stdout.splitlines()[0]
    except OSError:
        curl = None
    meta = dict(
        access_date=ACCESS_DATE, generated_by='results/published/build_sota.py',
        command='.venv/bin/python -I results/published/build_sota.py' + (' --offline' if offline else ''),
        exploration_commands=['curl -sS URL | sed -E "s/<[^>]+>//g" | grep/sed (read-only, nothing saved)',
                              'WebFetch of the GPT-2 PDF, then pdftotext -f 5 -l 5 -layout'],
        tool_versions=dict(python=platform.python_version(), platform=platform.platform(), curl=curl),
        definitions=dict(
            bpb='8 * size_bytes / original_size, recomputed here; bpb_total uses total_bytes (data + decompressor)',
            counts_decompressor='true when size_bytes already includes the decompressor (self-extracting archive, Hutter total)',
            verified='true = read from the primary source on access_date AND re-found on the quoted row by this script; '
                     'false = secondary source or check failed; null = offline build (not re-checked)'),
        sources={k: dict(url=c or u, fetched=u, sha256=ck.sha.get(k) or (GPT2_PDF_SHA256 if k == 'gpt2' else None))
                 for k, (u, c, _) in SOURCES.items()},
        check_failures=ck.failures,
    )
    (OUT / 'sota.json').write_text(json.dumps(dict(meta=meta, records=recs), indent=1, ensure_ascii=False) + '\n')
    (OUT / 'sota.md').write_text(md(recs, ck))
    n_ok = sum(r.get('verified') is True for r in recs)
    print(f'{len(recs)} records, {n_ok} verified, {len(ck.failures)} check failures')
    for f in ck.failures:
        print('FAIL', f)
    return 1 if ck.failures else 0


if __name__ == '__main__':
    sys.exit(main())

#!/bin/sh
# Build the strong competitor compressors that jevzip.baselines compares against:
# paq8px and cmix from source at pinned commits, and 7-Zip (7zz, for PPMd) from Homebrew.
# Idempotent: existing clones and binaries are reused; a clone at the wrong commit is moved to it.
#
#     tools/build_competitors.sh            # build what is missing, round-trip each tool, print versions
#     FORCE=1 tools/build_competitors.sh    # rebuild the binaries
#
# Layout (all under downloads/, which git ignores):
#     downloads/paq8px        clone            downloads/paq8px-build/paq8px   binary (cmake Release)
#     downloads/cmix          clone            downloads/cmix-build/cmix       binary + dictionary/english.dic
# Both binaries are built for this CPU (cmix -mcpu=native; paq8px's own flags), so their
# archives are only guaranteed to decode with the same binary.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DL="$ROOT/downloads"
JOBS=${JOBS:-4}
PAQ8PX_URL=https://github.com/hxim/paq8px
PAQ8PX_SHA=c84f576fc2c522194cd320743708652a154daf6b  # paq8px v217, 2026-09-28
CMIX_URL=https://github.com/byronknoll/cmix
CMIX_SHA=194af9cd133b8b741d2a53afca13ed0ca453c276    # cmix v21, 2026-09-26

clone() {  # URL SHA DIR: clone once, then make sure DIR is at SHA
    [ -d "$3/.git" ] || git clone --quiet "$1" "$3"
    if [ "$(git -C "$3" rev-parse HEAD)" != "$2" ]; then
        git -C "$3" fetch --quiet origin "$2"
        git -C "$3" -c advice.detachedHead=false checkout --quiet "$2"
    fi
    echo "$(basename "$3"): $(git -C "$3" rev-parse HEAD)"
}

roundtrip() {  # NAME: compress a small file with the named tool, decompress, compare
    t=$(mktemp -d)
    head -c 20000 "$ROOT/data/code/pydoc_topics.txt" > "$t/in"
    case $1 in
        paq8px) "$DL/paq8px-build/paq8px" -8 "$t/in" "$t/c" > /dev/null && "$DL/paq8px-build/paq8px" -d "$t/c" "$t/out" > /dev/null ;;
        cmix)   d="$DL/cmix-build/dictionary/english.dic"
                "$DL/cmix-build/cmix" -c "$d" "$t/in" "$t/c" > /dev/null 2>&1 && "$DL/cmix-build/cmix" -d "$d" "$t/c" "$t/out" > /dev/null 2>&1 ;;
        7zz)    7zz a -bso0 -bsp0 -t7z -m0=PPMd:mem=64m:o=16 "$t/c.7z" "$t/in" && 7zz e -bso0 -bsp0 -so "$t/c.7z" > "$t/out" ;;
    esac
    if cmp -s "$t/in" "$t/out"; then
        echo "$1: round-trip OK (20000 -> $(cat "$t"/c* | wc -c | tr -d ' ') bytes)"
    else
        echo "$1: ROUND-TRIP FAILED" >&2
        exit 1
    fi
    rm -rf "$t"
}

mkdir -p "$DL"

# --- paq8px: cmake Release with Apple clang (Homebrew GCC 15 cannot parse the macOS 26 SDK headers).
# Two portability shims, both generated in the build dir so the clone stays pristine:
#  * clang-filter, a compiler launcher that drops the GCC-only -floop-strip-mine (clang rejects it);
#  * darwin-size_t.hpp, force-included: on macOS size_t != uint64_t, which makes paq8px's
#    min(uint32_t, size_t) calls ambiguous; it resolves them to the 64-bit min they get on Linux.
# CMakeLists only knows "aarch64", so Apple's "arm64" gets the generic -O3 flags (a configure
# warning); paq8px picks NEON at run time.
clone "$PAQ8PX_URL" "$PAQ8PX_SHA" "$DL/paq8px"
if [ -n "${FORCE:-}" ] || [ ! -x "$DL/paq8px-build/paq8px" ]; then
    B="$DL/paq8px-build"
    rm -rf "$B"
    mkdir -p "$B"
    cat > "$B/clang-filter" <<'EOS'
#!/bin/sh
# CMAKE_<LANG>_COMPILER_LAUNCHER: run the compiler without GCC-only flags that clang rejects.
c=$1; shift
n=$#
while [ "$n" -gt 0 ]; do
    a=$1; shift; n=$((n - 1))
    [ "$a" = -floop-strip-mine ] || set -- "$@" "$a"
done
exec "$c" "$@"
EOS
    chmod +x "$B/clang-filter"
    cat > "$B/darwin-size_t.hpp" <<'EOS'
// Forced include for building paq8px on macOS. There size_t is unsigned long while uint64_t is
// unsigned long long, so calls like min(uint32_t, size_t) are ambiguous between paq8px's
// min(int, int) and min(uint64_t, uint64_t). On Linux size_t is uint64_t and such calls take the
// 64-bit unsigned overload; this template does the same here. It only matches unsigned pairs that
// involve unsigned long, so every call that already compiled keeps its overload.
#pragma once
#if __cplusplus >= 201703L  // CMake's compiler probe compiles in the default (pre-17) mode
#include <type_traits>
template <class A, class B, std::enable_if_t<std::is_unsigned_v<A> && std::is_unsigned_v<B> &&
                                             (std::is_same_v<A, unsigned long> || std::is_same_v<B, unsigned long>), int> = 0>
inline unsigned long long min(A a, B b) { return a < b ? a : b; }
#endif
EOS
    cmake -S "$DL/paq8px" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER="$B/clang-filter" -DCMAKE_CXX_COMPILER_LAUNCHER="$B/clang-filter" \
        "-DCMAKE_CXX_FLAGS=-include $B/darwin-size_t.hpp" > "$DL/paq8px-build.log" 2>&1
    cmake --build "$B" -j "$JOBS" >> "$DL/paq8px-build.log" 2>&1
    echo "paq8px $PAQ8PX_SHA, cmake Release, $(clang++ --version | head -1)" > "$B/VERSION"
fi

# --- cmix: its makefile wants clang++-17 and -march=native; Apple clang takes -mcpu=native instead.
# Same defines and source list as the makefile's "cmix" target, built outside the clone.
clone "$CMIX_URL" "$CMIX_SHA" "$DL/cmix"
if [ -n "${FORCE:-}" ] || [ ! -x "$DL/cmix-build/cmix" ]; then
    mkdir -p "$DL/cmix-build/dictionary"
    cp "$DL/cmix/dictionary/english.dic" "$DL/cmix-build/dictionary/"
    clang++ -std=c++17 -Wall -DP1_MODE=2 -DP3_BATCH -DP2_FASTMATH -DLSTM_QUANT=8 -Ofast -mcpu=native -DNDEBUG \
        $(find "$DL/cmix/src" -name '*.cpp' ! -path '*/enwik9-preproc/*' | sort) -o "$DL/cmix-build/cmix" > "$DL/cmix-build.log" 2>&1
    echo "cmix $CMIX_SHA, -Ofast -mcpu=native -DP1_MODE=2 -DP3_BATCH -DP2_FASTMATH -DLSTM_QUANT=8, $(clang++ --version | head -1)" > "$DL/cmix-build/VERSION"
fi

# --- 7-Zip: 7zz (PPMd lives only in the 7z container)
# brew can fail (e.g. a Homebrew prefix owned by another user): then 7zz rows are skipped.
command -v 7zz > /dev/null || brew install sevenzip || echo "7zz: brew install sevenzip failed; PPMd baselines will be skipped" >&2

echo "--- versions"
"$DL/paq8px-build/paq8px" 2>&1 | grep -m1 -i 'paq8px v' || true
"$DL/cmix-build/cmix" 2>&1 | grep -m1 'cmix version' || true
7zz i 2>/dev/null | grep -m1 '7-Zip' || true
zstd --version; brotli --version; xz --version | head -1
echo "--- round-trips"
roundtrip paq8px
roundtrip cmix
if command -v 7zz > /dev/null; then roundtrip 7zz; else echo "7zz: not installed, skipped"; fi

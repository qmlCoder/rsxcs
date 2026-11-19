#!/usr/bin/env bash
# Transpile libxc 7.1.2 (C) to Rust with c2rust.
#
# Requirements:
#   * cmake, a C compiler, curl/tar
#   * c2rust 0.22.1 and the nightly-2023-04-15 toolchain
#     (c2rust's generated code uses `extern_types`/`raw_ref_op`)
#
# Output:
#   c2rust/libxc/   complete mechanical translation (~53 MB of Rust).
#                   It is gitignored; the safe wrapper in
#                   c2rust/libxc-wrapper/ depends on it.
#
# Only three mechanical fixes are needed after transpilation:
#   1. c2rust emits `f128::f128::new(X).to_f64().unwrap()` for decimal
#      constants that fit in f64; `.to_f64()` is not const, so replace the
#      expression with the literal X.
#   2. `(-f128::f128::new(X)).to_f64()` must become `(X)` as well.
#   3. Rename the generated package to `libxc_c2rust`.
#
# Usage:
#   ./c2rust/generate.sh [path/to/libxc-7.1.2]
#
# Without an argument the libxc 7.1.2 release tarball is downloaded.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-/tmp/libxc-c2rust-build}"
LIBXC_VERSION=7.1.2
SRC_ARCHIVE="libxc-${LIBXC_VERSION}.tar.gz"
URL="https://gitlab.com/libxc/libxc/-/archive/${LIBXC_VERSION}/${SRC_ARCHIVE}"

mkdir -p "$WORK"

if [[ $# -ge 1 ]]; then
    LIBXC_SRC="$(cd "$1" && pwd)"
else
    if [[ ! -f "$WORK/$SRC_ARCHIVE" ]]; then
        echo "==> downloading libxc ${LIBXC_VERSION}"
        curl -L --fail -o "$WORK/$SRC_ARCHIVE" "$URL"
    fi
    if [[ ! -d "$WORK/libxc-${LIBXC_VERSION}" ]]; then
        tar -xzf "$WORK/$SRC_ARCHIVE" -C "$WORK"
    fi
    LIBXC_SRC="$WORK/libxc-${LIBXC_VERSION}"
fi

echo "==> building libxc ${LIBXC_VERSION} (to obtain compile_commands.json)"
cmake -S "$LIBXC_SRC" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_FORTRAN=OFF \
    -DENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON
cmake --build "$WORK/build" -j"$(nproc)"

echo "==> preparing a compilation database restricted to src/"
mkdir -p "$WORK/c2rust-input"
python3 - "$WORK/build/compile_commands.json" "$WORK/c2rust-input/compile_commands.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
db = json.load(open(src))
db = [e for e in db if "/src/" in e["file"]]
json.dump(db, open(dst, "w"), indent=1)
print(f"    {len(db)} translation units")
PY

echo "==> c2rust transpile"
rm -rf "$HERE/libxc"
cd "$WORK/c2rust-input"
c2rust transpile compile_commands.json -o "$HERE/libxc" -e

# c2rust occasionally writes an empty file for one translation unit when
# transpiling the whole database.  Re-transpile those individually.
python3 - "$WORK/c2rust-input/compile_commands.json" "$HERE/libxc" <<'PY'
import glob, json, os, shutil, subprocess, sys, tempfile

db_path, root = sys.argv[1], sys.argv[2]
db = json.load(open(db_path))
by_stem = {os.path.splitext(os.path.basename(e["file"]))[0]: e for e in db}
fixed = 0
for rs in glob.glob(os.path.join(root, "src", "*.rs")):
    if os.path.getsize(rs) != 0:
        continue
    stem = os.path.splitext(os.path.basename(rs))[0]
    with tempfile.TemporaryDirectory() as td:
        json.dump([by_stem[stem]], open(os.path.join(td, "compile_commands.json"), "w"))
        subprocess.run(
            ["c2rust", "transpile", "compile_commands.json", "-o", "out", "-e"],
            cwd=td, check=True,
        )
        shutil.copyfile(os.path.join(td, "out", "src", stem + ".rs"), rs)
    fixed += 1
print(f"    re-transpiled {fixed} empty module(s)")
PY

echo "==> applying post-transpile fixes"
python3 - "$HERE/libxc" <<'PY'
import glob, os, re, sys

root = sys.argv[1]
total = 0
# Handle `::f128::f128::new(X).to_f64().unwrap()` and the parenthesized
# `(-::f128::f128::new(X)).to_f64().unwrap()` form, possibly multi-line.
paren_pat = re.compile(
    r"\(\s*(?P<sign>-?)\s*::f128::f128::new\((?P<num>[^()]*)\)\s*\)"
    r"\s*\.to_f64\(\)\s*\.unwrap\(\)"
)
plain_pat = re.compile(
    r"::f128::f128::new\((?P<num>[^()]*)\)"
    r"\s*\.to_f64\(\)\s*\.unwrap\(\)"
)
for path in glob.glob(os.path.join(root, "src", "*.rs")):
    text = open(path).read()
    text, n1 = paren_pat.subn(
        lambda m: "(" + m.group("sign") + m.group("num") + ")", text
    )
    text, n2 = plain_pat.subn(lambda m: "(" + m.group("num") + ")", text)
    n = n1 + n2
    if n:
        open(path, "w").write(text)
        total += n
print(f"    replaced {total} f128 constant expressions")

cargo = os.path.join(root, "Cargo.toml")
text = open(cargo).read()
text = text.replace('name = "libxc"', 'name = "libxc_c2rust"')
open(cargo, "w").write(text)
print("    renamed package to libxc_c2rust")
PY

rm -rf "$HERE/libxc/target"

echo
echo "==> done: $HERE/libxc"
echo "    build the safe wrapper with:"
echo "      cd $HERE/libxc-wrapper"
echo "      cargo run --example compare     # uses c2rust/rust-toolchain.toml"

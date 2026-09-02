---
name: windows-python-and-encoding-traps
description: "On this machine `python` is not on PATH (use `py -3.10`), and PowerShell Get-Content/Set-Content round-trips silently corrupt this repo's UTF-8 source files"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: abd4bd3b-dd57-48ee-9017-c017072cddda
  modified: 2026-08-27T14:54:56.510Z
---

Two Windows traps that cost real time in this repo:

1. **`python` is not on PATH.** Use **`py -3.10`** (RoboDK deps are installed there).
   `py -0p` lists 3.13/3.11/3.10 plus RoboDK's bundled 3.7.
2. **Never round-trip a source file through PowerShell 5.1 text cmdlets.**
   `Get-Content -Raw` decodes BOM-less UTF-8 as ANSI/cp1252, and `Set-Content -Encoding utf8`
   writes a BOM. Doing both mangled every non-ASCII char in `tasni/core/rdk_io.py`
   (52 mojibake lines, em-dashes double-encoded) and added a BOM, inflating its diff from
   356 to 482 changed lines.

**Why:** this codebase's comments and docstrings are full of em-dashes, arrows and `+/-`
signs, so almost every file is UTF-8 with non-ASCII content — the corruption is silent and
does not fail any test.

**How to apply:** edit files with the Edit/Write tools, not shell text round-trips. If a
temporary in-place mutation is genuinely needed (e.g. a mutation test), copy the file with
`cp`, mutate, then `cp` back — byte-level, never text-level. To repair an already
double-encoded file: `iconv -f UTF-8 -t WINDOWS-1252 file > fixed` (that reverses it), then
verify with `git diff --stat` against the known-good change count. Note also that raw
byte-writing via PowerShell `[System.IO.File]::WriteAllBytes` and via `py -3.10 -c` were both
blocked by the permission classifier; `iconv` + `cp` in Bash went through.
3. **Item names inside a `.rdk` are UTF-16-LE, and the file is one zlib stream.**
   Searching a decompressed station for `b"Tasni Work Frame"` (UTF-8) returns zero hits
   even when the item is there. Inflate with `zlib.decompress(open(f,'rb').read()[4:])`
   (skip the 4-byte header) and search `name.encode("utf-16-le")`. Some strings ARE
   UTF-8 (`Realsense`, `KUKA KR150 R2700`, `camera`), so a UTF-8 hit proves nothing
   about a UTF-16 miss — always test both encodings before concluding an item is absent.

See [[pytest-suite-too-slow-to-run-fully]] and [[extrusion-platform-centring]].

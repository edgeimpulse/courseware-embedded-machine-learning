#!/usr/bin/env python3
"""
pptx_to_pdf.py — Batch-convert PowerPoint decks to PDF using LibreOffice.

Walks a directory tree, converts every .pptx to .pdf, and mirrors the source
folder structure. Designed for repeated runs against a large repository:

  * runs conversions in parallel, giving each worker its own LibreOffice
    profile (without this, concurrent soffice processes collide and silently
    convert nothing)
  * skips decks whose PDF is already newer than the source, so re-runs are cheap
  * enforces a per-file timeout, because LibreOffice occasionally hangs
  * verifies each output actually exists and is non-empty — soffice frequently
    exits 0 having produced no file
  * reports failures at the end instead of dying on the first one

Usage:
    python3 pptx_to_pdf.py .
    python3 pptx_to_pdf.py . --outdir build/pdf --jobs 4
    python3 pptx_to_pdf.py . --force

Requires: LibreOffice (provides the `soffice` binary).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

DEFAULT_TIMEOUT = 300  # seconds per deck

# Common install locations, checked when `soffice` is not on PATH.
CANDIDATES = [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",          # macOS
    r"C:\Program Files\LibreOffice\program\soffice.exe",             # Windows
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/snap/bin/libreoffice",
]

_print_lock = threading.Lock()
_local = threading.local()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def find_soffice(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        sys.exit(f"soffice not found at: {explicit}")

    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found

    for path in CANDIDATES:
        if Path(path).exists():
            return path

    sys.exit(
        "LibreOffice not found.\n"
        "  macOS:   brew install --cask libreoffice\n"
        "  Ubuntu:  sudo apt install libreoffice-impress\n"
        "  Windows: https://www.libreoffice.org/download/\n"
        "Or pass the binary explicitly with --soffice /path/to/soffice"
    )


def worker_profile(base: Path) -> str:
    """
    Return a LibreOffice user-profile URI unique to this thread.

    LibreOffice serializes on its profile directory. Without a distinct profile
    per worker, parallel invocations either block or exit 0 without writing a
    file — the most common cause of a batch that "succeeds" but produces
    nothing.
    """
    if not hasattr(_local, "profile"):
        d = base / f"profile_{threading.get_ident()}"
        d.mkdir(parents=True, exist_ok=True)
        _local.profile = d.as_uri()
    return _local.profile


def convert_one(src: Path, dest_dir: Path, soffice: str, profile_base: Path,
                timeout: int) -> tuple[Path, bool, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = dest_dir / (src.stem + ".pdf")

    cmd = [
        soffice,
        f"-env:UserInstallation={worker_profile(profile_base)}",
        "--headless",
        "--norestore",
        "--invisible",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        "--convert-to", "pdf:impress_pdf_Export",
        "--outdir", str(dest_dir),
        str(src),
    ]

    env = os.environ.copy()
    env["SAL_USE_VCLPLUGIN"] = "svp"   # force headless backend on Linux

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return expected, False, f"timed out after {timeout}s"

    # soffice exits 0 even when it writes nothing, so check the artifact.
    if not expected.exists() or expected.stat().st_size == 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return expected, False, (detail[-1] if detail else "no output produced")

    return expected, True, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch-convert .pptx to .pdf with LibreOffice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("root", nargs="?", default=".",
                    help="Directory to scan for .pptx files (default: current)")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Write PDFs into this directory, mirroring the source "
                         "tree. Default: alongside each .pptx.")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="Parallel conversions (default: CPU count, max 8). "
                         "Use 1 if you hit instability.")
    ap.add_argument("-f", "--force", action="store_true",
                    help="Reconvert even if the PDF is newer than the .pptx.")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Per-file timeout in seconds (default: {DEFAULT_TIMEOUT})")
    ap.add_argument("--soffice", default=None,
                    help="Path to the soffice binary, if not auto-detected.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Substring to skip; repeatable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be converted, then exit.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        return print(f"Not a directory: {root}") or 1

    soffice = find_soffice(args.soffice)
    outdir = Path(args.outdir).resolve() if args.outdir else None

    decks = sorted(
        p for p in root.rglob("*.pptx")
        if not p.name.startswith("~$")
        and not any(x in str(p) for x in args.exclude)
    )
    if not decks:
        return print(f"No .pptx files found under {root}") or 1

    todo: list[tuple[Path, Path]] = []
    skipped = 0
    for src in decks:
        rel = src.relative_to(root)
        dest_dir = (outdir / rel.parent) if outdir else src.parent
        pdf = dest_dir / (src.stem + ".pdf")
        if pdf.exists() and not args.force and \
                pdf.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue
        todo.append((src, dest_dir))

    print(f"LibreOffice: {soffice}")
    print(f"Found {len(decks)} deck(s); {len(todo)} to convert, "
          f"{skipped} already up to date.\n")

    if args.dry_run:
        for src, dest in todo:
            print(f"  {src.relative_to(root)}  ->  {dest}")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    jobs = args.jobs or min(8, os.cpu_count() or 4)
    if platform.system() == "Windows":
        jobs = min(jobs, 4)   # Windows soffice is less tolerant of concurrency

    failures: list[tuple[Path, str]] = []
    done = 0

    with tempfile.TemporaryDirectory(prefix="lo_profiles_") as tmp:
        base = Path(tmp)
        with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            future_map = {
                pool.submit(convert_one, src, dest, soffice, base, args.timeout): src
                for src, dest in todo
            }
            for fut in futures.as_completed(future_map):
                src = future_map[fut]
                done += 1
                try:
                    _, ok, err = fut.result()
                except Exception as exc:
                    ok, err = False, f"{type(exc).__name__}: {exc}"
                rel = src.relative_to(root)
                if ok:
                    log(f"[{done:>3}/{len(todo)}] ok    {rel}")
                else:
                    failures.append((rel, err))
                    log(f"[{done:>3}/{len(todo)}] FAIL  {rel}  ({err})")

    print(f"\nConverted {len(todo) - len(failures)}, "
          f"skipped {skipped}, failed {len(failures)}.")
    if failures:
        print("\nFailures:")
        for rel, err in failures:
            print(f"  {rel}\n    {err}")
        print("\nRetry just these with --jobs 1 for clearer errors.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

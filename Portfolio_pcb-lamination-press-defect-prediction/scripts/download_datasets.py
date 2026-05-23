"""Dataset download helper for the semiconductor PCB lamination project

Supports downloading a small set of public datasets used for pretraining and
baseline experiments (SECOM, DeepPCB, Bosch via Kaggle CLI). Designed for
Windows PowerShell environment but works on other OSes with Python 3.8+.

Usage examples (PowerShell):
  python scripts\download_datasets.py --secom
  python scripts\download_datasets.py --deeppcb
  python scripts\download_datasets.py --bosch
  python scripts\download_datasets.py --all

Notes:
 - Kaggle downloads require `kaggle` CLI and a valid kaggle.json at
   %USERPROFILE%\.kaggle\kaggle.json. See https://github.com/Kaggle/kaggle-api
 - Git must be available for cloning DeepPCB; otherwise the script will
   attempt to download the repo archive.
 - Some datasets (e.g., NASA C-MAPSS) may require manual download; the
   script will print instructions when automatic download is not available.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import zipfile

try:
    import requests
except Exception:  # pragma: no cover - requests should be in requirements
    requests = None


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_secom(dest: Path) -> None:
    """Download SECOM dataset (UCI) to dest/secom/"""
    if requests is None:
        print("The 'requests' package is required to download SECOM. Please pip install requests")
        return
    ensure_dir(dest)
    zip_url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
    zip_path = dest / "secom.zip"
    if not zip_path.exists():
        print(f"[SECOM] Downloading {zip_url} -> {zip_path}")
        r = requests.get(zip_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print(f"[SECOM] already exists: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = {"secom.data", "secom_labels.data"}
        present = set(zf.namelist())
        missing = sorted(members - present)
        if missing:
            raise FileNotFoundError(f"SECOM archive missing expected files: {missing}")
        for fname in sorted(members):
            out = dest / fname
            if out.exists():
                print(f"[SECOM] already exists: {out}")
                continue
            print(f"[SECOM] Extracting {fname} -> {out}")
            with zf.open(fname) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)

    legacy_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00397"
    legacy_file = dest / "secom.data"
    if not legacy_file.exists():
        legacy_url = f"{legacy_base}/secom.data"
        print(f"[SECOM] Fallback download attempt {legacy_url} -> {legacy_file}")
        try:
            r = requests.get(legacy_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(legacy_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as ex:
            print(f"[SECOM] fallback direct-file download failed: {ex}")
    print("[SECOM] done")


def download_deeppcb(dest: Path) -> None:
    """Clone or download DeepPCB into dest/deeppcb/"""
    repo = "https://github.com/tangsanli5201/DeepPCB.git"
    ensure_dir(dest.parent)
    if dest.exists():
        print(f"[DeepPCB] already exists: {dest}")
        return
    print(f"[DeepPCB] Cloning {repo} -> {dest}")
    git = shutil.which("git")
    if git:
        subprocess.run([git, "clone", repo, str(dest)], check=False)
        print("[DeepPCB] clone attempted (check output). If cloning failed, rerun with git available.")
    else:
        # fallback: try download zip archive
        archive_url = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
        if requests is None:
            print("requests is required to download DeepPCB zip. Please install requests or git.")
            return
        tmpzip = dest.parent / "deeppcb_master.zip"
        print(f"[DeepPCB] Downloading archive {archive_url} -> {tmpzip}")
        r = requests.get(archive_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(tmpzip, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        try:
            import zipfile

            with zipfile.ZipFile(tmpzip, "r") as z:
                z.extractall(dest.parent)
            # move extracted folder to expected path if needed
            extracted = dest.parent / "DeepPCB-master"
            if extracted.exists():
                extracted.rename(dest)
            print("[DeepPCB] archive downloaded and extracted")
        finally:
            if tmpzip.exists():
                tmpzip.unlink()


def download_bosch_kaggle(dest: Path) -> None:
    """Download Bosch Production Line dataset using kaggle CLI into dest/bosch/"""
    ensure_dir(dest)
    kaggle = shutil.which("kaggle")
    if not kaggle:
        print("[Bosch] kaggle CLI not found. Install kaggle and place kaggle.json under %USERPROFILE%\\.kaggle\\kaggle.json")
        return
    print("[Bosch] Attempting kaggle CLI download (this requires kaggle.json credentials)")
    try:
        subprocess.run([kaggle, "competitions", "download", "-c", "bosch-production-line-performance", "-p", str(dest)], check=True)
        # try to unzip any zip files found
        for z in dest.glob("*.zip"):
            print(f"[Bosch] Extracting {z}")
            try:
                import zipfile

                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(dest)
            except Exception as ex:
                print(f"[Bosch] extraction failed: {ex}")
    except subprocess.CalledProcessError as e:
        print(f"[Bosch] kaggle download failed: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download_datasets.py")
    parser.add_argument("--secom", action="store_true", help="Download SECOM (UCI)")
    parser.add_argument("--deeppcb", action="store_true", help="Download DeepPCB (GitHub)")
    parser.add_argument("--bosch", action="store_true", help="Download Bosch dataset via kaggle CLI (requires kaggle setup)")
    parser.add_argument("--all", action="store_true", help="Download all available datasets that can be fetched automatically")
    args = parser.parse_args(argv)

    secom_dest = DATA_RAW / "secom"
    deeppcb_dest = DATA_RAW / "deeppcb"
    bosch_dest = DATA_RAW / "bosch"

    if args.all or args.secom:
        download_secom(secom_dest)
    if args.all or args.deeppcb:
        download_deeppcb(deeppcb_dest)
    if args.all or args.bosch:
        download_bosch_kaggle(bosch_dest)

    print("Downloads finished (or attempted). Check the printed messages for errors and follow instructions for manual downloads if necessary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Upload a large artifact to Zenodo as a draft deposition.

Usage:
    ZENODO_TOKEN=<pat> uv run python scripts/zenodo_upload.py \\
        --file /mnt/data/artifacts/public/variant-viewer/builds/v5/clean.parquet \\
        --sandbox           # test against sandbox.zenodo.org first
        # ...then re-run without --sandbox against prod

Defaults create a draft (no DOI minted, files mutable). Pass --publish to
finalize and mint the DOI. The token must be supplied via the ZENODO_TOKEN
environment variable — never via a CLI flag (leaks into shell history).

Token creation:
    https://zenodo.org/account/settings/applications/
    https://sandbox.zenodo.org/account/settings/applications/
Scopes required: deposit:write, deposit:actions

Metadata defaults come from a Python dict below. Override any field by
passing a JSON file via --metadata-json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

CONSOLE = Console()

DEFAULT_PART_MB = 100  # Zenodo recommends 5MB–5GB; 100MB → fast PUT, modest part count
DEFAULT_WORKERS = 4    # parallel part uploads; too high risks Zenodo throttling
PART_TIMEOUT = (30, 15 * 60)  # per-part: connect=30s, read=15min (100MB at slow speeds)
MAX_RETRIES = 5


PAPER_DOI = "10.64898/2026.04.10.717844"
PAPER_TITLE = "EVEE: Interpretable variant effect prediction from genomic foundation model embeddings"

PAPER_ABSTRACT = (
    "Predicting the clinical significance of genetic variants remains a central "
    "challenge in genomic medicine, with most observed variants classified as "
    "variants of uncertain significance. Here we show that representations from "
    "Evo 2, a 7-billion-parameter genomic foundation model, support accurate and "
    "interpretable pathogenicity prediction across variant types from a single "
    "framework. An embedding-based classifier, or \"probe\", trained on Evo 2 "
    "embeddings achieves state-of-the-art performance across single nucleotide "
    "variant consequence types (0.997 overall AUROC on 833k ClinVar variants) and "
    "generalizes zero-shot to indels (0.991 AUROC), outperforming bioinformatic "
    "meta-predictors, protein models, and existing foundation model approaches. "
    "Performance is robust across conservation levels and transfers to deep "
    "mutational scanning datasets for BRCA1, BRCA2, TP53, and LDLR. To make these "
    "predictions interpretable, we train supervised annotation probes to quantify "
    "predicted disruptions caused by each variant, then synthesize these "
    "disruption profiles into natural language explanations using a frontier "
    "reasoning model. We provide pre-computed predictions and on-demand "
    "explanations for all 4.2 million ClinVar variants through the Evo Variant "
    "Effect Explorer (EVEE), an interactive web resource for the community."
)

# Author order and affiliations from Crossref (DOI above), v3 posted 2026-04-11.
PAPER_AUTHORS = [
    {"name": "Pearce, Michael T.", "affiliation": "Goodfire"},
    {"name": "Dooms, Thomas", "affiliation": "Goodfire", "orcid": "0009-0001-8534-2450"},
    {"name": "Yamamoto, Ryo", "affiliation": "Goodfire", "orcid": "0000-0003-3134-145X"},
    {"name": "Meehl, Joshua", "affiliation": "Mayo Clinic", "orcid": "0009-0003-4842-7687"},
    {"name": "Molnar, Carl", "affiliation": "Mayo Clinic", "orcid": "0009-0007-0084-4114"},
    {"name": "Bissell, Mark", "affiliation": "Goodfire"},
    {"name": "Hazra, Dron", "affiliation": "Goodfire", "orcid": "0009-0001-6362-6639"},
    {"name": "Fang, Ching", "affiliation": "Goodfire"},
    {"name": "Nguyen, Nam", "affiliation": "Goodfire"},
    {"name": "Anderson, Michael", "affiliation": "Goodfire"},
    {"name": "Osborne, Collin", "affiliation": "Mayo Clinic", "orcid": "0000-0001-5165-6729"},
    {"name": "Duffy, Patrick", "affiliation": "Mayo Clinic"},
    {"name": "Toomey, Bridget", "affiliation": "Mayo Clinic", "orcid": "0009-0009-6116-6227"},
    {"name": "Klee, Eric", "affiliation": "Mayo Clinic", "orcid": "0000-0003-2946-5795"},
    {"name": "Myasoedova, Elena", "affiliation": "Mayo Clinic", "orcid": "0000-0003-2006-1436"},
    {"name": "Ryu, Alexander J.", "affiliation": "Mayo Clinic", "orcid": "0000-0002-0138-5112"},
    {"name": "Ayanian, Shant", "affiliation": "Mayo Clinic", "orcid": "0000-0001-9319-9001"},
    {"name": "Korfiatis, Panos", "affiliation": "Mayo Clinic", "orcid": "0000-0003-2516-1751"},
    {"name": "Redlon, Matt", "affiliation": "Mayo Clinic", "orcid": "0009-0004-3445-2053"},
    {"name": "Jain, Archa", "affiliation": "Goodfire"},
    {"name": "Balsam, Daniel", "affiliation": "Goodfire"},
    {"name": "Wang, Nicholas K.", "affiliation": "Goodfire", "orcid": "0000-0003-1043-3072"},
]


def default_metadata(title: str, version: str, description: str) -> dict:
    return {
        "upload_type": "dataset",
        "title": title,
        "description": description,
        "version": version,
        "license": "cc-by-4.0",  # matches the bioRxiv preprint license
        "access_right": "open",
        "creators": PAPER_AUTHORS,
        "keywords": [
            "variant effect prediction",
            "ClinVar",
            "pathogenicity",
            "Evo 2",
            "genomic foundation model",
            "interpretability",
            "probing classifier",
            "EVEE",
        ],
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": PAPER_DOI,
                "resource_type": "publication-preprint",
                "scheme": "doi",
            },
        ],
        "notes": (
            "Per-variant flat table (builds/v5/clean.parquet) used to populate the "
            "EVEE web app (https://evee.goodfire.ai). One row per ClinVar variant; "
            "columns include variant_id, gene_name, consequence, ClinVar significance "
            "and label, an Evo 2 probe pathogenicity score, and ~4,900 additional "
            "probe heads (disruption, effect, and annotation categories). The "
            "variants.duckdb artifact served by the website is derived from this "
            "parquet via scripts in https://github.com/goodfire-ai/variant-viewer."
        ),
    }


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def create_deposition(base: str, token: str) -> dict:
    r = requests.post(
        f"{base}/api/deposit/depositions",
        json={},
        headers=_auth(token),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def get_deposition(base: str, token: str, dep_id: int) -> dict:
    r = requests.get(
        f"{base}/api/deposit/depositions/{dep_id}",
        headers=_auth(token),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


class _ProgressReader:
    """File-like wrapper that exposes __len__ (so requests sends Content-Length,
    never Transfer-Encoding: chunked) and reports read progress.

    Zenodo's bucket endpoint rejects chunked-encoded PUTs >~1GB with 400.
    """

    def __init__(self, path: Path, progress: Progress, task_id: int):
        self._f = open(path, "rb")
        self._size = path.stat().st_size
        self._progress = progress
        self._task = task_id

    def __len__(self) -> int:
        return self._size

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        if data:
            self._progress.update(self._task, advance=len(data))
        return data

    def close(self) -> None:
        self._f.close()


def upload_one_file(bucket_url: str, token: str, file_path: Path) -> dict:
    """Streamed single PUT into a Zenodo deposit bucket. Retries on transient failures."""
    size = file_path.stat().st_size
    key = file_path.name
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with Progress(
                TextColumn("[bold]{task.fields[name]}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=CONSOLE,
            ) as progress:
                task = progress.add_task("upload", total=size, name=key)
                reader = _ProgressReader(file_path, progress, task)
                try:
                    r = requests.put(
                        f"{bucket_url}/{key}",
                        data=reader,
                        headers={**_auth(token), "Content-Type": "application/octet-stream"},
                        timeout=(30, 60 * 60 * 2),  # connect 30s, read 2h
                    )
                finally:
                    reader.close()
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ConnectionError, OSError) as e:
            last_err = e
            backoff = min(60, 2**attempt)
            CONSOLE.print(
                f"[yellow]{key}: attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {backoff}s...[/yellow]"
            )
            time.sleep(backoff)
    raise RuntimeError(f"{key}: upload failed after {MAX_RETRIES} attempts: {last_err}")


def already_uploaded(base: str, token: str, dep_id: int) -> dict[str, int]:
    """Return {filename: filesize} of files already attached to the draft."""
    dep = get_deposition(base, token, dep_id)
    return {f["filename"]: f["filesize"] for f in dep.get("files", [])}


def put_metadata(base: str, token: str, dep_id: int, metadata: dict) -> dict:
    r = requests.put(
        f"{base}/api/deposit/depositions/{dep_id}",
        json={"metadata": metadata},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if not r.ok:
        CONSOLE.print(f"[red]Metadata response: {r.status_code}\n{r.text}[/red]")
        r.raise_for_status()
    return r.json()


def publish(base: str, token: str, dep_id: int) -> dict:
    r = requests.post(
        f"{base}/api/deposit/depositions/{dep_id}/actions/publish",
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    if not r.ok:
        CONSOLE.print(f"[red]Publish response: {r.status_code}\n{r.text}[/red]")
        r.raise_for_status()
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="Single file to upload.")
    src.add_argument("--files-dir", type=Path,
                     help="Directory containing files to upload as one deposition.")
    p.add_argument("--glob", default="*.parquet",
                   help="When --files-dir is used, match files with this glob. Default: *.parquet.")
    p.add_argument("--include-manifest", action="store_true",
                   help="When --files-dir is used, also upload manifest.json if present in the dir.")
    p.add_argument("--title", default="EVEE variant table (v5)",
                   help="Zenodo title. Default: 'EVEE variant table (v5)'.")
    p.add_argument("--version", default="v5", help="Version string. Default: v5.")
    p.add_argument(
        "--description-file",
        type=Path,
        default=Path(__file__).parent.parent / "README.md",
        help="Markdown file whose contents become the description. "
             "Default: evee-manuscript/README.md.",
    )
    p.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional JSON file to override/merge into default metadata dict. "
             "Top-level keys replace defaults; nested dicts/lists are not merged.",
    )
    p.add_argument("--sandbox", action="store_true",
                   help="Use sandbox.zenodo.org (testing, fake DOIs).")
    p.add_argument("--publish", action="store_true",
                   help="Publish immediately after upload. Default: leave as draft.")
    p.add_argument("--deposition-id", type=int, default=None,
                   help="Reuse an existing draft deposition instead of creating a new one.")
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("error: ZENODO_TOKEN env var not set")
    if not args.description_file.is_file():
        sys.exit(f"error: description file not found: {args.description_file}")

    # Resolve file list
    files: list[Path]
    if args.file:
        if not args.file.is_file():
            sys.exit(f"error: file not found: {args.file}")
        files = [args.file]
    else:
        if not args.files_dir.is_dir():
            sys.exit(f"error: dir not found: {args.files_dir}")
        files = sorted(args.files_dir.glob(args.glob))
        if args.include_manifest:
            m = args.files_dir / "manifest.json"
            if m.is_file():
                files.append(m)
        if not files:
            sys.exit(f"error: no files matched {args.glob} in {args.files_dir}")

    total_gb = sum(f.stat().st_size for f in files) / 1e9
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    CONSOLE.rule(f"Zenodo upload → {base}")
    CONSOLE.print(f"Files:   {len(files)} ({total_gb:.1f} GB total)")
    for f in files:
        CONSOLE.print(f"  - {f.name} ({f.stat().st_size / 1e9:.2f} GB)")
    CONSOLE.print(f"Title:   {args.title}")
    CONSOLE.print(f"Publish: {'yes' if args.publish else 'draft only'}")

    description = args.description_file.read_text()
    metadata = default_metadata(args.title, args.version, description)
    if args.metadata_json:
        override = json.loads(args.metadata_json.read_text())
        metadata.update(override)

    if args.deposition_id:
        CONSOLE.rule(f"1/3  reuse draft {args.deposition_id}")
        dep = get_deposition(base, token, args.deposition_id)
        if dep.get("submitted"):
            sys.exit(f"error: deposition {args.deposition_id} is already submitted; cannot modify.")
    else:
        CONSOLE.rule("1/3  create deposition")
        dep = create_deposition(base, token)
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    html = dep["links"].get("html") or dep["links"].get("latest_draft_html")
    CONSOLE.print(f"Draft id: {dep_id}")
    CONSOLE.print(f"URL:      {html}")

    CONSOLE.rule(f"2/3  upload {len(files)} file(s)")
    existing = already_uploaded(base, token, dep_id)
    for i, f in enumerate(files, start=1):
        if f.name in existing and existing[f.name] == f.stat().st_size:
            CONSOLE.print(f"[dim]{i}/{len(files)} {f.name}: already uploaded, skipping.[/dim]")
            continue
        CONSOLE.print(f"[bold]{i}/{len(files)}[/bold] {f.name}")
        upload_one_file(bucket, token, f)

    CONSOLE.rule("3/3  attach metadata")
    put_metadata(base, token, dep_id, metadata)
    CONSOLE.print("[green]Metadata attached.[/green]")

    if args.publish:
        CONSOLE.rule("publish")
        published = publish(base, token, dep_id)
        doi = published.get("doi") or published.get("conceptdoi")
        CONSOLE.print(f"[green]Published.[/green] DOI: {doi}")
        CONSOLE.print(f"URL: {published['links']['html']}")
    else:
        CONSOLE.print(
            "[yellow]Draft ready for review. Log in, verify metadata, and click "
            "'Publish' — or re-run with --publish.[/yellow]"
        )
        CONSOLE.print(f"URL: {html}")


if __name__ == "__main__":
    main()

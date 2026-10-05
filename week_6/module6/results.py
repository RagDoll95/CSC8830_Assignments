"""Cached results for the bundled examples, shared by the web app and the CLI.

The two sample videos and the bundled four-view photos never change, so their
results are computed once into week_6/module6/output/<name>/ and reused.
Run once after deploying, from the repository root, so even the first click
in the app is instant:
    python week_6/module6/results.py
A folder is only used when its marker matches RESULTS_VERSION; bump the
version when an output format changes and the caches rebuild themselves.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sfm  # noqa: E402
from motion import process_video  # noqa: E402
from validate import flow_summary, validate_pair  # noqa: E402

RESULTS_VERSION = 1
OUTPUT = HERE / 'output'
MARKER = 'complete.json'
SAMPLES = {'penguins': 'penguins.mp4', 'marble_race': 'marble_race.mp4'}
STRUCTURE = 'structure'


def is_complete(folder: Path) -> bool:
    """True if folder holds a finished build of the current version."""
    marker = folder / MARKER
    return marker.is_file() and json.loads(marker.read_text()).get('version') == RESULTS_VERSION


def cached(name: str, build: Callable[[Path], object]) -> Path:
    """Folder of finished results for name, building it first if needed.

    The build runs in a staging folder that is renamed into place only once
    complete, so a failed or interrupted build is never served.
    """
    folder = OUTPUT / name
    if is_complete(folder):
        return folder
    staging = OUTPUT / f'.{name}-{uuid.uuid4().hex}'
    staging.mkdir(parents=True)
    try:
        build(staging)
        (staging / MARKER).write_text(json.dumps({'version': RESULTS_VERSION}))
        if folder.exists():
            shutil.rmtree(folder)
        staging.rename(folder)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return folder


def build_sample(sample: str, folder: Path) -> None:
    process_video(HERE / SAMPLES[sample], folder)
    validate_pair(folder, sample)
    flow_summary(folder)


def sample_results(sample: str) -> Path:
    """Flow video, tracking check and flow statistics for a bundled sample.

    Raises:
        KeyError: if sample is not a bundled video.
    """
    if sample not in SAMPLES:
        raise KeyError(sample)
    return cached(sample, lambda folder: build_sample(sample, folder))


def structure_results() -> Path:
    """Four-view factorization of the bundled checkerboard photos."""
    return cached(STRUCTURE, sfm.run)


if __name__ == '__main__':
    for name in SAMPLES:
        print(name, sample_results(name))
    print(STRUCTURE, structure_results())

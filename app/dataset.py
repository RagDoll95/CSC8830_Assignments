"""Swapping an assignment's dataset for a new one, atomically.

THE SWAP IS ALL-OR-NOTHING
    stage() builds the replacement in a scratch folder and install() swaps it
    in only once the caller is satisfied with it. A calibration upload whose
    corner detection fails therefore leaves the previous dataset exactly as it
    was, instead of deleting it and leaving the page with nothing to show.

WHY DELETION AND NOT A FLAG
    install() removes what it replaces rather than marking it superseded. A
    "hide the old data" flag has to be honoured by every future page that reads
    the folder, and one that forgets puts stale data back on screen. Removing
    the files makes the guarantee structural: there is nothing left to show.
    Everything here is in git, so `git checkout` brings a dataset back.
"""

import os
import shutil
import uuid


def remove(*paths):
    """Delete these exact files or trees.

    For the derived artifacts -- plots, statistics, augmented CSVs -- that were
    computed from a dataset that has since changed. Their names say nothing
    about their provenance, so the caller names them explicitly.
    """
    removed = []
    for path in paths:
        if os.path.exists(path):
            _remove(path)
            removed.append(os.path.basename(path))
    return removed


def stage(scratch_root):
    """A fresh empty directory to build a replacement dataset in."""
    path = os.path.join(scratch_root, "staging-" + uuid.uuid4().hex[:10])
    os.makedirs(path, exist_ok=True)
    return path


def install(staging, target):
    """Replace `target` wholesale with the contents of `staging`.

    The old directory goes, including any earlier upload: an upload is the
    dataset now, not an addition to it. Called only after the caller has
    confirmed the staged data is good.
    """
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(staging, target)
    return target


def discard(staging):
    """Throw away a staged dataset that did not work out."""
    shutil.rmtree(staging, ignore_errors=True)


def _remove(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except OSError:
            pass

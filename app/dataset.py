"""Replacing the synthetic stand-in data with real captures, permanently.

WHY THIS EXISTS
    Each assignment ships synthetic -- rendered, not photographed -- stand-in
    data so the pipeline can be run and scored before any real photo exists.
    The moment a real photo is uploaded that stand-in has to disappear: not be
    hidden, not be sorted below the real thing, but be deleted from disk. A
    rendered checkerboard sitting next to a photographed one on screen during a
    presentation is worse than having no sample data at all.

WHAT "SYNTHETIC" MEANS ON DISK
    tools/make_synthetic_data.py names everything it writes `synthetic_*` and
    drops a README_SYNTHETIC.txt beside it. is_synthetic() is that naming
    convention and nothing else, which is what makes purging safe: a phone
    photo is IMG_4021.jpg and cannot match, so nothing here can delete a real
    capture by accident.

THE SWAP IS ALL-OR-NOTHING
    stage() builds the replacement in a scratch folder and install() swaps it
    in only once the caller is satisfied with it. A calibration upload whose
    corner detection fails therefore leaves the previous dataset exactly as it
    was, instead of deleting it and leaving the page with nothing to show.

WHY DELETION AND NOT A FLAG
    A "hide the synthetic data" flag has to be honoured by every future page
    that reads the folder, and one that forgets puts rendered data back on
    screen. Removing the files makes the guarantee structural: there is nothing
    left to show. The files are in git, so `git checkout` brings them back for
    anyone who wants the scored synthetic run again.
"""

import os
import shutil
import uuid

SYNTHETIC_PREFIX = "synthetic_"
SYNTHETIC_README = "README_SYNTHETIC.txt"


def is_synthetic(name):
    """True for a file written by tools/make_synthetic_data.py."""
    base = os.path.basename(name)
    return base.startswith(SYNTHETIC_PREFIX) or base == SYNTHETIC_README


def purge_synthetic(*paths):
    """Delete the synthetic files in each directory given.

    A path that is a file is deleted if its own name is synthetic; a path that
    is a directory has its synthetic entries deleted and is itself kept.
    Returns the basenames removed, so a page can say what it just threw away
    rather than silently changing the repository underneath the user.
    """
    removed = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if is_synthetic(name):
                    _remove(os.path.join(path, name))
                    removed.append(name)
        elif os.path.isfile(path) and is_synthetic(path):
            _remove(path)
            removed.append(os.path.basename(path))
    return removed


def remove(*paths):
    """Delete these exact files or trees regardless of their names.

    For the derived artifacts -- plots, statistics, augmented CSVs -- that were
    computed *from* synthetic data. Their names say nothing about their
    provenance, so the caller names them explicitly.
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

    The old directory goes, including any synthetic files and any earlier real
    upload: an upload is the dataset now, not an addition to it. Called only
    after the caller has confirmed the staged data is good.
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

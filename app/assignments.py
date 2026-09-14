"""The registry of assignments this web application serves.

WHY THIS FILE EXISTS
    The app used to live at week_2/module2/app, one level inside the very
    assignment it served, so its paths ("../calibration/output") could only
    ever reach Module 2. It now sits at the repository root and reads the list
    below, so every assignment in the repo is reachable from one running
    server and one navigation bar.

ADDING AN ASSIGNMENT
    1. write app/modules/<name>.py exposing `bp` (a Flask Blueprint) and `NAV`
    2. add one Assignment(...) entry below
    Nothing in app/app.py changes: it imports whatever this list names.

A FOLDER THAT IS NOT LISTED HERE IS STILL VISIBLE
    unregistered() globs week_*/module* and returns the ones no entry claims.
    The home page lists them as present-but-not-wired-up rather than silently
    hiding them, so a half-finished week is obvious instead of invisible.
"""

import glob
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)

# Session scratch for every assignment: uploads/<slug>/... Kept under app/ and
# git-ignored (app/.gitignore) because uploads are scratch, not source.
UPLOAD_ROOT = os.path.join(APP_DIR, "uploads")


class Assignment(object):
    """One assignment: where its code lives and how the app mounts it.

    slug      URL prefix and uploads subfolder, e.g. "module-2" -> /module-2/
    path      location of the assignment's own code, relative to the repo root
    module    importable name of the blueprint module under app/
    """

    def __init__(self, slug, number, title, subtitle, path, module,
                 summary="", parts=()):
        self.slug = slug
        self.number = number
        self.title = title
        self.subtitle = subtitle
        self.path = path
        self.module = module
        self.summary = summary
        self.parts = list(parts)

    @property
    def dir(self):
        """Absolute path to the assignment's code."""
        return os.path.join(REPO_ROOT, self.path)

    @property
    def url_prefix(self):
        return "/" + self.slug

    @property
    def index_endpoint(self):
        """Endpoint of the assignment's overview page, for url_for()."""
        return "%s.index" % self.module.rsplit(".", 1)[-1]

    @property
    def present(self):
        """False if the folder is missing -- a stale registry entry."""
        return os.path.isdir(self.dir)

    def uploads(self):
        path = os.path.join(UPLOAD_ROOT, self.slug)
        os.makedirs(path, exist_ok=True)
        return path


ASSIGNMENTS = [
    Assignment(
        slug="module-2",
        number=2,
        title="Camera Calibration and Real-World Measurement",
        subtitle="Zhang calibration, measurement at known Z, validation, two cameras",
        path=os.path.join("week_2", "module2"),
        module="modules.module2",
        summary="Calibrate a smartphone camera, invert the perspective projection "
                "at a known object distance to measure real objects, and score "
                "the result against ruler ground truth over 20 measurements.",
        # The three pages, which are the three things done in a browser. Part D
        # is written, not interactive: theory/two_camera_derivation.md.
        parts=["Calibration", "Measure", "Records"],
    ),
]


def get(slug):
    """The registered assignment with this slug, or KeyError."""
    for a in ASSIGNMENTS:
        if a.slug == slug:
            return a
    raise KeyError("no assignment registered with slug %r" % slug)


def unregistered():
    """Assignment folders in the repo that no entry above claims.

    Returns (week, folder, relative path) triples, sorted, so the home page can
    show them as work that exists on disk but has no pages yet.
    """
    claimed = set(os.path.normpath(a.path) for a in ASSIGNMENTS)
    found = []
    for path in glob.glob(os.path.join(REPO_ROOT, "week_*", "module*")):
        if not os.path.isdir(path):
            continue
        rel = os.path.normpath(os.path.relpath(path, REPO_ROOT))
        if rel in claimed:
            continue
        week, folder = rel.split(os.sep)[:2]
        found.append((week, folder, rel))
    return sorted(found)

"""The web application: every assignment in this repository from one page.

HOW TO RUN
    # from the repository root
    pip install -r week_2/module2/requirements.txt
    python app/app.py
    # then open http://127.0.0.1:5000

    # options
    python app/app.py --port 8000 --host 0.0.0.0 --debug

WHY IT LIVES AT THE REPOSITORY ROOT
    It used to live at week_2/module2/app, inside the one assignment it served,
    so every path it knew ("../calibration/output") pointed into Module 2 and
    nothing else could ever be reached from it. Sitting above the week folders,
    it mounts one blueprint per assignment and serves them all from one server:

        /                    this home page, listing every assignment
        /module-2/...        Module 2 (week_2/module2)

WHAT THIS FILE OWNS
    The Flask instance, the upload cap, the home page, and the navigation
    context shared by every page. Nothing about any individual assignment --
    that is app/modules/<name>.py, listed in app/assignments.py.

WHAT IT DELIBERATELY DOES NOT OWN
    Any geometry, statistics or calibration code. Those stay in the assignment
    folders and are imported by the blueprints, so the web app and the CLI run
    the same functions and cannot disagree.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)          # so `modules.*` and `assignments` import
                                      # whether run as a script or imported

from flask import Flask, render_template, request  # noqa: E402

from assignments import (ASSIGNMENTS, REPO_ROOT, UPLOAD_ROOT,  # noqa: E402
                         unregistered)

# Phone photos are large, so the cap is generous; it is a property of the
# Flask instance, which is why it lives here and not in a blueprint.
MAX_CONTENT_MB = 200


def mount(app):
    """Import and register a blueprint per registered assignment.

    Returns {blueprint name: (assignment, nav)} for the navigation context,
    and the list of assignments that could not be mounted, so the home page
    can say so out loud instead of quietly dropping them.
    """
    mounted, broken = {}, []
    for assignment in ASSIGNMENTS:
        if not assignment.present:
            broken.append((assignment, "folder %s is missing" % assignment.path))
            continue
        try:
            module = importlib.import_module(assignment.module)
        except Exception as exc:                      # a half-finished week
            broken.append((assignment, "%s: %s" % (type(exc).__name__, exc)))
            continue
        app.register_blueprint(module.bp, url_prefix=assignment.url_prefix)
        mounted[module.bp.name] = (assignment, getattr(module, "NAV", []))
    return mounted, broken


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
    mounted, broken = mount(app)

    @app.context_processor
    def inject_chrome():
        """Shared chrome: the assignment switcher, plus the nav of the
        assignment whose blueprint is handling this request."""
        assignment, nav = mounted.get(request.blueprint, (None, []))
        return {
            "assignments": [a for a, _ in mounted.values()],
            "assignment": assignment,
            "nav": nav,
            "active_endpoint": request.endpoint,
        }

    @app.route("/")
    def home():
        return render_template(
            "home.html",
            cards=[{"assignment": a, "nav": nav} for a, nav in mounted.values()],
            broken=broken,
            pending=unregistered(),
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return render_template("error.html",
                              message="Upload exceeds the %d MB limit."
                                      % MAX_CONTENT_MB), 413

    app.mounted = mounted
    app.broken = broken
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="CSc 8830 assignments web application")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    app = create_app()

    print("=" * 66)
    print("CSc 8830 -- assignments web application")
    print("=" * 66)
    for name, (assignment, _) in sorted(app.mounted.items()):
        print("mounted  : %-10s %s  (%s)"
              % (assignment.url_prefix, assignment.title, assignment.path))
    for assignment, why in app.broken:
        print("SKIPPED  : %-10s %s" % (assignment.url_prefix, why))
    if not app.mounted:
        print("mounted  : NOTHING -- check the registry in app/assignments.py")
    print("open     : http://%s:%d" % (args.host, args.port))
    print("=" * 66)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())

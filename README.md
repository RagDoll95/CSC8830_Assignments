# CSc 8830 — Computer Vision · weekly assignments

One repository per course, one folder per week, and **one web application at the root
that reaches every assignment**.

```bash
# from this folder
pip install -r week_2/module2/requirements.txt
python app/app.py
# open http://127.0.0.1:5000
```

The home page lists every assignment; the switcher in the header moves between them from
any page, and each assignment keeps its own navigation bar for its own parts.

| URL | Assignment | Folder |
|---|---|---|
| `/` | Home — every assignment in the repository | — |
| `/module-2/` | Module 2 — camera calibration and real-world measurement | `week_2/module2/` |

## Layout

```
app/                     the web application (all assignments)
  app.py                 Flask instance, home page, shared navigation, upload cap
  assignments.py         the registry: which assignments exist and where they live
  modules/module2.py     Module 2's pages, as a Flask blueprint
  templates/             base.html + home.html, then one folder per assignment
  static/style.css
week_2/module2/          Module 2's own code: calibration, measurement, validation, theory
```

The application holds no computation of its own. Each assignment's pages import that
assignment's own modules — `week_2/module2/measurement/geometry.py` and friends — so the
web pages and the command-line tools run the same functions and cannot disagree;
`week_2/module2/tools/verify_pipeline.py` asserts numerically that they don't.

## Adding next week's assignment

1. Write the assignment under `week_N/moduleN/`, with its own CLI scripts.
2. Add `app/modules/moduleN.py` exposing a Flask blueprint `bp` and a `NAV` list.
3. Add one `Assignment(...)` entry to `app/assignments.py`.

Nothing in `app/app.py` changes — it mounts whatever the registry names. A `week_*/module*`
folder with no registry entry still shows up on the home page, listed as present in the
repository but with no pages yet, so unfinished work is visible rather than invisible.

## Per-assignment documentation

- [`week_2/module2/README.md`](week_2/module2/README.md) — method, results, layout, and the
  assignment-requirement map
- [`week_2/module2/NEXT_STEPS.md`](week_2/module2/NEXT_STEPS.md) — the run sheet for
  capturing real data and producing the submission

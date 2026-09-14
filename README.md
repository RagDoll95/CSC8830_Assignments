# CSc 8830 — Computer Vision · weekly assignments

Clone this repository, navigate to the root repo directory run
the following lines to launch

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

## Adding next week's assignment

1. Write the assignment under `week_N/moduleN/`, with its own CLI scripts.
2. Add `app/modules/moduleN.py` exposing a Flask blueprint `bp` and a `NAV` list.
3. Add one `Assignment(...)` entry to `app/assignments.py`.

## Per-assignment documentation

- [`week_2/module2/README.md`](week_2/module2/README.md) — method, results, layout, and the
  assignment-requirement map

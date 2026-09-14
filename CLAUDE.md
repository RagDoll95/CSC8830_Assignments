# CSc 8830 — Computer Vision assignments

Coursework repo. One Flask app at the repository root serves every weekly
assignment; each assignment's own code stays in its week folder and is imported
by the app, never reimplemented in it.

## Layout

```
app/                      the web app — serves every assignment
  app.py                  the Flask instance, the home page, shared nav context
  assignments.py          the registry: one Assignment(...) per assignment
  dataset.py              replacing synthetic stand-in data with real captures
  modules/<name>.py       one blueprint per assignment
  templates/, static/
week_2/module2/           Module 2's own code (calibration, measurement, validation, theory)
week_2/.venv/             the virtualenv (Python 3.9)
```

## Running

```bash
week_2/.venv/bin/python app/app.py          # http://127.0.0.1:5000
week_2/.venv/bin/python app/app.py --port 8000 --debug
```

Use `week_2/.venv/bin/python`, not the system Python — cv2, numpy, flask and
matplotlib are only installed there.

## Verifying a change

`week_2/module2/tools/verify_pipeline.py` is the real test suite: 25 checks that
score calibration against known synthetic ground truth, round-trip the geometry,
and assert the **web app and the CLI produce identical numbers**. Run it from
`week_2/module2/` after touching anything numeric:

```bash
cd week_2/module2 && ../.venv/bin/python tools/verify_pipeline.py
```

It imports `scale_click` from `app/modules/module2.py` directly, so that
function's name and signature are part of the contract.

## Conventions that matter here

**No geometry in the web layer.** Every number on a page comes from
`measurement/geometry.py`, `calibration/calibrate.py` or `validation/analyze.py`
— the same functions the CLI runs — so the two cannot disagree. A blueprint that
recomputes something itself is a bug even when it agrees.

**Uploads replace the synthetic data permanently.** `synthetic_*` files are
rendered, not photographed. A real upload deletes them from disk (see
`app/dataset.py` for why deletion rather than a hide-flag), so they cannot
reappear during a presentation. `dataset.stage()`/`install()` means the swap only
happens once the new data is known good — a failed calibration fit leaves the
previous dataset untouched. `git checkout -- week_2/module2` restores the
synthetic set.

**K is in pixels.** It is only valid at the resolution it was fitted at. Canvas
clicks arrive in display coordinates and must be scaled to full resolution before
they reach the geometry — server-side, against the server's own read of the file.
This is the single most expensive bug in the module; `scale_click()` exists only
to make it testable.

**The server recomputes, the browser never decides.** A recorded measurement
posts its pixel coordinates back and `measured_mm` is recomputed from them, so
the CSV is always consistent with the calibration and re-derivable via
`analyze.py --recompute`.

**Presentation over exposition.** These pages get demoed live. Keep step-by-step
instructions and long prose out of the visible page; put substance worth having
on hand in a `<details class="panel">` instead.

## Module 2 pages

| URL | What |
|---|---|
| `/module-2/` | Calibration — board photos in, K and distortion out |
| `/module-2/measure` | Photo + Z, click two points, Record against ruler ground truth |
| `/module-2/records` | Logged measurements, error statistics, plots, report figures `.zip` |

Part D (the two-camera derivation) is written, not interactive:
`week_2/module2/theory/two_camera_derivation.md`.

## Adding an assignment

1. Write `app/modules/<name>.py` exposing `bp` (a Flask Blueprint) and `NAV`.
2. Add one `Assignment(...)` entry to `app/assignments.py`.

`app/app.py` needs no change. A `week_*/module*` folder with no registry entry is
listed on the home page as unwired rather than silently hidden.

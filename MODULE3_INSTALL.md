# Module 3 installation

Copy this bundle over the root of `CSC8830_Assignments`, preserving the folder
structure. Existing `README.md`, `app/assignments.py`, and
`app/static/style.css` are replacements; the other files are additions.

Then run:

```bash
pip install -r requirements.txt
python -m unittest week_3/module3/test_image_blur.py
python app/app.py --host 0.0.0.0
```

Open `http://127.0.0.1:5000/module-3/`.

## Added files

- `app/modules/module3.py`
- `app/templates/module3/index.html`
- `week_3/module3/.gitignore`
- `week_3/module3/image_blur.py`
- `week_3/module3/test_image_blur.py`
- `week_3/module3/README.md`
- `week_3/module3/theory.md`
- `requirements.txt`

## Replaced files

- `README.md`
- `app/assignments.py`
- `app/static/style.css`

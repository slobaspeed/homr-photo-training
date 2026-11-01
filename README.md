# Photo training data for homr

Real phone photos of printed pages, with the printed music erased and other music with an exact answer pasted onto the same paper. Written for [homr](https://github.com/liebharc/homr); see the discussion in [liebharc/homr#206](https://github.com/liebharc/homr/discussions/206).

This is research code, copied as it ran on my machine. Paths point to where my data lives (see below). I wrote it together with Claude (an AI assistant). The numbers below are from my runs.

## The steps

| Script | What it does |
|---|---|
| `make_templates.py` | Printable A4 pages (`templates/templates.pdf`): music shrunk a little, with 8 ArUco squares around it (corners and edge middles). The squares say which page a photo shows and where each part of the flat page landed. |
| `generate_photos.py` | For a photo of a printed template: UVDoc guesses where each photo pixel lies on the flat page, the squares correct that guess, the printed music and squares are painted over with the paper around them, and new music is drawn through the same map, with ink darkness and blur matched to the photo. |
| `trial_check.py` | Compares real photos with the same template pasted back onto them (paper, ink, sharpness, grain), and measures placement: UVDoc alone vs corrected by the squares. On my 15 trial photos: 6.5 mm off (median) vs 4.0 mm. |
| `make_training.py` | `pages`: OpenScore Lieder songs engraved by Verovio (first page, random font and size), put onto the photos. `cut`: homr's own staff detection and dewarping on every generated photo; each crop is traced back through the map to the flat page to find its staff and bars, and the tokens come from homr's Lieder converter. Crops whose ends don't land on a drawn staff are dropped (3,835 of 4,056 kept). |
| `fine_tune.py` | Trains homr model 480 further on the crops and scores held-out crops (photos of a page never used in training). |
| `real_test.py` | Scores homr on real photos of printed templates against their exact answer keys (notes and rests, like `validation/sheet_music_photos.py`). |
| `make_pages.py` | Helpers for reading Lieder scores and cutting them to one page (used by `make_templates.py`). |

## What it needs

- Python with `verovio`, `resvg_py`, `PyMuPDF` (`fitz`), `opencv-python`, `numpy`, `scipy`, `torch`.
- [UVDoc](https://github.com/tanguymagne/UVDoc) (MIT), cloned into `third_party/UVDoc` (`model.py` and `model/best_model.pkl` are used).
- A homr checkout on `PYTHONPATH` for `make_training.py cut`, `fine_tune.py` and `real_test.py`.
- Music, at the paths in the scripts (two folders up from here, under `omr/runs/`): [OpenScore Lieder](https://github.com/OpenScore/Lieder), [ASAP](https://github.com/fosfrancesco/asap-dataset) and Mutopia pages. Change the constants at the top of `make_templates.py` and `make_pages.py` to point at your copies.
- Your own photos of the printed templates in `trial/`. At least a few squares should be visible; 4 or more spread around the page work best.

Pages T01–T22 of the templates use the music of homr's sheet-music-photos test set and are not used for training.

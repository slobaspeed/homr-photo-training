"""Trial check: are generated photos like real photos of the same printed template?

For every photo in trial/ (a phone photo of a printed page from templates.pdf), the marks say which template it shows.
The generator then puts that same template's flat page onto the photo, so the real and the generated photo show the
same music on the same paper, in the same light and at the same size. Both are measured the same way, on the page
area only:
  paper      brightness of the paper (0-255) and how uneven it is (spread of the paper level / its mean)
  ink        how dark the core of the ink is, as a share of the paper around it (lower = blacker)
  bold       share of the page that is ink (more = thicker strokes)
  sharp      steepness of the ink edges (90th percentile of the gradient at the edges)
             The generator sets its ink darkness and blur to match the photo's on similar measures, so ink and sharp
             are close to 1 by construction: look at the patches
  grain      pixel noise on blank paper
  place      how far a mark lands from where it really is, in mm on the page: UVDoc alone, and UVDoc corrected by
             the photo's other marks (each mark left out of the correction that is checked on it)

Writes trial_check/<photo>.jpg (the same patch, real left, generated right) and prints the table.

Run: server/.venv/Scripts/python research/synthetic-omr/trial_check.py
"""
import collections
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

import generate_photos as gp

HERE = Path(__file__).parent
TRIAL = HERE / 'trial'
OUT = HERE / 'trial_check'
DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
MARK_MM = 10
FLAT_MM = 300 / 25.4  # flat page pixels per mm
KEYS = ['paper', 'uneven', 'ink', 'bold', 'sharp', 'grain']
TEMPLATES = {t['name']: t for t in json.loads((HERE / 'templates' / 'templates.json').read_text(encoding='utf-8'))['templates']}


def marks(img):
    corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return {} if ids is None else {i: c[0] for i, c in zip(ids.ravel().tolist(), corners)}


def upright(photo, found):
    """The photo turned by quarter turns so the page's top is up. UVDoc takes the photo's top as the page's top, so a
    sideways page got its music drawn across the paper. A mark's first two corners run along the mark's top edge."""
    angle = statistics.median(float(np.degrees(np.arctan2(*(c[1] - c[0])[::-1]))) for c in found.values())
    turn = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}.get(round(angle / 90) % 4)
    return photo if turn is None else cv2.rotate(photo, turn)


def measure(img, page, px_mm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(3, round(3 * px_mm) | 1)  # wider than a stroke, so closing removes the ink and leaves the paper
    bg = cv2.medianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(np.uint8), k)
    n = gray / np.maximum(bg.astype(np.float32), 1)
    page = cv2.erode(page.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    any_ink = (n < 0.82) & page
    core = (n < 0.6) & page
    paper = page & ~cv2.dilate(any_ink.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    gx, gy = cv2.Sobel(n, cv2.CV_32F, 1, 0), cv2.Sobel(n, cv2.CV_32F, 0, 1)
    edges = cv2.dilate(any_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & page & ~core
    level = bg[page].astype(np.float32)
    return {'paper': float(np.median(level)), 'uneven': float(level.std() / level.mean()),
            'ink': float(np.median(n[core])) if core.any() else float('nan'), 'bold': float(any_ink.sum() / page.sum()),
            'sharp': float(np.percentile(np.hypot(gx, gy)[edges], 90)),
            'grain': float((n - cv2.GaussianBlur(n, (0, 0), 2))[paper].std())}


def placement(raw, found, truth):
    """Mean corner error in mm of UVDoc alone, and of UVDoc corrected by the other marks (None with one mark)."""
    before, after = [], []
    raw = gp.extended(*raw)
    for i in found:
        guess = gp.at(raw, found[i])
        if np.isnan(guess).any():
            continue
        before.append(np.linalg.norm(guess - truth[i], axis=1).mean() / FLAT_MM)
        others = [j for j in found if j != i]
        g = gp.at(raw, np.concatenate([found[j] for j in others])) if others else np.zeros((0, 2))
        ok = ~np.isnan(g).any(axis=1)
        if ok.sum() >= 4:
            m = gp.mark_fit(g[ok], np.concatenate([truth[j] for j in others])[ok])
            x, y = gp.apply_fit(m, guess[:, 0], guess[:, 1])
            after.append(np.linalg.norm(np.stack([x, y], 1) - truth[i], axis=1).mean() / FLAT_MM)
    return (statistics.mean(before) if before else float('nan')), (statistics.mean(after) if after else None)


def old_print(template):
    """Where a template's print lies on the flat page: the music with 8 mm to spare (the map is off by about 4 mm), and
    the small label at the bottom (make_templates.on_template) with 3 mm."""
    mm = FLAT_MM
    x, y, w, h = template['music_box']
    label_x, label_y = gp.FW / 2 + 12 * mm, gp.FH - 7 * mm
    return [(x - 8 * mm, y - 8 * mm, x + w + 8 * mm, y + h + 8 * mm),
            (label_x - 3 * mm, label_y - 6 * mm, gp.FW - 16 * mm, label_y + 4 * mm)]


def mark_boxes(template):
    """Every mark of a template on the flat page, at 1.6 times its size (the map is off by about 4 mm)."""
    boxes = []
    for corners in template['marks']['corners']:
        c = np.float32(corners)
        middle, half = c.mean(axis=0), 0.8 * (c.max(axis=0) - c.min(axis=0))
        boxes.append((*(middle - half), *(middle + half)))
    return boxes


def prepare(path):
    """The photo upright at the generator's size, its template, its marks and the generator's conditions."""
    photo = cv2.imread(str(path))
    photo = cv2.resize(photo, None, fx=2000 / max(photo.shape[:2]), fy=2000 / max(photo.shape[:2]),
                       interpolation=cv2.INTER_AREA)  # the size the generator works at
    found = marks(photo)
    if not found:
        return None
    number = collections.Counter(i // 8 for i in found).most_common(1)[0][0]
    photo = upright(photo, {i: c for i, c in found.items() if i // 8 == number})
    found = {i: c for i, c in marks(photo).items() if i // 8 == number}
    template = TEMPLATES[f'T{number:02d}']
    truth = {i: np.float32(c) for i, c in zip(template['marks']['ids'], template['marks']['corners'])}
    px_mm = statistics.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean() for c in found.values()) / MARK_MM
    raw = gp.page_maps(photo)
    known = (np.concatenate(list(found.values())), np.concatenate([truth[i] for i in found]))
    cond = gp.conditions(photo, known, raw, old_print(template), mark_boxes(template))
    flat = cv2.imread(str(HERE / 'templates' / 'flat' / f'T{number:02d}.png'), cv2.IMREAD_GRAYSCALE)
    return {'name': path.stem, 'number': number, 'photo': photo, 'found': found, 'px_mm': px_mm, 'cond': cond,
            'flat': flat, 'place': placement(raw, found, truth)}


def patch(real, made, page):
    """The same 600x450 patch of both, from the middle of the page."""
    ys, xs = np.nonzero(page)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = real.shape[:2]
    y0, x0 = min(max(cy - 225, 0), h - 450), min(max(cx - 300, 0), w - 600)
    return np.hstack([real[y0:y0 + 450, x0:x0 + 600], np.full((450, 8, 3), 255, np.uint8), made[y0:y0 + 450, x0:x0 + 600]])


def main(bleed=gp.BLEED):
    OUT.mkdir(exist_ok=True)
    ratios = collections.defaultdict(list)
    befores, afters = [], []
    print(f"{'photo':16} {'page':4} " + ' '.join(f'{k:>13}' for k in KEYS) + '   place mm UVDoc -> corrected')
    for path in sorted(TRIAL.glob('*.jpg')):
        p = prepare(path)
        if p is None:
            print(f'{path.stem[-12:]:16} no marks found, skipped')
            continue
        made = gp.put_on_photo(p['photo'], p['cond'], p['flat'], bleed)
        page = p['cond'][2]
        real_m, made_m = measure(p['photo'], page, p['px_mm']), measure(made, page, p['px_mm'])
        for k in KEYS:
            ratios[k].append(made_m[k] / real_m[k])
        before, after = p['place']
        befores.append(before)
        if after is not None:
            afters.append(after)
        cv2.imwrite(str(OUT / f'{path.stem}.jpg'), patch(p['photo'], made, page), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cells = ' '.join(f'{real_m[k]:6.3g}/{made_m[k]:<6.3g}' for k in KEYS)
        print(f"{path.stem[-12:]:16} T{p['number']:02d}  {cells}   {before:5.1f} -> "
              f"{'  -  ' if after is None else f'{after:4.1f}'} ({len(p['found'])} marks)")
    print(f'\ngenerated / real, median over photos (1.00 = same), bleed {bleed}:')
    for k in KEYS:
        print(f'  {k:7} {statistics.median(ratios[k]):.2f}   (range {min(ratios[k]):.2f} .. {max(ratios[k]):.2f})')
    print(f'  place   UVDoc alone {statistics.median(befores):.1f} mm (median, worst {max(befores):.1f}); corrected by the '
          f'other marks {statistics.median(afters):.1f} mm (median, worst {max(afters):.1f}, {len(afters)} photos)')


if __name__ == '__main__':
    main()

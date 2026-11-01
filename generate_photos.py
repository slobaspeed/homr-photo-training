"""SPIKE: put the flat pages of make_pages.py onto every real photo of a sheet-music-photos page.

Same method as spike_uvdoc.py: UVDoc predicts the page's bending from the photo alone, the old ink is painted
over, and the new page darkens the paper as much as the photo's own ink does. Writes, to gen/:
  <photo>__<song>.jpg   the generated photo
  <song>__flat.png      the flat page itself (homr's "clean scan" baseline for that song)

Run: python generate_photos.py <sheet-music-photos page folder>
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import griddata

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'third_party' / 'UVDoc'))
from model import UVDocnet  # noqa: E402  (plain torch layers, read before use)

IMG_SIZE = (488, 712)
FW, FH = 2480, 3508  # A4 at 300 dpi, the resolution of sheet-music-photos' print.pdf (1-bit, specks included)
SUPER = 2  # draw the page at twice the photo's size, then shrink: what a camera plus INTER_AREA does
BLEED = 0.25  # see put_on_photo; ink cover vs real prints 0.99 on trial pages T13 T23 T24 (picked there), 0.94 on T14 T29
BLURS = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]  # Gaussian sigmas (photo pixels) tried to match the photo's ink edges
OUT = HERE / 'gen'
OUT.mkdir(exist_ok=True)

model = UVDocnet(num_filter=32, kernel_size=5)
model.load_state_dict(torch.load(HERE / 'third_party' / 'UVDoc' / 'model' / 'best_model.pkl', map_location='cpu',
                                 weights_only=True)['model_state'])
model.eval()


def page_maps(photo):
    """UVDoc's guess of where each photo pixel lies on the flat page, in flat page pixels (NaN off the page)."""
    ph, pw = photo.shape[:2]
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    with torch.no_grad():
        grid, _ = model(torch.from_numpy(cv2.resize(rgb, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0))
    grid = grid[0].numpy()
    rows, cols = grid.shape[1:]
    pts = np.stack([((grid[0] + 1) / 2 * (pw - 1)).ravel(), ((grid[1] + 1) / 2 * (ph - 1)).ravel()], axis=1)
    u, v = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    qx, qy = np.meshgrid(np.arange(pw), np.arange(ph))
    map_u = griddata(pts, u.ravel(), (qx, qy), method='linear')
    map_v = griddata(pts, v.ravel(), (qx, qy), method='linear')
    return (map_u * (FW - 1)).astype(np.float32), (map_v * (FH - 1)).astype(np.float32)


def mark_fit(guess, truth):
    """The 3x3 transform that moves UVDoc's guessed flat positions of the mark corners onto their true positions:
    a shift, turn and scale for one or two marks, affine for three, a homography for four or more (the best on the
    trial photos, each mark predicted from the others)."""
    guess, truth = np.float32(guess), np.float32(truth)
    if len(guess) >= 16:  # three marks often sit in a row, where a homography swings wildly beyond them
        return cv2.findHomography(guess, truth, 0)[0]
    fit = cv2.estimateAffine2D if len(guess) >= 12 else cv2.estimateAffinePartial2D  # two marks can't hold a shear
    return np.vstack([fit(guess, truth, method=cv2.LMEDS)[0], [0, 0, 1]])


def apply_fit(m, x, y):
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w


def extended(map_x, map_y):
    """The maps carried past the page UVDoc saw, by a homography fitted to its map. In a close-up UVDoc's page stops
    short of the frame, so the marks near the edge fell off it and the map could not be corrected."""
    gone = np.isnan(map_x)
    if not gone.any() or gone.all():
        return map_x, map_y
    ys, xs = np.nonzero(~gone[::20, ::20])
    src = np.stack([xs * 20, ys * 20], axis=1).astype(np.float32)
    dst = np.stack([map_x[ys * 20, xs * 20], map_y[ys * 20, xs * 20]], axis=1)
    m = cv2.findHomography(src, dst, 0)[0]
    qy, qx = np.nonzero(gone)
    fx, fy = apply_fit(m, qx.astype(np.float32), qy.astype(np.float32))
    map_x, map_y = map_x.copy(), map_y.copy()
    map_x[qy, qx], map_y[qy, qx] = fx, fy
    return map_x, map_y


def at(maps, points):
    """The maps' values at photo points (x, y)."""
    map_x, map_y = maps
    xi = np.clip(np.round(points[:, 0]).astype(int), 0, map_x.shape[1] - 1)
    yi = np.clip(np.round(points[:, 1]).astype(int), 0, map_x.shape[0] - 1)
    return np.stack([map_x[yi, xi], map_y[yi, xi]], axis=1)


def edge_sharpness(n, page):
    """Steepness of the ink edges in an image divided by its paper (1 = paper): 90th percentile of the gradient."""
    any_ink = (n < 0.82) & page
    edges = cv2.dilate(any_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & page & (n >= 0.6)
    if not edges.any():
        return 0.0
    return float(np.percentile(np.hypot(cv2.Sobel(n, cv2.CV_32F, 1, 0), cv2.Sobel(n, cv2.CV_32F, 0, 1))[edges], 90))


def ink_strength(darkness, page, level):
    """How much a fully dark pixel darkens the paper, so the core of the drawn ink is as dark as the photo's (level,
    a share of the paper). Thin strokes never reach full darkness once shrunk and blurred, so the ink came out pale."""
    strength = 1 - level
    for _ in range(5):
        core = (1 - strength * darkness < 0.6) & page
        if not core.any():
            break
        strength = min((1 - level) / max(float(np.median(darkness[core])), 1e-3), 1.0)
    return strength


def erased(photo, gray, around, inside, where, solid, known=None):
    """The photo with its music painted over by the paper around it. The ink is taken generously, its soft grey edges
    too (a blurry photo's edges were left behind and smeared into grey ghosts of the old music), and every hole is
    filled with the paper level of the blank pixels near it plus fresh grain as fine and as strong as the photo's.
    where: the pixels the old print can be on; a strip of dark table beside the paper's edge passes for ink too.
    solid: pixels painted over whatever they show (marks cut off by the frame, which the detector can't find).
    inside: the pixels surely on the paper, the only ones the paper level is taken from (the table around the page
    darkened the paper put into the big holes of the marks, near the edges)."""
    ink = (((gray < 0.96 * around) & where) | solid).astype(np.uint8)
    if known is not None:  # the marks too: solid squares wider than the paper estimate, so their insides were left
        # behind, and homr read the half-erased mark beside the new one as a short staff
        for quad in known[0].reshape(-1, 4, 2):  # with a margin: a mark's grey halo darkened the paper put there
            middle = quad.mean(axis=0)
            cv2.fillConvexPoly(ink, np.round(middle + 1.5 * (quad - middle)).astype(np.int32), 1)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    f = photo.astype(np.float32)
    clear = ((ink == 0) & inside).astype(np.float32)[..., None]
    level = None
    for sigma in (120, 40, 12):  # a wide hole takes its paper from further away
        weight = cv2.GaussianBlur(clear, (0, 0), sigma)[..., None]
        guess = cv2.GaussianBlur(f * clear, (0, 0), sigma) / np.maximum(weight, 1e-6)
        near = np.clip((weight - 0.1) / 0.2, 0, 1)  # blended, so no edge where one reach takes over
        level = guess if level is None else near * guess + (1 - near) * level
    fine = f - cv2.GaussianBlur(f, (0, 0), 2)
    grain = np.random.default_rng(0).normal(0, 1, f.shape).astype(np.float32)
    grain -= cv2.GaussianBlur(grain, (0, 0), 2)
    grain *= fine[(ink == 0) & inside].std(axis=0) / grain.reshape(-1, 3).std(axis=0)
    w = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.5)[..., None]  # no seam at the hole's edge
    return f * (1 - w) + (level + grain) * w


def conditions(photo, known=None, maps=None, old=None, marks=None):
    """What a photo does to a page: the map from photo pixels to the flat page, the paper, the ink level and how sharp
    the ink edges are. known: (photo points, flat points) of a template's marks. UVDoc takes a close-up for the whole
    page and draws the music too small; the marks move its map to where the page really is. old: boxes on the flat page
    (left, top, right, bottom) the photo's own print lies in; only there is it painted over. marks: boxes on the flat
    page painted over entirely, found or not."""
    map_x, map_y = page_maps(photo) if maps is None else maps
    if known is not None:
        map_x, map_y = extended(map_x, map_y)
        guess = at((map_x, map_y), known[0])
        ok = ~np.isnan(guess).any(axis=1)
        if ok.sum() >= 4:
            map_x, map_y = apply_fit(mark_fit(guess[ok], known[1][ok]), map_x, map_y)
    with np.errstate(invalid='ignore'):
        on_page = ~np.isnan(map_x) & (map_x >= 0) & (map_x <= FW - 1) & (map_y >= 0) & (map_y <= FH - 1)
    map_x = np.where(on_page, map_x, -1).astype(np.float32)
    map_y = np.where(on_page, map_y, -1).astype(np.float32)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    around = cv2.medianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8)), 21)
    where = on_page
    if old is not None:
        where = np.zeros_like(on_page)
        for l, t, r, b in old:
            where |= on_page & (map_x >= l) & (map_x <= r) & (map_y >= t) & (map_y <= b)
    solid = np.zeros_like(on_page)
    for l, t, r, b in marks or []:
        solid |= on_page & (map_x >= l) & (map_x <= r) & (map_y >= t) & (map_y <= b)
    edge = 6 * FW / 210  # 6 mm: further in than the map is off
    inside = on_page & (map_x >= edge) & (map_x <= FW - 1 - edge) & (map_y >= edge) & (map_y <= FH - 1 - edge)
    paper = erased(photo, gray, around, inside, where, solid, known)
    n = gray.astype(np.float32) / np.maximum(around, 1)
    ink_level = float(np.median(n[(n < 0.6) & on_page]))  # the core of the ink, as a share of the paper around it
    sharp = edge_sharpness(n, on_page)
    return map_x, map_y, on_page, paper, ink_level, sharp


def put_on_photo(photo, cond, flat, bleed=BLEED):
    """bleed: printer ink spreads, so the flat page's strokes are widened by this many 300 dpi pixels each side.
    The ink is then blurred as much as the photo's own ink (out of focus, shaken, soft lens)."""
    map_x, map_y, on_page, paper, ink_level, sharp = cond
    if bleed:  # a fraction blends the two whole widths around it
        low = int(bleed)
        wide = [flat if r == 0 else cv2.erode(flat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                for r in (low, low + 1)]
        part = bleed - low
        flat = wide[0] if part == 0 else cv2.addWeighted(wide[0], 1 - part, wide[1], part, 0)
    h, w = map_x.shape
    big = (w * SUPER, h * SUPER)
    big_x = cv2.resize(map_x, big, interpolation=cv2.INTER_LINEAR)
    big_y = cv2.resize(map_y, big, interpolation=cv2.INTER_LINEAR)
    lit = cv2.remap(flat, big_x, big_y, cv2.INTER_LINEAR, borderValue=255)
    darkness = 1 - cv2.resize(lit, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    tried = []
    for s in BLURS:
        d = cv2.GaussianBlur(darkness, (0, 0), s) if s else darkness
        strength = ink_strength(d, on_page, ink_level)
        tried.append((abs(edge_sharpness(1 - d * strength, on_page) - sharp), s, d, strength))
    _, _, darkness, strength = min(tried, key=lambda t: t[:2])
    made = paper * (1 - darkness[..., None] * strength)
    result = photo.copy()
    result[on_page] = np.clip(made, 0, 255).astype(np.uint8)[on_page]
    return result


def main():
    page_dir = Path(sys.argv[1])
    songs = sorted((HERE / 'pages').glob('*.png'))
    for song in songs:
        shutil.copyfile(song, OUT / f'{song.stem}__flat.png')
    for photo_path in sorted((page_dir / 'photos').glob('*.jpg')):
        photo = cv2.imread(str(photo_path))
        scale = 2000 / max(photo.shape[:2])
        photo = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cond = conditions(photo)
        for song in songs:
            flat = cv2.imread(str(song), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(OUT / f'{photo_path.stem}__{song.stem}.jpg'), put_on_photo(photo, cond, flat),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'{photo_path.name}: {len(songs)} songs put on it (ink {cond[4]:.2f} of the paper)')


if __name__ == '__main__':
    main()

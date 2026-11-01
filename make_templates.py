"""Template pages to print and photograph: the photos give the generator real paper, light and bending.

Every template is an A4 page with the music shrunk a little (SCALE) and eight ArUco marks around it (corners and
edge middles). The marks say which template a photo shows and where each part of the flat page landed, even when
the page is cut off by the frame. Page k carries the mark ids 8k .. 8k+7, in the order of MARK_SPOTS.

Three kinds of music, so the photos show old and modern ink:
  old      the 22 sheet-music-photos pages in templates/src_old (print.pdf, pre-1930 scans; answer key in that repo)
  lilypond Mutopia pages engraved by LilyPond (MUTOPIA in paths.py; answer key only as MIDI)
  verovio  OpenScore Lieder (voice + piano) and ASAP (piano) excerpts engraved by Verovio in five fonts and three
           sizes; the exact answer key is written next to the page

Writes templates/flat/T<nn>.png (the flat page at 300 dpi, what the photo is compared against),
templates/keys/T<nn>.musicxml (verovio pages), templates/templates.json and templates/templates.pdf (to print at
100%, no "fit to page").

Run: python make_templates.py
"""
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import fitz
import numpy as np
import resvg_py
import verovio

from make_pages import first_measures, read_mxl
from paths import ASAP, LIEDER, MUTOPIA

HERE = Path(__file__).parent
OUT = HERE / 'templates'
W, H = 2480, 3508  # A4 at 300 dpi
MM = 300 / 25.4
SCALE = 0.86  # the music's share of the page width, leaving room for the marks
MARK = round(10 * MM)  # mark side
MARK_CENTER = 10 * MM  # from the page edge (printers leave about 4 mm blank)
MARK_SPOTS = ['top left', 'top middle', 'top right', 'right middle', 'bottom right', 'bottom middle', 'bottom left',
              'left middle']
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
FONTS = ['Leipzig', 'Bravura', 'Gootville', 'Leland', 'Petaluma']
SIZES = [30, 40, 55]  # Verovio scale: small, normal, large staves


def mark_centers():
    lo_x, hi_x, mid_x = MARK_CENTER, W - MARK_CENTER, W / 2
    lo_y, hi_y, mid_y = MARK_CENTER, H - MARK_CENTER, H / 2
    return [(lo_x, lo_y), (mid_x, lo_y), (hi_x, lo_y), (hi_x, mid_y), (hi_x, hi_y), (mid_x, hi_y), (lo_x, hi_y),
            (lo_x, mid_y)]


def on_template(music, number, label):
    """The music (grey, any size) centred on an A4 page at SCALE, with the marks and a small label."""
    f = min(SCALE * W / music.shape[1], (H - 2 * (MARK_CENTER + MARK)) / music.shape[0])
    small = cv2.resize(music, (round(music.shape[1] * f), round(music.shape[0] * f)), interpolation=cv2.INTER_AREA)
    page = np.full((H, W), 255, np.uint8)
    y, x = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
    page[y:y + small.shape[0], x:x + small.shape[1]] = small
    corners = []
    for spot, (cx, cy) in enumerate(mark_centers()):
        x0, y0 = round(cx - MARK / 2), round(cy - MARK / 2)
        quiet = page[y0 - MARK // 4:y0 + MARK + MARK // 4, x0 - MARK // 4:x0 + MARK + MARK // 4]
        if (quiet < 128).mean() > 0.002:
            print(f'   T{number:02d}: ink near the {MARK_SPOTS[spot]} mark ({(quiet < 128).mean():.1%} of its border)')
        page[y0:y0 + MARK, x0:x0 + MARK] = cv2.aruco.generateImageMarker(ARUCO, 8 * number + spot, MARK)
        corners.append([[x0, y0], [x0 + MARK, y0], [x0 + MARK, y0 + MARK], [x0, y0 + MARK]])
    cv2.putText(page, f'T{number:02d} {label}', (round(W / 2 + 12 * MM), round(H - 7 * MM)), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, 0, 2, cv2.LINE_AA)
    return page, corners, [x, y, small.shape[1], small.shape[0]]


def pdf_page(path, number):
    pix = fitz.open(path)[number].get_pixmap(dpi=300, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def old_pages():
    order = json.loads((OUT / 'src_old' / 'order.json').read_text())
    for name in order:
        yield pdf_page(OUT / 'src_old' / f'{name}.pdf', 0), {'kind': 'old', 'source': f'sheet-music-photos {name}'}


def lilypond_pages(rng):
    ensembles = set(json.loads((MUTOPIA / 'ensembles.json').read_text(encoding='utf-8')))
    duets = set(json.loads((MUTOPIA / 'duets.json').read_text(encoding='utf-8')))
    pdfs = sorted((MUTOPIA / 'pdf').glob('*.pdf'))
    groups = [('piano', [p for p in pdfs if p.stem not in ensembles | duets], 6), ('ensemble', [p for p in pdfs if p.stem in ensembles], 3),
              ('piano duet', [p for p in pdfs if p.stem in duets], 3)]
    for layout, paths, count in groups:
        for i, path in enumerate(rng.sample(paths, count)):
            doc = fitz.open(path)
            page = 0 if i % 3 == 0 or doc.page_count == 1 else 1  # every third a first page, with its title
            yield pdf_page(path, page), {'kind': 'lilypond', 'layout': layout, 'source': f'Mutopia {path.stem} page {page + 1}'}


def verovio_page(xml, font, size):
    """The most measures from the start that fit on one page, engraved; None when fewer than 4 fit."""
    options = {'pageWidth': 2100, 'pageHeight': 2970, 'scale': size, 'adjustPageHeight': False, 'footer': 'none',
               'font': font, 'pageMarginLeft': 150, 'pageMarginRight': 150,  # staves clear of the side marks
               'header': 'none'}  # Verovio's titles ran off the page or lost letters; old and LilyPond pages have titles

    def engrave(count):
        tk = verovio.toolkit()
        tk.setOptions(options)
        tk.loadData(first_measures(xml, count))
        return tk

    low, high = 1, len(ET.fromstring(xml.encode('utf-8')).find('part').findall('measure'))
    while low < high:
        mid = (low + high + 1) // 2
        if engrave(mid).getPageCount() == 1:
            low = mid
        else:
            high = mid - 1
    if low < 4:
        return None, None
    png = resvg_py.svg_to_bytes(svg_string=engrave(low).renderToSVG(1), background='white', zoom=1.5)
    return cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_GRAYSCALE), first_measures(xml, low)


def verovio_pages(rng):
    songs = sorted(LIEDER.rglob('*.mxl'))
    rng.shuffle(songs)
    pieces = sorted(ASAP.rglob('xml_score.musicxml'))
    rng.shuffle(pieces)
    sources = [('voice and piano', songs, read_mxl), ('piano', pieces, lambda p: p.read_text(encoding='utf-8'))]
    n = 0
    for layout, paths, read in sources:
        made = 0
        for path in paths:
            xml = read(path)
            if '<tremolo' in xml or len(ET.fromstring(xml.encode('utf-8')).findall('part')) > 2:
                continue
            font, size = FONTS[n % len(FONTS)], SIZES[n % len(SIZES)]
            music, key = verovio_page(xml, font, size)
            if music is None:
                continue
            name = re.sub(r'[^A-Za-z0-9]+', '_', str(path.relative_to(path.parents[2])))[:60]
            yield music, {'kind': 'verovio', 'layout': layout, 'font': font, 'size': size, 'source': name, 'key': key}
            n += 1
            made += 1
            if made == 8:
                break


def main():
    rng = random.Random(27)
    (OUT / 'flat').mkdir(parents=True, exist_ok=True)
    (OUT / 'keys').mkdir(exist_ok=True)
    templates, pdf = [], fitz.open()
    for number, (music, info) in enumerate([*old_pages(), *lilypond_pages(rng), *verovio_pages(rng)], 1):
        label = info['kind'] + (f" {info['font']}" if 'font' in info else '')
        page, corners, box = on_template(music, number, label)
        name = f'T{number:02d}'
        cv2.imwrite(str(OUT / 'flat' / f'{name}.png'), page)
        key = info.pop('key', None)
        if key:
            (OUT / 'keys' / f'{name}.musicxml').write_text(key, encoding='utf-8')
            info['key'] = f'keys/{name}.musicxml'
        templates.append({'name': name, **info, 'marks': {'ids': list(range(8 * number, 8 * number + 8)), 'corners': corners},
                          'music_box': box})
        ok, png = cv2.imencode('.png', page)
        pdf.new_page(width=595.28, height=841.89).insert_image(fitz.Rect(0, 0, 595.28, 841.89), stream=png.tobytes())
        print(f"{name}: {info['kind']:8} {info.get('layout', ''):15} {info.get('font', '')} {info['source']}")
    (OUT / 'templates.json').write_text(json.dumps({'page_px': [W, H], 'dpi': 300, 'mark_spots': MARK_SPOTS,
                                                    'templates': templates}, indent=1), encoding='utf-8')
    pdf.save(OUT / 'templates.pdf', garbage=3, deflate=True)
    print(f'{len(templates)} templates, {(OUT / "templates.pdf").stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()

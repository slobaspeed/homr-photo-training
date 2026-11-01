"""SPIKE: one-page excerpts of OpenScore Lieder songs, each with its exact answer key.

Picks songs (fixed seed) without tremolo shorthand (Verovio may draw it differently from what the XML means),
cuts each to as many measures as fit on one A4 page in Verovio, and writes for every song:
  <name>.musicxml  the measures on the page (the answer key)
  <name>.png       the page engraved flat, 1240 x 1754 (A4 at 150 dpi, as in spike_uvdoc.py)

Run: server/.venv/Scripts/python research/synthetic-omr/make_pages.py <how many>
"""
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import resvg_py
import verovio

HERE = Path(__file__).parent
SCORES = HERE.parent.parent / 'omr' / 'runs' / 'lieder_full' / 'repo' / 'scores'
OUT = HERE / 'pages'
OUT.mkdir(exist_ok=True)
OPTIONS = {'pageWidth': 2100, 'pageHeight': 2970, 'scale': 40, 'adjustPageHeight': False, 'footer': 'none'}


def read_mxl(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(('.xml', '.musicxml')) and not n.startswith('META-INF'))
        return z.read(name).decode('utf-8')


def first_measures(xml, count):
    """The score with only its first `count` measures in every part."""
    root = ET.fromstring(xml.encode('utf-8'))
    for part in root.findall('part'):
        for measure in part.findall('measure')[count:]:
            part.remove(measure)
    return ET.tostring(root, encoding='unicode')


def pages(xml):
    tk = verovio.toolkit()
    tk.setOptions(OPTIONS)
    tk.loadData(xml)
    return tk


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    songs = sorted(SCORES.rglob('*.mxl'))
    random.Random(27).shuffle(songs)
    made = 0
    for path in songs:
        xml = read_mxl(path)
        root = ET.fromstring(xml.encode('utf-8'))
        if '<tremolo' in xml or len(root.findall('part')) > 2 or len(root.findall('part/measure')) < 8:
            continue
        low, high = 1, len(root.find('part').findall('measure'))
        while low < high:  # the most measures that still fit on one page
            mid = (low + high + 1) // 2
            if pages(first_measures(xml, mid)).getPageCount() == 1:
                low = mid
            else:
                high = mid - 1
        if low < 8:
            continue
        excerpt = first_measures(xml, low)
        tk = pages(excerpt)
        png = np.frombuffer(bytes(resvg_py.svg_to_bytes(svg_string=tk.renderToSVG(1), background='white', zoom=1)), np.uint8)
        flat = cv2.resize(cv2.imdecode(png, cv2.IMREAD_GRAYSCALE), (1240, 1754), interpolation=cv2.INTER_AREA)
        name = re.sub(r'[^A-Za-z0-9]+', '_', f'{path.parent.parent.parent.name}_{path.parent.name}')[:60].strip('_')
        (OUT / f'{name}.musicxml').write_text(excerpt, encoding='utf-8')
        cv2.imwrite(str(OUT / f'{name}.png'), flat)
        print(f'{name}: {low} measures on the page, {len(root.findall("part"))} part(s)')
        made += 1
        if made == want:
            break


if __name__ == '__main__':
    main()

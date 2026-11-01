"""Training lines for homr from real photos: real songs put onto real paper, cut into staves the way homr reads them.

Two steps, in two Pythons (the generator needs torch, homr's staff finder runs in homr's own environment):

  pages   for every song, the first page engraved by Verovio (random font and size) and put onto one of the trial
          photos. Writes to train_photos/:
            <id>.jpg        the generated photo
            <id>.musicxml   the music on it (the exact answer)
            <id>.json       the photo it was put on, and where every staff of every system sits on the flat page
            maps/<photo>.npz  where each photo pixel lies on the flat page
  cut     homr finds and straightens the staves of each generated photo, exactly as when it reads one; each staff is
          traced back through the photo's map to the flat page, so it is known which staff it is and which bars it
          holds, and the answer comes from homr's own training code (the Lieder converter's MeasureCutter). A staff
          whose ends don't land on the ends of a flat staff is dropped, so a crop never gets a wrong answer.
          Writes train_photos/staves/<id>-<n>.png and .tokens, and train_photos/index.txt in homr's format
          (image,tokens per line).

Only photos of pages T23 and up are used: T01-T22 are the music of homr's photo test set, and erased music can leave
traces. Songs on the test pages T35-T50 are left out.

Run:  python make_training.py pages [songs]
      PYTHONPATH=<homr checkout> python make_training.py cut
"""
import collections
import json
import random
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'train_photos'
FIRST_TRAINING_PAGE = 23
END_SLACK = 0.04  # how far (share of the staff's width) a found staff's ends may land from the flat staff's ends


def layout(svg, zoom, f, x0, y0):
    """Per system: the number of bars and each staff's box (left, top, right, bottom of the lines) in flat pixels.
    Verovio draws in its own units: the page is the viewBox, the music is shifted by the page margin."""
    import xml.etree.ElementTree as ET

    width = float(re.search(r'<svg width="([\d.]+)px"', svg).group(1))
    view = float(re.search(r'class="definition-scale"[^>]*viewBox="0 0 ([\d.]+)', svg).group(1))
    mx, my = map(float, re.search(r'class="page-margin" transform="translate\(([\d.-]+), ([\d.-]+)\)', svg).groups())
    k = width / view * zoom * f

    def flat(x, y):
        return x0 + (x + mx) * k, y0 + (y + my) * k

    systems = []
    for system in ET.fromstring(svg).iter('{http://www.w3.org/2000/svg}g'):
        if system.get('class') != 'system':
            continue
        measures = [g for g in system if g.get('class', '').startswith('measure')]
        boxes = {}
        for measure in measures:
            staffs = [g for g in measure if g.get('class') == 'staff']
            for n, staff in enumerate(staffs):
                lines = [list(map(float, re.findall(r'[\d.-]+', p.get('d')))) for p in staff
                         if p.tag.endswith('path')][:5]
                (l, t), (r, b) = flat(min(a[0] for a in lines), min(a[1] for a in lines)), \
                    flat(max(a[2] for a in lines), max(a[1] for a in lines))
                box = boxes.get(n, [l, t, r, b])
                boxes[n] = [min(box[0], l), min(box[1], t), max(box[2], r), max(box[3], b)]
        systems.append({'bars': len(measures), 'staffs': [boxes[n] for n in sorted(boxes)]})
    return systems


def engrave(xml, font, size):
    """The most bars from the start that fit on one page: the page (grey, Verovio's page at zoom 1.5), its SVG and the
    music engraved. None when fewer than 4 bars fit."""
    import xml.etree.ElementTree as ET

    import cv2
    import numpy as np
    import resvg_py
    import verovio

    from make_pages import first_measures
    options = {'pageWidth': 2100, 'pageHeight': 2970, 'scale': size, 'adjustPageHeight': False, 'footer': 'none',
               'font': font, 'pageMarginLeft': 150, 'pageMarginRight': 150, 'header': 'none'}

    def toolkit(count):
        tk = verovio.toolkit()
        tk.setOptions(options)
        tk.loadData(first_measures(xml, count))
        return tk

    low, high = 1, len(ET.fromstring(xml.encode('utf-8')).find('part').findall('measure'))
    while low < high:
        mid = (low + high + 1) // 2
        if toolkit(mid).getPageCount() == 1:
            low = mid
        else:
            high = mid - 1
    if low < 4:
        return None
    svg = toolkit(low).renderToSVG(1)
    png = resvg_py.svg_to_bytes(svg_string=svg, background='white', zoom=1.5)
    return cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_GRAYSCALE), svg, first_measures(xml, low)


def songs():
    """Lieder songs and ASAP piano pieces, shuffled, without the ones on the template pages."""
    import make_templates as mt
    from make_pages import read_mxl

    used = {t['source'] for t in json.loads((HERE / 'templates' / 'templates.json').read_text(encoding='utf-8'))['templates']}
    paths = [(p, read_mxl) for p in sorted(mt.LIEDER.rglob('*.mxl'))]
    paths += [(p, lambda q: q.read_text(encoding='utf-8')) for p in sorted(mt.ASAP.rglob('xml_score.musicxml'))]
    random.Random(29).shuffle(paths)
    for path, read in paths:
        name = re.sub(r'[^A-Za-z0-9]+', '_', str(path.relative_to(path.parents[2])))[:60]
        if name in used:
            continue
        xml = read(path)
        if '<tremolo' in xml or xml.count('<part ') > 2:
            continue
        yield name, xml


def pages():
    import cv2
    import numpy as np

    import generate_photos as gp
    import make_templates as mt
    import trial_check as tc

    want = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    (OUT / 'maps').mkdir(parents=True, exist_ok=True)
    photos = []
    for path in sorted(tc.TRIAL.glob('*.jpg')):
        p = tc.prepare(path)
        if p is None or p['number'] < FIRST_TRAINING_PAGE:
            continue
        np.savez_compressed(OUT / 'maps' / f'{p["name"]}.npz', map_x=p['cond'][0], map_y=p['cond'][1])
        photos.append(p)
        print(f'{p["name"]}: T{p["number"]:02d}', flush=True)
    rng = random.Random(31)
    made = 0
    for name, xml in songs():
        if made == want:
            break
        engraved = engrave(xml, rng.choice(mt.FONTS), rng.choice(mt.SIZES))
        if engraved is None:
            continue
        music, svg, excerpt = engraved
        # placed as on a template, without the marks: the photo's map is in template pixels
        f = min(mt.SCALE * mt.W / music.shape[1], (mt.H - 2 * (mt.MARK_CENTER + mt.MARK)) / music.shape[0])
        small = cv2.resize(music, (round(music.shape[1] * f), round(music.shape[0] * f)), interpolation=cv2.INTER_AREA)
        flat = np.full((mt.H, mt.W), 255, np.uint8)
        y0, x0 = (mt.H - small.shape[0]) // 2, (mt.W - small.shape[1]) // 2
        flat[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        photo = photos[made % len(photos)]
        stem = f'{made:05d}'
        cv2.imwrite(str(OUT / f'{stem}.jpg'), gp.put_on_photo(photo['photo'], photo['cond'], flat),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / f'{stem}.musicxml').write_text(excerpt, encoding='utf-8')
        (OUT / f'{stem}.json').write_text(json.dumps({'photo': photo['name'], 'song': name,
                                                      'systems': layout(svg, 1.5, f, x0, y0)}))
        made += 1
        print(f'{stem}: {name} on {photo["name"]}', flush=True)


def answers(xml_path, systems):
    """The tokens of every (system, part), cut bar by bar in page order as homr's Lieder converter does, and how many
    staves each part has. None when the parts don't fill the systems' staves."""
    from homr.circle_of_fifths import strip_naturals
    from training.omr_datasets.convert_lieder import MeasureCutter, _count_staffs, contains_only_supported_clefs
    from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
    from training.transformer.training_vocabulary import token_lines_to_str

    voices = music_xml_file_to_tokens(str(xml_path))
    staves = [max(_count_staffs(v), 1) for v in voices]
    if any(len(s['staffs']) != sum(staves) for s in systems) or sum(s['bars'] for s in systems) != len(voices[0]):
        return None, staves
    cutters = [MeasureCutter(v) for v in voices]
    result = {}
    for i, system in enumerate(systems):
        for part, cutter in enumerate(cutters):
            symbols = cutter.extract_measures(system['bars'])
            result[i, part] = token_lines_to_str(strip_naturals(symbols)) if contains_only_supported_clefs(symbols) else None
    return result, staves


def cut():
    sys.modules.setdefault('torch', types.SimpleNamespace(Tensor=object))  # homr's tokens code imports it, unused here
    import cv2
    import numpy as np

    import homr.staff_parsing as sp
    from homr.debug import Debug
    from homr.main import ProcessingConfig, detect_staffs_in_image
    from homr.point_mapping import chain
    from homr.staff_regions import StaffRegions
    from homr.system_repair import repair_systems

    # the straightened staff before homr pads it onto the transformer's canvas: training pads it again itself
    captured = {}
    on_canvas = sp.center_image_on_canvas

    def capture(image, *args, **kwargs):
        captured['image'] = image
        return on_canvas(image, *args, **kwargs)

    sp.center_image_on_canvas = capture
    config = ProcessingConfig(enable_debug=False, enable_cache=False, write_staff_positions=False,
                              read_staff_positions=False, selected_staff=-1, transformer_use_gpu=False,
                              segnet_use_gpu=True, coreml_encoder=False, title_detection=False)
    (OUT / 'staves').mkdir(exist_ok=True)
    maps, lines, kept, dropped = {}, [], 0, collections.Counter()
    for image_path in sorted(OUT.glob('*.jpg')):
        info = json.loads(image_path.with_suffix('.json').read_text())
        if info['photo'] not in maps:
            m = np.load(OUT / 'maps' / f'{info["photo"]}.npz')
            maps[info['photo']] = (m['map_x'], m['map_y'])
        map_x, map_y = maps[info['photo']]
        systems = info['systems']
        try:
            tokens, staves = answers(image_path.with_suffix('.musicxml'), systems)
        except Exception as e:  # e.g. octave shifts, which homr's training code refuses (its Lieder converter skips them too)
            print(f'{image_path.stem}: no answer ({e}), skipped', flush=True)
            continue
        if tokens is None:
            print(f'{image_path.stem}: the answer does not fit the drawn staves, skipped', flush=True)
            continue
        # the flat box of every (system, part): a grand staff is its two staves together
        targets = []
        for i, system in enumerate(systems):
            first = 0
            for part, count in enumerate(staves):
                boxes = system['staffs'][first:first + count]
                targets.append(((i, part), [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                            max(b[2] for b in boxes), max(b[3] for b in boxes)]))
                first += count
        try:
            multi_staffs, image, debug, _, _, to_input = detect_staffs_in_image(str(image_path), config)
            multi_staffs = repair_systems(debug, multi_staffs)
        except Exception as e:
            print(f'{image_path.stem}: homr failed ({e}), skipped', flush=True)
            continue
        regions = StaffRegions(multi_staffs)
        n = 0
        for row in multi_staffs:
            for staff in row.staffs:
                n += 1
                try:
                    _, _, to_page = sp.prepare_staff_image(Debug(image, str(image_path), False), n, staff, image, regions)
                except Exception:
                    dropped['homr could not straighten it'] += 1
                    continue
                crop = captured['image']
                # homr's staff (left end, middle, right end), not the crop: the crop reaches further towards the
                # farther neighbour, so its middle can lie above or below the staff
                mid_y = (staff.min_y + staff.max_y) / 2
                ends = []
                for x in (staff.min_x, (staff.min_x + staff.max_x) / 2, staff.max_x):
                    px, py = to_input((x, mid_y))
                    xi, yi = int(round(px)), int(round(py))
                    inside = 0 <= yi < map_x.shape[0] and 0 <= xi < map_x.shape[1] and map_x[yi, xi] >= 0
                    ends.append((map_x[yi, xi], map_y[yi, xi]) if inside else None)
                if None in ends:
                    dropped['an end off the page'] += 1
                    continue
                (left, _), (_, middle), (right, _) = ends
                match = None
                for key, (l, t, r, b) in targets:
                    slack = END_SLACK * (r - l)
                    if t <= middle <= b and abs(left - l) <= slack and abs(right - r) <= slack:
                        match = key
                if match is None:
                    dropped['ends not on a drawn staff'] += 1
                    near = [(k, round(left - b[0]), round(right - b[2])) for k, b in targets if b[1] <= middle <= b[3]]
                    print(f'   staff {n}: flat x {left:.0f}..{right:.0f}, y {middle:.0f}; off the drawn ends by {near}',
                          flush=True)
                    continue
                text = tokens[match]
                if text is None:
                    dropped['unsupported clef'] += 1
                    continue
                name = f'{image_path.stem}-{n}'
                cv2.imwrite(str(OUT / 'staves' / f'{name}.png'), crop)
                (OUT / 'staves' / f'{name}.tokens').write_text(text, encoding='utf-8')
                # which (system, part) it is, so the answer can be written again by another version of homr's code
                (OUT / 'staves' / f'{name}.where').write_text(json.dumps(match))
                lines.append(f'staves/{name}.png,staves/{name}.tokens')
                kept += 1
        print(f'{image_path.stem}: {n} staves found, {len(targets)} drawn; {kept} kept so far', flush=True)
    (OUT / 'index.txt').write_text('\n'.join(lines) + '\n')
    print(f'{kept} staves written; dropped: {dict(dropped)}')


def check():
    """homr's reader on every staff crop against its answer: symbol error per crop (edit distance over the symbol
    lines / their number). A crop whose answer belongs to another staff reads as nearly all wrong."""
    sys.modules.setdefault('torch', types.SimpleNamespace(Tensor=object))
    import statistics

    import cv2

    from homr.circle_of_fifths import strip_naturals
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score
    from homr.transformer.vocabulary import is_upper_or_has_no_position
    from training.transformer.training_vocabulary import token_lines_to_str

    def distance(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, row[0] = row[0], i
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
        return row[-1]

    reader = Staff2Score(Config())
    errors = []
    for line in (OUT / 'index.txt').read_text().split():
        image, tokens = line.split(',')
        truth = (OUT / tokens).read_text(encoding='utf-8').splitlines()
        read = reader.predict(add_image_into_tr_omr_canvas(cv2.imread(str(OUT / image), cv2.IMREAD_GRAYSCALE)))
        if not any(' lower' in t for t in truth):
            read = [r for r in read if is_upper_or_has_no_position(r.position)]
        got = token_lines_to_str(strip_naturals(read)).splitlines()
        errors.append((distance(got, truth) / len(truth), image))
    errors.sort()
    print(f'{len(errors)} staves: median error {statistics.median(e for e, _ in errors):.1%}, '
          f'mean {statistics.mean(e for e, _ in errors):.1%}, over 30%: {sum(e > 0.3 for e, _ in errors)}')
    for e, image in errors[-12:]:
        print(f'   {e:.1%} {image}')


if __name__ == '__main__':
    {'pages': pages, 'cut': cut, 'check': check}[sys.argv[1]]()

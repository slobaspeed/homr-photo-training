"""The real-photo test: phone photos of the printed templates T43-T50 (never trained on), read by homr as a user
would take them, scored against each template's exact answer key (notes and rests only, as homr's
sheet-music-photos benchmark). The flat page of every template is read too, so photo damage shows as the rise from
flat to photo.

real_test/<model>/ holds T<nn>__<photo>.jpg and T<nn>__flat.png; homr reads it first (homr <folder>).

Run: PYTHONPATH=<homr checkout with validation/> python real_test.py m480 tuned
"""
import statistics
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'real_test'


def scores(model):
    """{image stem: error, or None when homr gave no reading}"""
    from validation.ned_score import _ned_from_parts, _parse_output

    found = {}
    for image in sorted(p for p in (OUT / model).iterdir() if p.suffix.lower() in ('.png', '.jpg')
                        and not p.stem.endswith('_teaser')):  # homr's preview of each page it read
        key = (HERE / 'templates' / 'keys' / f"{image.stem.split('__')[0]}.musicxml").read_text(encoding='utf-8')
        output = image.with_suffix('.musicxml')
        found[image.stem] = _ned_from_parts(*_parse_output(key, output.read_text(encoding='utf-8'), 'native', 'native',
                                                           False, True)).ned if output.exists() else None
    return found


def main():
    sys.modules.setdefault('torch', types.SimpleNamespace(Tensor=object))
    models = sys.argv[1:]
    found = {m: scores(m) for m in models}
    stems = sorted(found[models[0]])
    print('page / photo'.ljust(34) + ''.join(m.rjust(10) for m in models))
    for stem in stems:
        print(stem.ljust(34) + ''.join((f'{found[m][stem]:.1%}' if found[m].get(stem) is not None else 'no read').rjust(10)
                                       for m in models))
    for kind, pick in (('flat pages', lambda s: s.endswith('__flat')), ('photos', lambda s: not s.endswith('__flat'))):
        print(f'\n{kind}:')
        for m in models:
            errors = [found[m][s] for s in stems if pick(s) and found[m].get(s) is not None]
            missing = sum(1 for s in stems if pick(s) and found[m].get(s) is None)
            print(f'   {m}: mean {statistics.mean(errors):.1%}, median {statistics.median(errors):.1%}, '
                  f'over 20%: {sum(e > 0.2 for e in errors)} of {len(errors)} (not read {missing})')
    if len(models) == 2:
        a, b = models
        pairs = [(found[a][s], found[b][s]) for s in stems if not s.endswith('__flat')
                 and None not in (found[a].get(s), found[b].get(s))]
        print(f'\nphotos, {b} against {a}: better on {sum(y < x - 0.005 for x, y in pairs)}, '
              f'worse on {sum(y > x + 0.005 for x, y in pairs)}, same on the rest of {len(pairs)}')


if __name__ == '__main__':
    main()

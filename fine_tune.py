"""Does training on staves from our generated photos help homr? Model 480 trained further on them, and read before and
after on staves from photos it never saw: every page put onto the three photos of T29 is held back, so the test paper,
light and camera angle are new to it, and so are the songs.

This only shows whether the training works at all. Generated photos are not real ones: whether real phone photos read
better can only be seen on real photos with answer keys (T43-T50).

Run with PYTHONPATH=<a homr checkout whose code matches model 480> and torch installed:
  fine_tune.py split               train.txt and test.txt from train_photos/index.txt
  fine_tune.py score [weights]     symbol error on the test staves (default: model 480 as published)
  fine_tune.py train [epochs]      writes train_photos/run/fine_tuned.pth
"""
import os
import statistics
import sys
from pathlib import Path

from paths import MODEL_480

os.environ.setdefault('USE_TF', '0')  # transformers would load the system Python's TensorFlow, broken with numpy 2

HERE = Path(__file__).parent
OUT = HERE / 'train_photos'
RUN = OUT / 'run'
TEST_PAPER = {'IMG_20261027_171957', 'IMG_20261027_172018', 'IMG_20261027_172034'}  # the photos of T29


def split():
    """The answers are written again with this code: the cut ran with a newer homr, whose answers hold symbols model
    480 has never seen (dynamics)."""
    import json

    import make_training as mt
    from training.transformer.training_vocabulary import read_tokens, to_decoder_branches

    train, test, pages, lost = [], [], {}, 0
    for line in (OUT / 'index.txt').read_text().split():
        image, _ = line.split(',')
        name = image.split('/')[-1][:-len('.png')]
        page = name.split('-')[0]
        info = json.loads((OUT / f'{page}.json').read_text())
        if page not in pages:
            try:
                pages[page] = mt.answers(OUT / f'{page}.musicxml', info['systems'])[0]
            except Exception:
                pages[page] = None
        system, part = json.loads((OUT / 'staves' / f'{name}.where').read_text())
        text = (pages[page] or {}).get((system, part))
        if text is None:
            lost += 1
            continue
        tokens = OUT / 'staves' / f'{name}.tokens480'
        tokens.write_text(text, encoding='utf-8')
        if info['photo'] in TEST_PAPER:
            test.append(f'{OUT / image},{tokens}')
            continue
        try:  # a few answers hold combinations model 480's vocabulary lacks (e.g. staccatissimo_trill)
            to_decoder_branches(read_tokens(str(tokens)))
        except KeyError:
            lost += 1
            continue
        train.append(f'{OUT / image},{tokens}')
    RUN.mkdir(exist_ok=True)
    (RUN / 'train.txt').write_text('\n'.join(train) + '\n')
    (RUN / 'test.txt').write_text('\n'.join(test) + '\n')
    print(f'{len(train)} staves to train on, {len(test)} to test on; {lost} without an answer in this code')


def load(weights):
    import torch

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR

    config = Config()
    model = TrOMR(config)
    missing, unexpected = model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=False)
    if missing or unexpected:  # homr loads with strict=False; a weights file that doesn't fit would read as noise
        sys.exit(f'{weights} does not fit this code: {len(missing)} missing, {len(unexpected)} unexpected')
    return model.to('cuda'), config


def score():
    import cv2
    import torch

    from homr.circle_of_fifths import strip_naturals
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    from homr.transformer.vocabulary import is_upper_or_has_no_position
    from training.transformer.image_utils import ndarray_to_tensor, pad_to_3_dims
    from training.transformer.training_vocabulary import token_lines_to_str

    def distance(a, b):
        row = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, row[0] = row[0], i
            for j, y in enumerate(b, 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
        return row[-1]

    weights = sys.argv[2] if len(sys.argv) > 2 else str(MODEL_480)
    model, _ = load(weights)
    model.eval_mode()
    errors = []
    for line in (RUN / 'test.txt').read_text().split():
        image, tokens = line.split(',')
        truth = Path(tokens).read_text(encoding='utf-8').splitlines()
        canvas = add_image_into_tr_omr_canvas(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
        with torch.no_grad():
            read = model.generate(pad_to_3_dims(ndarray_to_tensor(canvas)).unsqueeze(0).to('cuda'))
        if not any(' lower' in t for t in truth):  # as check() in make_training.py
            read = [r for r in read if is_upper_or_has_no_position(r.position)]
        got = token_lines_to_str(strip_naturals(read)).splitlines()
        errors.append(distance(got, truth) / len(truth))
        if len(errors) % 100 == 0:
            print(f'{len(errors)} read, median so far {statistics.median(errors):.1%}', flush=True)
    print(f'{Path(weights).name} on {len(errors)} test staves: median error {statistics.median(errors):.1%}, '
          f'mean {statistics.mean(errors):.1%}, over 30%: {sum(e > 0.3 for e in errors)}')


def train():
    import torch
    from transformers import TrainingArguments

    from training.transformer.data_loader import label_names, load_dataset
    from training.transformer.distribute import Distribute
    from training.transformer.metrics import HomrTrainer

    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model, config = load(MODEL_480)  # everything trainable: the photo problem is in what the model sees
    lines = (RUN / 'train.txt').read_text().split()
    data = load_dataset(lines, config, val_split=0.05)
    args = TrainingArguments(
        str(RUN / 'checkpoints'),
        eval_strategy='epoch',
        save_strategy='no',
        learning_rate=1e-5,  # homr's own fine-tuning rate
        per_device_train_batch_size=4,  # a 6 GB card, full precision (a GTX 1060 has no bf16)
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        num_train_epochs=epochs,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        label_names=label_names,
        logging_steps=25,
        report_to=[],
        dataloader_num_workers=2,  # few: the PC cuts out when the GPU and every CPU core work at once
    )
    trainer = HomrTrainer(model, args, train_dataset=data['train'], eval_dataset=data['validation'],
                          distribute=Distribute())
    trainer.train()
    torch.save(model.state_dict(), RUN / 'fine_tuned.pth')
    print('written', RUN / 'fine_tuned.pth')


if __name__ == '__main__':
    {'split': split, 'score': score, 'train': train}[sys.argv[1]]()

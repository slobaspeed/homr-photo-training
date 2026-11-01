"""Where the music and the model live. By default in data/ next to these scripts; each can be set with an environment
variable instead (PHOTO_DATA moves all of them at once)."""
import os
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(os.environ.get('PHOTO_DATA', HERE / 'data'))


def _path(variable, default):
    return Path(os.environ.get(variable, DATA / default))


LIEDER = _path('LIEDER_SCORES', 'lieder/scores')  # the scores/ folder of github.com/OpenScore/Lieder
ASAP = _path('ASAP_DATASET', 'asap')  # github.com/fosfrancesco/asap-dataset
MUTOPIA = _path('MUTOPIA_PAGES', 'mutopia')  # pdf/*.pdf from mutopiaproject.org, plus ensembles.json and duets.json
MODEL_480 = _path('HOMR_MODEL_480', 'pytorch_model_480-5eed84636c96eca5ce6d8e2f62580b74e7ed85e8.pth')  # homr's model 480

"""Where the checkpoint, body models and datasets live. Nothing is hard-coded in the code: every path comes from
configs/paths.yaml (copy configs/paths.example.yaml), or from the file named by $MULTISMPL_PATHS.

  from multismpl.paths import PATHS
  PATHS.checkpoint, PATHS.smplx_root, PATHS.mamma_dance, ...
"""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

REPO = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO / 'configs/model_nano4.yaml'
KEYS = ('checkpoint', 'smplx_root', 'smpl_root', 'smplx2smpl', 'j_regressor_h36m', 'smpl_neutral',
        'mamma_dance', 'mamma_singles', 'chi3d', 'harmony4d_test')


def load_paths(path=None):
    path = Path(path or os.environ.get('MULTISMPL_PATHS') or REPO / 'configs/paths.yaml')
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found: copy configs/paths.example.yaml to configs/paths.yaml and fill it in')
    raw = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in KEYS if k not in raw]
    if missing:
        raise KeyError(f'{path}: missing {missing}')
    return SimpleNamespace(**{k: (Path(v) if Path(v).is_absolute() else REPO / v) for k, v in raw.items()})


class _Lazy:
    """PATHS.<key> reads the yaml on first use, so importing the package never fails."""

    def __getattr__(self, key):
        if '_p' not in self.__dict__:
            self.__dict__['_p'] = load_paths()
        return getattr(self.__dict__['_p'], key)


PATHS = _Lazy()

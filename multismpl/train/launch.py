# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Training entry point.

  python -m multismpl.train.launch --config train_mamma_harmony4d          # configs/train/<name>.yaml
  torchrun --nproc_per_node N -m multismpl.train.launch --config ...      # multi-GPU (DDP)
"""
import argparse

import multismpl.paths  # noqa: F401  (sets the torch env flag before torch is imported)
import torch
from hydra import compose, initialize_config_dir

from multismpl.paths import REPO
from multismpl.train.trainer import Trainer

CONFIG_DIR = REPO / 'configs' / 'train'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, default='train_mamma_harmony4d', help='config name in configs/train (no .yaml)')
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=args.config)

    print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == '__main__':
    main()

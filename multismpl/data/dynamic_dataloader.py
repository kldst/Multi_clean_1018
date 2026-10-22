# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from hydra.utils import instantiate
import logging
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
import torch.distributed as dist
from abc import ABC, abstractmethod

from .worker_fn import get_worker_init_fn

class DynamicTorchDataset(ABC):
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
        dataset_weights: Optional[list] = None,
        samples_per_epoch: Optional[int] = None,
        dataset_img_nums: Optional[list] = None,
        max_frames_per_gpu: Optional[int] = None,
        view_group_step_shares: Optional[dict] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        """
        dataset_weights: optional per-source share, one entry per dataset in the
            composed dataset's ``dataset_configs``, in order. Left unset, every clip
            is equally likely, so a source's share is its length over the total and
            adding a dataset shrinks all the others; set, each source contributes a
            fixed share regardless of its size (see ``DatasetWeightedSampler``).
            Weights need not sum to 1 and may be 0 to keep a source built but unused.
        samples_per_epoch: with ``dataset_weights``, the number of clips drawn per
            epoch; defaults to the concatenated length.
        dataset_img_nums: optional fixed views-per-frame, one entry per dataset in
            ``dataset_configs``, in order. Replaces ``img_nums`` so monocular and
            multi-view sources can train together (see ``ViewGroupedBatchSampler``).
        max_frames_per_gpu: with ``dataset_img_nums``, also cap each batch at this
            many timesteps (clips * T). SMPL decoding and its losses scale with
            frames x people, not images, so few-view batches need this cap.
        view_group_step_shares: with ``dataset_img_nums``, ``{views: share}`` of
            each epoch's steps, e.g. ``{4: 0.7, 1: 0.3}``. Unset, every group
            runs all its clips once per epoch.
        steps_per_epoch: with ``view_group_step_shares``, steps per epoch per
            rank. Defaults to the most steps for which no clip is drawn twice.
        """
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums    # e.g., [2, 24]

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        temporal_multiplier = (
            int(getattr(common_config, "temporal_clip_length", 1))
            if bool(getattr(common_config, "use_temporal_training", False))
            else 1
        )

        # Create samplers
        if dataset_img_nums is not None:
            if dataset_weights is not None:
                raise ValueError("dataset_img_nums and dataset_weights cannot be combined")
            # One object serves as both: get_loader only needs set_epoch on it.
            self.sampler = self.batch_sampler = ViewGroupedBatchSampler(
                self.dataset,
                source_img_nums=dataset_img_nums,
                aspect_ratio_range=self.aspect_ratio_range,
                max_img_per_gpu=max_img_per_gpu,
                max_frames_per_gpu=max_frames_per_gpu,
                temporal_multiplier=temporal_multiplier,
                shuffle=shuffle,
                seed=seed,
                group_step_shares=view_group_step_shares,
                steps_per_epoch=steps_per_epoch,
            )
            total = max(len(self.sampler), 1)
            for views, sources, clips, batch_size, batches in self.sampler.composition():
                logging.info(
                    "ViewGroupedBatchSampler: %d-view group (sources %s): %d clips, "
                    "batch size %d, %d batches/epoch per rank (%.1f%% of steps)",
                    views, sources, clips, batch_size, batches, 100.0 * batches / total,
                )
            return

        if dataset_weights is not None:
            self.sampler = DatasetWeightedSampler(
                self.dataset, weights=dataset_weights, num_draws=samples_per_epoch, seed=seed
            )
            for src, drawn, share, available in self.sampler.composition():
                logging.info(
                    "DatasetWeightedSampler: source %d draws %d/epoch (%.1f%%) of %d clips",
                    src, drawn, 100 * share, available,
                )
        else:
            self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu,
            temporal_multiplier=temporal_multiplier,
        )

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48,
                 temporal_multiplier=1):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu
        self.temporal_multiplier = max(1, int(temporal_multiplier))

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                # A temporal dataset sample contains T independent multi-view frames,
                # even though ``img_nums`` continues to mean views PER frame.
                batch_size = self.max_img_per_gpu / (
                    random_image_num * self.temporal_multiplier
                )
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Real (estimated) number of batches per epoch, so the trainer's
        # `iters_per_epoch = len(loader)` — and therefore the lr schedule's
        # `where = exact_epoch / max_epochs` — is correct when
        # limit_train_batches is null (auto). Previously this returned a dummy
        # 1_000_000, which made `where` advance only ~1/1e6 per step and froze the
        # intra-epoch lr.
        #
        # Each batch consumes `batch_size = floor(max_img_per_gpu / image_num)`
        # samples, with image_num drawn from `possible_nums`. Batches per epoch =
        # samples_per_replica / mean(batch_size). With a fixed img_nums (e.g.
        # [4, 4]) mean(batch_size) is a single value, so this is exact; with a
        # range it is an expectation (batch count is genuinely stochastic then).
        num_samples = len(self.sampler)  # per-replica (DistributedSampler)
        batch_sizes = [
            max(1, int(self.max_img_per_gpu // (int(num) * self.temporal_multiplier)))
            for num in self.possible_nums
        ]
        mean_batch_size = float(np.dot(self.normalized_weights, batch_sizes))
        if mean_batch_size <= 0:
            return 1
        return max(1, int(np.ceil(num_samples / mean_batch_size)))


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if not dist.is_available() or not dist.is_initialized():
            num_replicas = 1
            rank = 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num


class DatasetWeightedSampler(DynamicDistributedSampler):
    """
    Draws clips so every source dataset contributes a fixed share.

    The default sampler makes every clip equally likely, so a source's share of
    training is its clip count over the total, and adding a large dataset
    silently shrinks every other one. This sampler fixes the share instead.

    Over ``num_draws`` draws, source ``i`` contributes exactly
    ``round(num_draws * w_i)`` clips (largest-remainder rounding, so the counts
    sum to ``num_draws``). Each source is read without replacement from a fresh
    per-epoch permutation, and cycled only when its quota exceeds its size.
    Sources are interleaved by always drawing next from the source furthest
    behind its quota, which keeps every prefix of the schedule within one clip
    of the target mix -- so each batch carries that mix, not just the epoch.
    """

    def __init__(self, dataset, weights, num_draws: Optional[int] = None, seed: int = 0,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True,
                         seed=seed, drop_last=True)
        # The parent forces a single replica when torch.distributed is not up;
        # honour an explicit split so it can be exercised without a process group.
        if num_replicas is not None:
            self.num_replicas = int(num_replicas)
        if rank is not None:
            self.rank = int(rank)
        concat = getattr(dataset, "base_dataset", dataset)
        sizes = np.diff([0] + list(concat.cumulative_sizes)).astype(np.int64)
        w = np.asarray(list(weights), dtype=np.float64)
        if len(w) != len(sizes):
            raise ValueError(f"dataset_weights has {len(w)} entries for {len(sizes)} datasets")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"dataset_weights must be non-negative and not all zero, got {list(w)}")
        if ((w > 0) & (sizes == 0)).any():
            raise ValueError(f"a dataset with positive weight is empty: sizes={list(sizes)}")

        self.source_sizes = sizes
        self.source_offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.weights = w / w.sum()

        draws = int(num_draws) if num_draws else int(sizes.sum())
        self.total_size = draws - draws % self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

        raw = self.weights * self.total_size
        quota = np.floor(raw).astype(np.int64)
        quota[np.argsort(-(raw - quota), kind="stable")[: self.total_size - quota.sum()]] += 1
        self.quota = quota

    def composition(self):
        """(source, clips drawn per epoch, share, clips available) for logging."""
        return [(i, int(q), float(q) / max(self.total_size, 1), int(n))
                for i, (q, n) in enumerate(zip(self.quota, self.source_sizes))]

    def _schedule(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Largest-deficit interleave: every prefix stays within one clip of target.
        share = self.quota / max(self.total_size, 1)
        taken = np.zeros(len(self.quota), dtype=np.int64)
        source_order = np.empty(self.total_size, dtype=np.int64)
        for t in range(self.total_size):
            deficit = (t + 1) * share - taken
            deficit[taken >= self.quota] = -np.inf
            src = int(np.argmax(deficit))
            source_order[t] = src
            taken[src] += 1

        indices = np.empty(self.total_size, dtype=np.int64)
        for src, q in enumerate(self.quota):
            if not q:
                continue
            n = int(self.source_sizes[src])
            perm = np.concatenate([rng.permutation(n) for _ in range(-(-q // n))])[:q]
            indices[source_order == src] = self.source_offsets[src] + perm
        return indices

    def __iter__(self):
        for idx in self._schedule()[self.rank:self.total_size:self.num_replicas]:
            yield (int(idx), self.image_num, self.aspect_ratio)

    def __len__(self):
        return self.num_samples


class ViewGroupedBatchSampler(Sampler):
    """
    Batches sources by their own fixed views-per-frame count.

    ``img_nums`` sets one view count for every source, so a monocular source
    (BEDLAM) cannot share a loader with 4-view rigs: a batch collates only if
    all its clips have the same V, and a 1-view clip asked for 4 views just
    resamples until it gives up. Here each source declares its count; sources
    sharing a count form a group, and every batch is drawn from one group with
    ``max_img_per_gpu // (V * T)`` clips, so each batch still carries the same
    number of images whatever its V -- capped at ``max_frames_per_gpu // T``
    clips when set, because SMPL decoding scales with timesteps, not images.

    Each group's indices are (optionally) shuffled per epoch and dealt
    round-robin to ranks after truncating to a multiple of the world size, so
    every rank yields the same number of batches from every group -- DDP needs
    equal step counts; per-step shapes may differ across ranks.

    ``group_step_shares`` (``{views: share}``) instead fixes each group's share
    of an epoch's steps: a group then draws a fresh random subset of its clips
    every epoch (re-permuting only if its quota exceeds its size). Epoch length
    is ``steps_per_epoch``, or by default the most steps for which no group
    repeats a clip.
    """

    def __init__(self, dataset, source_img_nums, aspect_ratio_range, max_img_per_gpu=48,
                 temporal_multiplier=1, shuffle=True, seed=42, max_frames_per_gpu=None,
                 group_step_shares: Optional[dict] = None, steps_per_epoch: Optional[int] = None,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.aspect_ratio_range = aspect_ratio_range

        concat = getattr(dataset, "base_dataset", dataset)
        sizes = np.diff([0] + list(concat.cumulative_sizes)).astype(np.int64)
        views = [int(v) for v in source_img_nums]
        if len(views) != len(sizes):
            raise ValueError(f"dataset_img_nums has {len(views)} entries for {len(sizes)} datasets")
        if min(views) < 1:
            raise ValueError(f"dataset_img_nums must be >= 1, got {views}")
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])

        T = max(1, int(temporal_multiplier))
        self.groups = []
        for v in sorted(set(views), reverse=True):
            sources = [i for i, n in enumerate(views) if n == v]
            indices = np.concatenate(
                [np.arange(offsets[i], offsets[i] + sizes[i]) for i in sources]
            ).astype(np.int64)
            batch_size = int(max_img_per_gpu) // (v * T)
            if max_frames_per_gpu is not None:
                batch_size = min(batch_size, int(max_frames_per_gpu) // T)
            batch_size = max(1, batch_size)
            per_rank = len(indices) // self.num_replicas
            self.groups.append({
                "views": v,
                "sources": sources,
                "indices": indices,
                "batch_size": batch_size,
                "per_rank": per_rank,
                "num_batches": -(-per_rank // batch_size),
                "sampled": False,
            })
        if group_step_shares:
            self._apply_step_shares(group_step_shares, steps_per_epoch)

    def _apply_step_shares(self, group_step_shares, steps_per_epoch):
        shares = {int(k): float(v) for k, v in dict(group_step_shares).items()}
        present = {g["views"] for g in self.groups}
        if set(shares) != present:
            raise ValueError(f"view_group_step_shares keys {sorted(shares)} != view groups {sorted(present)}")
        weights = np.asarray([shares[g["views"]] for g in self.groups], dtype=np.float64)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(f"view_group_step_shares must be non-negative, not all zero: {shares}")
        weights = weights / weights.sum()
        if steps_per_epoch is None:
            full = np.asarray([g["per_rank"] // g["batch_size"] for g in self.groups], dtype=np.float64)
            steps = int(min(full[i] / weights[i] for i in range(len(weights)) if weights[i] > 0))
        else:
            steps = int(steps_per_epoch)
        if steps < 1:
            raise ValueError("an epoch must hold at least one step")
        raw = weights * steps
        quota = np.floor(raw).astype(np.int64)
        quota[np.argsort(-(raw - quota), kind="stable")[: steps - int(quota.sum())]] += 1
        for g, q in zip(self.groups, quota):
            if g["per_rank"] == 0 and q > 0:
                raise ValueError(f"{g['views']}-view group has a positive share but no clips")
            g["num_batches"] = int(q)
            g["sampled"] = True

    def composition(self):
        """(views, sources, clips, batch size, batches per rank) for logging."""
        return [(g["views"], g["sources"], len(g["indices"]), g["batch_size"], g["num_batches"])
                for g in self.groups]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _group_order(self, rng):
        counts = np.asarray([g["num_batches"] for g in self.groups], dtype=np.int64)
        if self.shuffle:
            return rng.permutation(np.repeat(np.arange(len(counts)), counts))
        # Deterministic even interleave, so a truncated val pass sees every group.
        total = int(counts.sum())
        share = counts / max(total, 1)
        taken = np.zeros_like(counts)
        order = np.empty(total, dtype=np.int64)
        for t in range(total):
            deficit = (t + 1) * share - taken
            deficit[taken >= counts] = -np.inf
            order[t] = int(np.argmax(deficit))
            taken[order[t]] += 1
        return order

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        R = self.num_replicas
        for g in self.groups:
            bs = g["batch_size"]
            if g["sampled"]:
                # This epoch's quota of full batches, drawn without replacement
                # until the group is exhausted, then from a fresh permutation.
                need = g["num_batches"] * bs * R
                pool, total = [], 0
                while total < need:
                    chunk = rng.permutation(g["indices"]) if self.shuffle else g["indices"]
                    pool.append(chunk)
                    total += len(chunk)
                pool = np.concatenate(pool) if pool else g["indices"][:0]
                mine = pool[:need][self.rank::R]
            else:
                indices = rng.permutation(g["indices"]) if self.shuffle else g["indices"]
                mine = indices[: g["per_rank"] * R][self.rank::R]
            batches.append([mine[i:i + bs] for i in range(0, len(mine), bs)])
        cursors = [0] * len(self.groups)
        lo, hi = self.aspect_ratio_range
        for group_id in self._group_order(rng):
            chunk = batches[group_id][cursors[group_id]]
            cursors[group_id] += 1
            aspect_ratio = round(float(rng.uniform(lo, hi)), 2)
            views = self.groups[group_id]["views"]
            yield [(int(idx), views, aspect_ratio) for idx in chunk]

    def __len__(self):
        return int(sum(g["num_batches"] for g in self.groups))

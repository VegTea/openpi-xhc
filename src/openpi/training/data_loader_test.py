import dataclasses

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


class EpisodeDataset:
    def __init__(self, episode_lengths: list[int]):
        starts = []
        ends = []
        cursor = 0
        self._episode_index = []
        for episode_index, episode_length in enumerate(episode_lengths):
            starts.append(cursor)
            cursor += episode_length
            ends.append(cursor)
            self._episode_index.extend([episode_index] * episode_length)
        self.episode_data_index = {"from": torch.tensor(starts), "to": torch.tensor(ends)}
        self.num_episodes = len(episode_lengths)

    def __getitem__(self, index):
        return {"episode_index": self._episode_index[index]}

    def __len__(self):
        return len(self._episode_index)


def test_split_torch_dataset_uses_disjoint_episodes():
    dataset = EpisodeDataset([3, 2, 4, 5, 1])

    train_dataset = _data_loader.split_torch_dataset(dataset, split="train", val_split_fraction=0.4, seed=0)
    val_dataset = _data_loader.split_torch_dataset(dataset, split="val", val_split_fraction=0.4, seed=0)

    train_indices = set(train_dataset.indices.tolist())
    val_indices = set(val_dataset.indices.tolist())
    assert train_indices.isdisjoint(val_indices)

    train_episodes = {train_dataset[i]["episode_index"] for i in range(len(train_dataset))}
    val_episodes = {val_dataset[i]["episode_index"] for i in range(len(val_dataset))}
    assert train_episodes.isdisjoint(val_episodes)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

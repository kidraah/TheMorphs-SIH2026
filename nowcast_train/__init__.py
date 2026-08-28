"""Training: day/episode-aware splitting, explicit NaN policy, resumable runs."""
from .checkpoint import find_latest, load_checkpoint, save_checkpoint
from .dataset import (ChannelStats, DatasetConfig, NaNPolicy, SEVIRDataset,
                      TargetConfig, collate, compute_channel_stats)
from .loop import TrainConfig, head_eval_configs, run_validation, train
from .splits import Split, SplitReport, count_episodes, episode_ids, split_by_time, split_sevir

__all__ = [
    "split_by_time", "split_sevir", "Split", "SplitReport", "count_episodes",
    "episode_ids", "SEVIRDataset", "DatasetConfig", "NaNPolicy", "TargetConfig",
    "ChannelStats", "compute_channel_stats", "collate",
    "save_checkpoint", "load_checkpoint", "find_latest",
    "train", "TrainConfig", "run_validation", "head_eval_configs",
]

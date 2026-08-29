"""Training: day/episode-aware splitting, explicit NaN policy, resumable runs."""
from .checkpoint import find_latest, load_checkpoint, save_checkpoint
from .cache import (CachedEvents, CacheConfig, CacheMismatch, build_cache,
                    read_manifest, verify_manifest)
from .calibration import (IsotonicCalibrator, apply_calibrators, fit_calibrators,
                          operating_points, select_threshold,
                          select_threshold_far_constrained, select_thresholds)
from .dataset import (CachedSEVIRDataset, ChannelStats, DatasetConfig, NaNPolicy,
                      SEVIRDataset, TargetConfig, collate, compute_channel_stats)
from .loop import (TrainConfig, format_operating_points, head_eval_configs,
                   run_validation, train)
from .splits import Split, SplitReport, count_episodes, episode_ids, split_by_time, split_sevir

__all__ = [
    "split_by_time", "split_sevir", "Split", "SplitReport", "count_episodes",
    "episode_ids", "SEVIRDataset", "CachedSEVIRDataset", "DatasetConfig",
    "NaNPolicy", "TargetConfig",
    "CachedEvents", "CacheConfig", "CacheMismatch", "build_cache",
    "read_manifest", "verify_manifest",
    "IsotonicCalibrator", "fit_calibrators", "apply_calibrators",
    "select_threshold", "select_thresholds", "select_threshold_far_constrained",
    "operating_points",
    "ChannelStats", "compute_channel_stats", "collate",
    "save_checkpoint", "load_checkpoint", "find_latest",
    "train", "TrainConfig", "run_validation", "head_eval_configs",
    "format_operating_points",
]

"""Tests for exo_bridge.py — Grove ↔ exo cluster integration."""

import json
import os
import tempfile
import pytest

from grove.exo_bridge import (
    ExoGroveWorld,
    ExoTransferBenchmark,
    ExoSparseSyncOptimizer,
    ExoAutoResearch,
    STAR_PLATINUM_NODES,
)


class TestExoGroveWorld:
    """Tests for ExoGroveWorld discovery and world creation."""

    def test_exo_grove_world_instantiation(self):
        """ExoGroveWorld can be created with default or custom URL."""
        world = ExoGroveWorld()
        assert world.exo_url == "http://localhost:52415"

        world2 = ExoGroveWorld(exo_url="http://192.168.1.100:8080")
        assert world2.exo_url == "http://192.168.1.100:8080"

    def test_exo_grove_world_fallback_nodes(self):
        """Returns Star Platinum nodes when exo API not available."""
        world = ExoGroveWorld(exo_url="http://invalid-host:99999")
        nodes = world.get_node_addresses()

        # Should fall back to Star Platinum hardcoded IPs
        assert len(nodes) == 4
        assert "192.168.1.151" in nodes
        assert "192.168.1.158" in nodes
        assert "192.168.1.177" in nodes
        assert "192.168.1.157" in nodes

    def test_exo_grove_world_node_count(self):
        """get_node_count returns correct count after discovery."""
        world = ExoGroveWorld(exo_url="http://invalid-host:99999")
        count = world.get_node_count()
        assert count == 4  # Star Platinum fallback

    def test_exo_grove_world_to_grove_world(self):
        """to_grove_world creates a valid Grove World object."""
        import grove
        # Save original state
        orig_rank, orig_world_size = grove.rank, grove.world_size
        
        try:
            world = ExoGroveWorld(exo_url="http://invalid-host:99999")
            grove_world = world.to_grove_world(rank=2)

            assert grove.rank == 2
            assert grove.world_size == 4
        finally:
            # Restore original state
            grove.rank = orig_rank
            grove.world_size = orig_world_size


class TestExoTransferBenchmark:
    """Tests for ExoTransferBenchmark bandwidth measurement and param recommendation."""

    def test_transfer_benchmark_instantiation(self):
        """ExoTransferBenchmark can be created with node list."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES)
        assert len(benchmark.nodes) == 4
        assert benchmark.test_sizes_mb == [1, 10, 50, 100]

    def test_transfer_benchmark_custom_sizes(self):
        """Custom test sizes are respected."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES, test_sizes_mb=[5, 25])
        assert benchmark.test_sizes_mb == [5, 25]

    def test_transfer_benchmark_recommend_params_fast(self):
        """Correct params for TB4-class (>10 Gbps) links."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES)
        params = benchmark.recommend_params(15.0)

        assert params["chunk_size"] == 8192
        assert params["topk"] == 256
        assert params["use_dct"] is False
        assert params["H"] == 50
        assert params["label"] == "tb4-raw"

    def test_transfer_benchmark_recommend_params_medium(self):
        """Correct params for WiFi-class (1-10 Gbps) links."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES)
        params = benchmark.recommend_params(2.5)

        assert params["chunk_size"] == 4096
        assert params["topk"] == 64
        assert params["use_dct"] is True
        assert params["H"] == 200
        assert params["label"] == "wifi-dct"

    def test_transfer_benchmark_recommend_params_slow(self):
        """Correct params for slow (<1 Gbps) links."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES)
        params = benchmark.recommend_params(0.5)

        assert params["chunk_size"] == 1024
        assert params["topk"] == 16
        assert params["use_dct"] is True
        assert params["H"] == 500
        assert params["label"] == "slow-link"

    def test_transfer_benchmark_run_returns_dict(self):
        """run() returns dict with expected structure."""
        benchmark = ExoTransferBenchmark(STAR_PLATINUM_NODES[:2])
        result = benchmark.run(timeout=5.0)

        assert "pairs" in result
        assert "min_bandwidth_gbps" in result
        assert "max_bandwidth_gbps" in result
        assert "avg_bandwidth_gbps" in result
        assert "recommended" in result
        assert "timestamp" in result


class TestExoSparseSyncOptimizer:
    """Tests for ExoSparseSyncOptimizer gradient synchronization."""

    def test_exo_sparse_sync_instantiation(self):
        """ExoSparseSyncOptimizer can be instantiated with benchmark results."""
        pytest.importorskip("mlx")
        import mlx.nn as nn

        model = nn.Linear(64, 64)
        benchmark_results = {
            "recommended": {
                "chunk_size": 4096,
                "topk": 64,
                "use_dct": False,
                "H": 100,
            },
            "min_bandwidth_gbps": 5.0,
            "avg_bandwidth_gbps": 8.0,
        }

        optimizer = ExoSparseSyncOptimizer(model, benchmark_results)

        assert optimizer.chunk_size == 4096
        assert optimizer.topk == 64
        assert optimizer.use_dct is False
        assert optimizer.H == 100
        assert optimizer.min_bandwidth == 5.0
        assert optimizer.inner_step == 0
        assert optimizer.outer_step == 0

    def test_exo_sparse_sync_step_returns_dict(self):
        """step() returns dict with sync stats."""
        pytest.importorskip("mlx")
        import mlx.nn as nn

        model = nn.Linear(64, 64)
        benchmark_results = {
            "recommended": {"chunk_size": 64, "topk": 8, "use_dct": False, "H": 2},
            "min_bandwidth_gbps": 1.0,
        }

        optimizer = ExoSparseSyncOptimizer(model, benchmark_results, H=2)

        # First step: no sync yet
        result1 = optimizer.step(model)
        assert result1["synced"] is False
        assert result1["inner_step"] == 1

        # Second step: should trigger sync
        result2 = optimizer.step(model)
        assert result2["synced"] is True
        assert "elapsed_ms" in result2
        assert "compression_ratio" in result2
        assert "throughput_mb_s" in result2


class TestExoAutoResearch:
    """Tests for ExoAutoResearch parameter tuning loop."""

    def test_autoresearch_param_grid(self):
        """All 6 configs in PARAM_GRID are valid."""
        assert len(ExoAutoResearch.PARAM_GRID) == 6

        required_keys = {"chunk_size", "topk", "use_dct", "H", "label"}
        labels = set()

        for config in ExoAutoResearch.PARAM_GRID:
            assert required_keys.issubset(config.keys())
            assert isinstance(config["chunk_size"], int)
            assert isinstance(config["topk"], int)
            assert isinstance(config["use_dct"], bool)
            assert isinstance(config["H"], int)
            assert isinstance(config["label"], str)
            labels.add(config["label"])

        # All labels should be unique
        assert len(labels) == 6

    def test_autoresearch_instantiation(self):
        """ExoAutoResearch can be instantiated."""
        research = ExoAutoResearch(STAR_PLATINUM_NODES)
        assert research.nodes == STAR_PLATINUM_NODES
        assert research.save_path == "grove_research_results.json"

    def test_autoresearch_run_trial_shape(self):
        """run_trial returns dict with expected keys."""
        research = ExoAutoResearch(STAR_PLATINUM_NODES)
        params = ExoAutoResearch.PARAM_GRID[0]

        result = research.run_trial(params, n_steps=5)

        required_keys = {
            "label",
            "chunk_size",
            "topk",
            "use_dct",
            "H",
            "throughput_mb_s",
            "compression_ratio",
            "stability_score",
            "gradient_variance",
            "score",
        }
        assert required_keys.issubset(result.keys())

        # Values should be reasonable
        assert result["throughput_mb_s"] > 0
        assert result["compression_ratio"] > 1
        assert 0 < result["stability_score"] <= 1
        assert result["score"] > 0

    def test_autoresearch_save_load_config(self):
        """save/load config roundtrip works."""
        research = ExoAutoResearch(STAR_PLATINUM_NODES)

        config = {
            "chunk_size": 4096,
            "topk": 64,
            "use_dct": True,
            "H": 100,
            "label": "test-config",
            "avg_score": 123.45,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            research.save_config(config, temp_path)

            loaded = ExoAutoResearch.load_best_config(temp_path)

            assert loaded["chunk_size"] == 4096
            assert loaded["topk"] == 64
            assert loaded["use_dct"] is True
            assert loaded["H"] == 100
            assert loaded["label"] == "test-config"
            assert loaded["avg_score"] == 123.45
        finally:
            os.unlink(temp_path)

    def test_autoresearch_run_single_round(self):
        """run() with 1 round completes and returns winner."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            research = ExoAutoResearch(STAR_PLATINUM_NODES, save_path=temp_path)
            best = research.run(n_rounds=1)

            # Should return a config dict
            assert "chunk_size" in best
            assert "topk" in best
            assert "use_dct" in best
            assert "H" in best
            assert "label" in best
            assert "avg_score" in best

            # Results file should exist
            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                results = json.load(f)

            assert "winner" in results
            assert "leaderboard" in results
            assert len(results["leaderboard"]) == 6
        finally:
            os.unlink(temp_path)

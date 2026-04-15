import json

from matpropnet.tasks.core import run_embedding_visualization
from matpropnet.visualization.embedding import (
    generate_embedding_visualizations,
    replot_embedding_visualizations,
)


def test_generate_embedding_visualizations_writes_artifacts(tmp_path):
    predictions = {
        "id": ["mp-1", "mp-2", "mp-3", "mp-4"],
        "target_log_b": [0.1, 0.2, 0.3, 0.4],
        "pred_log_b": [0.11, 0.19, 0.29, 0.41],
        "target_log_g": [0.5, 0.6, 0.7, 0.8],
        "pred_log_g": [0.49, 0.62, 0.68, 0.79],
        "z": [
            json.dumps([0.0, 0.1, 0.2]),
            json.dumps([0.3, 0.4, 0.5]),
            json.dumps([0.6, 0.7, 0.8]),
            json.dumps([0.9, 1.0, 1.1]),
        ],
    }

    result = generate_embedding_visualizations(
        predictions,
        task_names=["log_b", "log_g"],
        representation="z",
        reducer_name="pca",
        reducer_params={"n_components": 2, "random_state": 0},
        plot_params={
            "figsize": [4, 3],
            "dpi": 80,
            "point_size": 12,
            "alpha": 0.8,
            "cmap": "viridis",
        },
        output_dir=tmp_path / "embed_vis",
        save_format="png",
    )

    assert (tmp_path / "embed_vis" / "embedding_table.csv").exists()
    assert (tmp_path / "embed_vis" / "plot_spec.yaml").exists()
    assert len(result["figures"]) == 2
    assert (tmp_path / "embed_vis" / "figures" / "z_pca_color_log_b.png").exists()
    assert (tmp_path / "embed_vis" / "figures" / "z_pca_color_log_g.png").exists()


def test_generate_embedding_visualizations_supports_node_emb_mean(tmp_path):
    predictions = {
        "id": ["mp-1", "mp-2", "mp-3"],
        "target_log_g": [0.5, 0.6, 0.7],
        "pred_log_g": [0.48, 0.59, 0.72],
        "node_emb": [
            json.dumps([[0.0, 0.2], [0.2, 0.4]]),
            json.dumps([[0.4, 0.6], [0.6, 0.8]]),
            json.dumps([[0.8, 1.0], [1.0, 1.2]]),
        ],
    }

    result = generate_embedding_visualizations(
        predictions,
        task_names=["log_g"],
        representation="node_emb",
        reducer_name="pca",
        reducer_params={"n_components": 2, "random_state": 0},
        plot_params={"figsize": [4, 3], "dpi": 80},
        output_dir=tmp_path / "node_embed_vis",
        save_format="png",
        node_reduction="mean",
    )

    assert len(result["figures"]) == 1
    assert (
        tmp_path / "node_embed_vis" / "figures" / "node_emb_pca_color_log_g.png"
    ).exists()


def test_replot_embedding_visualizations_reuses_saved_spec(tmp_path):
    predictions = {
        "id": ["mp-1", "mp-2", "mp-3", "mp-4"],
        "target_log_g": [0.5, 0.6, 0.7, 0.8],
        "pred_log_g": [0.48, 0.61, 0.69, 0.81],
        "graph_emb": [
            json.dumps([0.0, 0.1, 0.2]),
            json.dumps([0.2, 0.3, 0.4]),
            json.dumps([0.4, 0.5, 0.6]),
            json.dumps([0.6, 0.7, 0.8]),
        ],
    }

    generate_embedding_visualizations(
        predictions,
        task_names=["log_g"],
        representation="graph_emb",
        reducer_name="pca",
        reducer_params={"n_components": 2, "random_state": 0},
        plot_params={"figsize": [4, 3], "dpi": 80},
        output_dir=tmp_path / "base",
        save_format="png",
    )

    result = replot_embedding_visualizations(
        plot_spec_path=tmp_path / "base" / "plot_spec.yaml",
        output_dir=tmp_path / "replot",
    )

    assert len(result["figures"]) == 1
    assert (
        tmp_path / "replot" / "figures" / "graph_emb_pca_color_log_g.png"
    ).exists()


def test_run_embedding_visualization_loads_checkpoint_via_task_setup(monkeypatch, tmp_path):
    class DummyTrainer:
        def __init__(self):
            self.test_loader = object()
            self.loaded_checkpoint = None
            self.closed = False

        def load_checkpoint(self, checkpoint):
            self.loaded_checkpoint = checkpoint

        def predict(self, *args, **kwargs):
            return {
                "id": ["mp-1", "mp-2", "mp-3"],
                "target_log_g": [0.1, 0.2, 0.3],
                "pred_log_g": [0.11, 0.19, 0.31],
                "z": [
                    json.dumps([0.0, 0.1]),
                    json.dumps([0.2, 0.3]),
                    json.dumps([0.4, 0.5]),
                ],
            }

        def close_datasets(self):
            self.closed = True

    class DummyTask:
        def setup(self, trainer):
            trainer.load_checkpoint("dummy.ckpt")

    monkeypatch.setattr(
        "matpropnet.tasks.core._load_runtime_config",
        lambda *args, **kwargs: {
            "checkpoint": "dummy.ckpt",
            "run_dir": str(tmp_path),
            "hide_eval_progressbar": False,
            "task": {
                "tasks": {
                    "log_g": {"type": "regression"},
                },
                "predict": {},
            },
            "dataset": {"test": {"src": "dummy.lmdb"}},
        },
    )
    trainer = DummyTrainer()
    monkeypatch.setattr("matpropnet.tasks.core._build_trainer", lambda config: trainer)
    monkeypatch.setattr(
        "matpropnet.tasks.core._build_task",
        lambda config, trainer_obj: DummyTask().setup(trainer_obj),
    )

    result = run_embedding_visualization(
        "dummy.yml",
        checkpoint="dummy.ckpt",
        output_dir=tmp_path / "embed_cli",
        representation="z",
        reducer="pca",
    )

    assert trainer.loaded_checkpoint == "dummy.ckpt"
    assert trainer.closed is True
    assert (tmp_path / "embed_cli" / "plot_spec.yaml").exists()
    assert len(result["figures"]) == 1

from __future__ import annotations

import csv
import math

from matpropnet.ensemble import aggregate_ensemble_predictions
from matpropnet.cli.ensemble_aggregate import main as ensemble_aggregate_main


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_ensemble_aggregate_decomposes_uncertainty(tmp_path):
    member0 = tmp_path / "member0.csv"
    member1 = tmp_path / "member1.csv"
    _write_csv(
        member0,
        [
            {"id": "a", "pred_H": "1.0", "pred_H_sigma": "0.5", "target_H": "1.2"},
            {"id": "b", "pred_H": "2.0", "pred_H_sigma": "0.25", "target_H": "1.8"},
        ],
    )
    _write_csv(
        member1,
        [
            {"id": "a", "pred_H": "3.0", "pred_H_sigma": "1.0", "target_H": "1.2"},
            {"id": "b", "pred_H": "4.0", "pred_H_sigma": "0.75", "target_H": "1.8"},
        ],
    )

    rows = aggregate_ensemble_predictions(
        [member0, member1], output_path=tmp_path / "ensemble.csv"
    )

    assert (tmp_path / "ensemble.csv").exists()
    assert rows[0]["pred_H_mean"] == 2.0
    assert math.isclose(rows[0]["pred_H_var_epistemic"], 1.0)
    assert math.isclose(rows[0]["pred_H_var_aleatoric"], 0.625)
    assert math.isclose(rows[0]["pred_H_var_total"], 1.625)


def test_ensemble_aggregate_supports_deterministic_members(tmp_path):
    member0 = tmp_path / "member0.csv"
    member1 = tmp_path / "member1.csv"
    _write_csv(member0, [{"id": "a", "pred_H": "1.0"}])
    _write_csv(member1, [{"id": "a", "pred_H": "3.0"}])

    rows = aggregate_ensemble_predictions([member0, member1])

    assert rows[0]["pred_H_mean"] == 2.0
    assert rows[0]["pred_H_var_aleatoric"] == 0.0
    assert rows[0]["pred_H_std_total"] == rows[0]["pred_H_std_epistemic"]


def test_ensemble_aggregate_cli(tmp_path):
    member0 = tmp_path / "member0.csv"
    member1 = tmp_path / "member1.csv"
    out = tmp_path / "ensemble.csv"
    _write_csv(member0, [{"id": "a", "pred_H": "1.0"}])
    _write_csv(member1, [{"id": "a", "pred_H": "3.0"}])

    exit_code = ensemble_aggregate_main(
        [
            "--predictions",
            str(member0),
            str(member1),
            "--out",
            str(out),
            "--tasks",
            "H",
        ]
    )

    assert exit_code == 0
    assert out.exists()

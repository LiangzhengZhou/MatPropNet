import csv
import json
from pathlib import Path

from matpropnet.preprocessing import parse_args, run_preprocess

from tests.helpers import SAMPLE_CIF


def test_preprocess_handles_bom_csv_and_writes_lmdb(tmp_path):
    csv_path = tmp_path / "all.csv"
    out_root = tmp_path / "data"
    content = (
        "\ufeffid,target1,cif,split\n"
        f'mp-1,1.0,"{SAMPLE_CIF}",train\n'
        f'mp-2,2.0,"{SAMPLE_CIF}",val\n'
        f'mp-3,3.0,"{SAMPLE_CIF}",test\n'
    )
    csv_path.write_text(content, encoding="utf-8")

    args = parse_args(
        [
            "--csv",
            str(csv_path),
            "--out-root",
            str(out_root),
            "--target-columns",
            "target1",
            "--task-types",
            "regression",
            "--id-column",
            "id",
            "--cif-column",
            "cif",
            "--cif-mode",
            "inline",
            "--split-column",
            "split",
            "--get-edges",
        ]
    )
    summary = run_preprocess(args)

    assert (out_root / "train" / "data.lmdb").exists()
    assert summary["train"]["num_samples"] == 1
    assert summary["val"]["num_samples"] == 1
    assert summary["test"]["num_samples"] == 1


def test_preprocess_skip_failed_writes_failure_manifest(tmp_path):
    csv_path = tmp_path / "all.csv"
    out_root = tmp_path / "data"
    bad_cif = "not a valid cif"
    content = (
        "id,target1,cif,split\n"
        f'mp-1,1.0,"{SAMPLE_CIF}",train\n'
        f'mp-2,2.0,"{bad_cif}",train\n'
    )
    csv_path.write_text(content, encoding="utf-8")

    args = parse_args(
        [
            "--csv",
            str(csv_path),
            "--out-root",
            str(out_root),
            "--target-columns",
            "target1",
            "--task-types",
            "regression",
            "--id-column",
            "id",
            "--cif-column",
            "cif",
            "--cif-mode",
            "inline",
            "--split-column",
            "split",
            "--get-edges",
            "--skip-failed",
        ]
    )
    summary = run_preprocess(args)

    assert summary["train"]["num_samples"] == 1
    assert summary["train"]["num_failed"] == 1
    failed_manifest = out_root / "train" / "failed_samples.json"
    assert failed_manifest.exists()
    failures = json.loads(failed_manifest.read_text(encoding="utf-8"))
    assert failures[0]["sample_id"] == "mp-2"


def test_preprocess_rejects_invalid_task_type(tmp_path):
    csv_path = tmp_path / "all.csv"
    csv_path.write_text(
        f'id,target1,cif\nmp-1,1.0,"{SAMPLE_CIF}"\n',
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--csv",
            str(csv_path),
            "--out-path",
            str(tmp_path / "data"),
            "--target-columns",
            "target1",
            "--task-types",
            "classfication",
            "--cif-mode",
            "inline",
        ]
    )

    try:
        run_preprocess(args)
    except SystemExit as exc:
        assert "Unsupported task type" in str(exc)
    else:
        raise AssertionError("Expected invalid task type to raise SystemExit.")


def test_random_split_rejects_invalid_fractions():
    from matpropnet.preprocessing.property_csv import random_split

    rows = [{"id": "a"}, {"id": "b"}]

    for fractions in ([0.0, 0.0, 0.0], [0.8, -0.1, 0.3], [1.0, 0.0]):
        try:
            random_split(rows, fractions, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid fractions to raise ValueError.")

import csv
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

import logging

from matpropnet.utils.runtime import setup_runtime_logging


def test_runtime_logging_writes_file(tmp_path):
    log_file = tmp_path / "matpropnet.log"
    setup_runtime_logging(level="DEBUG", log_file=str(log_file), force=True)
    logging.getLogger("matpropnet.test").info("hello logging")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file.exists()
    assert "hello logging" in log_file.read_text(encoding="utf-8")

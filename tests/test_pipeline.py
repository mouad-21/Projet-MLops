import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import build_pipeline
from config import load_config


def test_build_pipeline_logreg():
    pipe = build_pipeline(["a", "b"], [], "logreg")
    assert "pre" in dict(pipe.named_steps)
    assert "model" in dict(pipe.named_steps)


def test_build_pipeline_random_forest():
    pipe = build_pipeline(["a", "b"], ["c"], "random_forest")
    assert "pre" in dict(pipe.named_steps)
    assert "model" in dict(pipe.named_steps)


def test_build_pipeline_invalid_model_type():
    try:
        build_pipeline(["a"], [], "not_a_model")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_load_config():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml"))
    assert cfg.data.target == "Class"
    assert cfg.model.type in ("logreg", "random_forest")
    assert len(cfg.features.numeric) > 0

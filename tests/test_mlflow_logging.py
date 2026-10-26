from types import SimpleNamespace

import pytest

from follower import mlflow_logging as ml
from follower.training_config import Experiment


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path / "mlflow.db"}')
    return ml.client()


class Writer:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step=None):
        self.calls.append((tag, value, step))


def test_flatten_uses_dotted_keys():
    flat = ml.flatten(Experiment().dict())
    assert flat['environment.grid_config.num_agents'] == 64
    assert flat['use_mlflow'] is True


def test_training_run_params_metrics_and_eval(store, monkeypatch):
    monkeypatch.setattr(ml, 'snapshot_code', lambda dst: None)
    exp = Experiment(experiment='unit_s3', seed=3, mlflow_experiment='unit')
    run_id = ml.start_training_run(exp, SimpleNamespace(seed=3))
    assert run_id is not None
    run = store.get_run(run_id)
    assert run.data.params['seed'] == '3' and run.data.tags['group'] == 'unit'

    observer = ml.MlflowObserver(run_id)
    writer = Writer()
    observer.wrap(writer)
    writer.add_scalar('train/loss', 0.5, 100)
    writer.add_scalar('train/loss', 0.25, 200)
    assert writer.calls == [('train/loss', 0.5, 100), ('train/loss', 0.25, 200)]
    observer.flush()
    history = store.get_metric_history(run_id, 'train/loss')
    assert [(m.step, m.value) for m in history] == [(100, 0.5), (200, 0.25)]

    ml.log_eval('unit_s3', 'shield', {'02-mazes': {'avg_throughput': 1.5}}, [])
    assert store.get_run(run_id).data.metrics['eval/shield/02-mazes/avg_throughput'] == 1.5


def test_failures_do_not_propagate(store, monkeypatch):
    observer = ml.MlflowObserver('no-such-run')
    writer = Writer()
    observer.wrap(writer)
    writer.add_scalar('train/loss', 1.0, 1)
    observer.flush()
    assert len(observer.buffer) == 1, 'unsent metrics are kept for the next flush'
    ml.log_eval('missing_model', 'default', {}, [])

    def broken():
        raise RuntimeError('store down')
    monkeypatch.setattr(ml, 'client', broken)
    assert ml.start_training_run(Experiment(), SimpleNamespace()) is None
    ml.finish_training_run('x', observer, None, 0)

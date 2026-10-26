"""MLflow tracking for training (main.py), evaluation (eval_lite.py) and the cross-seed summary (aggregate.py).

One run per training: parameters, the code as it was (tracked and untracked files plus the launching .sh script),
every scalar sample_factory writes to tensorboard, the checkpoints and an ONNX export. eval_lite.py adds the
per-map throughput to the same run, found by its 'sf_experiment' tag (the newest run of that name).

Every entry point catches its own exceptions: a broken tracking store never stops training or evaluation.
Store: $MLFLOW_TRACKING_URI, by default the server `mlflow server` (started in mlruns/) serves on 127.0.0.1:5000.
That server is a newer MLflow than this env's and upgraded mlruns/mlflow.db to a schema this env can't open, so
log through the server rather than to the database file. A sqlite:/// URI still works for a separate database
(the tests use one); it keeps the artifacts in artifacts/<experiment> next to the database.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.utils import log

ROOT = Path(__file__).resolve().parent.parent
# what a training run executes: the entry point, the model/env/planner code and the map definitions
CODE_DIRS = ('follower/', 'follower_cpp/', 'utils/', 'env/')
CODE_FILES = {'main.py'}
CODE_SUFFIXES = {'.py', '.h', '.hpp', '.cpp', '.yaml'}
FLUSH_INTERVAL_SEC = 30
MAX_METRICS_PER_BATCH = 1000


def tracking_uri():
    return os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')


def client():
    import mlflow
    from mlflow.tracking import MlflowClient
    uri = tracking_uri()
    if uri.startswith('sqlite:///'):
        Path(uri[len('sqlite:///'):]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(uri)


def experiment_id(mlflow_client, name):
    """Get or create; a parallel run may create the same experiment first, so retry the lookup once."""
    for attempt in range(2):
        exp = mlflow_client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        try:
            location = None
            if tracking_uri().startswith('sqlite:///'):
                location = (Path(tracking_uri()[len('sqlite:///'):]).parent / 'artifacts' / name).as_uri()
            return mlflow_client.create_experiment(name, artifact_location=location)
        except Exception:
            if attempt:
                raise
            time.sleep(1)


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def metric_key(tag):
    """MLflow keys allow alphanumerics, '_', '-', '.', ' ', '/'."""
    return re.sub(r'[^\w\-. /]', '_', tag)


def find_run_id(mlflow_client, sf_experiment):
    """The newest training run of this experiment name, or None."""
    runs = mlflow_client.search_runs(
        [e.experiment_id for e in mlflow_client.search_experiments()],
        filter_string=f"tags.sf_experiment = '{sf_experiment}'", order_by=['attributes.start_time DESC'],
        max_results=1)
    return runs[0].info.run_id if runs else None


def git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=30).decode()
    except Exception:
        return ''


def launcher_script():
    """The nearest ancestor process running a .sh script (e.g. `bash run_mem.sh`), as an absolute path."""
    pid = os.getppid()
    while pid > 1:
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            cwd = Path(os.readlink(f'/proc/{pid}/cwd'))
            for arg in args[1:]:
                if arg.endswith('.sh') and (cwd / arg).is_file():
                    return cwd / arg
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except Exception:
            return None
    return None


def snapshot_code(dst: Path):
    """The training code (tracked and untracked, not ignored), the diff to HEAD and the launching script."""
    for name in git('ls-files', '-co', '--exclude-standard').splitlines():
        src = ROOT / name
        if (name in CODE_FILES or name.startswith(CODE_DIRS)) and src.suffix in CODE_SUFFIXES and src.is_file():
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / name)
    (dst / 'git.diff').write_text(git('diff', 'HEAD'))
    script = launcher_script()
    if script is not None:
        shutil.copy2(script, dst / 'launcher.sh')
    return script


def environment_tags():
    from importlib.metadata import version
    import torch
    tags = {'host': socket.gethostname(), 'python': sys.version.split()[0], 'torch': torch.__version__,
            'sample_factory': version('sample-factory'), 'pogema': version('pogema'),
            'cuda': str(torch.version.cuda), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    if torch.cuda.is_available():
        tags['gpu'] = torch.cuda.get_device_name(0)
    return tags


def start_training_run(exp, cfg):
    """-> run id, or None when MLflow is unavailable."""
    try:
        from mlflow.entities import Param, RunTag
        mlflow_client = client()
        commit = git('rev-parse', 'HEAD').strip()
        tags = {'sf_experiment': exp.experiment, 'seed': str(exp.seed),
                'group': re.sub(r'_s\d+$', '', exp.experiment), 'git_commit': commit,
                'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD').strip(),
                'git_dirty': str(bool(git('status', '--porcelain').strip())),
                'command': ' '.join(sys.argv), 'mlflow.source.git.commit': commit, **environment_tags()}
        run = mlflow_client.create_run(experiment_id(mlflow_client, exp.mlflow_experiment), run_name=exp.experiment,
                                       tags=tags)
        run_id = run.info.run_id
        params = [Param(k, str(v)[:6000]) for k, v in flatten(exp.dict()).items()]
        for i in range(0, len(params), 100):
            mlflow_client.log_batch(run_id, params=params[i:i + 100])

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'code').mkdir()
            script = snapshot_code(tmp / 'code')
            if script is not None:
                mlflow_client.log_batch(run_id, tags=[RunTag('launcher', str(script))])
            (tmp / 'env').mkdir()
            (tmp / 'env' / 'requirements.txt').write_text(subprocess.check_output(
                [sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL, timeout=120).decode())
            (tmp / 'sf_config.json').write_text(json.dumps(cfg if isinstance(cfg, dict) else vars(cfg), indent=2, default=str))
            mlflow_client.log_artifacts(run_id, str(tmp))
        log.info(f'MLflow run {run_id} ({tracking_uri()})')
        return run_id
    except Exception as e:
        log.warning(f'MLflow disabled for this run: {e!r}')
        return None


class MlflowObserver(AlgoObserver):
    """Forwards every scalar written to the runner's tensorboard writers to MLflow, in batches."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.buffer = []
        self.last_flush = time.time()

    def on_start(self, runner: Runner) -> None:
        for writer in runner.writers.values():
            self.wrap(writer)

    def wrap(self, writer):
        add_scalar = writer.add_scalar

        def tee(tag, scalar_value, global_step=None, *args, **kwargs):
            add_scalar(tag, scalar_value, global_step, *args, **kwargs)
            try:
                from mlflow.entities import Metric
                self.buffer.append(Metric(metric_key(tag), float(scalar_value), int(time.time() * 1000),
                                          int(global_step or 0)))
            except Exception:
                pass

        writer.add_scalar = tee

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        if time.time() - self.last_flush > FLUSH_INTERVAL_SEC:
            self.flush()

    def on_stop(self, runner: Runner) -> None:
        self.flush()

    def flush(self):
        """On failure (e.g. sqlite locked by a parallel run) the metrics stay buffered for the next flush."""
        self.last_flush = time.time()
        try:
            mlflow_client = client()
            while self.buffer:
                mlflow_client.log_batch(self.run_id, metrics=self.buffer[:MAX_METRICS_PER_BATCH])
                del self.buffer[:MAX_METRICS_PER_BATCH]
        except Exception as e:
            log.warning(f'MLflow metric flush failed, {len(self.buffer)} kept for retry: {e!r}')


def finish_training_run(run_id, observer, runner, status):
    """After runner.run(): sample_factory saves the final checkpoint only after AlgoObserver.on_stop."""
    if run_id is None:
        return
    try:
        observer.flush()
        mlflow_client = client()
        exp_dir = Path(runner.cfg.train_dir) / runner.cfg.experiment
        metrics = {'train_wall_time_sec': time.time() - runner.start_time,
                   'final_env_steps': sum(runner.env_steps.values())}
        best = [float(m.group(1)) for p in exp_dir.glob('checkpoint_p0/best_*.pth')
                if (m := re.search(r'_([-\d.]+)\.pth$', p.name))]
        if best:
            metrics['best_train_avg_throughput'] = max(best)
        for name in ('checkpoint_p0', '.summary'):
            if (exp_dir / name).is_dir():
                mlflow_client.log_artifacts(run_id, str(exp_dir / name), name.lstrip('.'))
        for name in ('config.json', 'sf_log.txt', 'git.diff'):
            if (exp_dir / name).is_file():
                mlflow_client.log_artifact(run_id, str(exp_dir / name), 'train_dir')
        if status == ExperimentStatus.SUCCESS:
            metrics['num_params'] = export_onnx(mlflow_client, run_id, exp_dir)
        for k, v in metrics.items():
            mlflow_client.log_metric(run_id, k, float(v))
    except Exception as e:
        log.warning(f'MLflow final logging failed: {e!r}')
    finally:
        try:
            names = {ExperimentStatus.SUCCESS: 'SUCCESS', ExperimentStatus.FAILURE: 'FAILURE',
                     ExperimentStatus.INTERRUPTED: 'INTERRUPTED'}
            mlflow_client = client()
            mlflow_client.set_tag(run_id, 'status', names.get(status, str(status)))
            mlflow_client.set_terminated(run_id, {ExperimentStatus.SUCCESS: 'FINISHED',
                                                  ExperimentStatus.INTERRUPTED: 'KILLED'}.get(status, 'FAILED'))
        except Exception as e:
            log.warning(f'MLflow could not close run {run_id}: {e!r}')


def export_onnx(mlflow_client, run_id, exp_dir):
    """The latest checkpoint as ONNX (as export_onnx.py does), -> number of parameters."""
    from follower.inference import FollowerInference, FollowerInferenceConfig
    algo = FollowerInference(FollowerInferenceConfig(path_to_weights=str(exp_dir), device='cpu'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f'{exp_dir.name}.onnx'
        algo.to_onnx(str(path))
        mlflow_client.log_artifact(run_id, str(path), 'model')
    return algo.get_model_parameters()


def log_eval(model_name, variant, per_map, files):
    """per_map: {map folder: {metric: mean over the grid}}; files: [(local path, artifact dir)]."""
    try:
        mlflow_client = client()
        run_id = find_run_id(mlflow_client, model_name)
        if run_id is None:
            log.warning(f'MLflow: no training run tagged sf_experiment={model_name}, evaluation not logged')
            return
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        metrics = [Metric(metric_key(f'eval/{variant}/{folder}/{k}'), float(v), now, 0)
                   for folder, values in per_map.items() for k, v in values.items()]
        mlflow_client.log_batch(run_id, metrics=metrics)
        for path, artifact_dir in files:
            mlflow_client.log_artifact(run_id, str(path), artifact_dir)
        log.info(f'MLflow: evaluation {variant} logged to run {run_id}')
    except Exception as e:
        log.warning(f'MLflow evaluation logging failed: {e!r}')


def log_aggregate(groups, metrics, summary_text, verdicts):
    """groups: {label: [model names]}; metrics: {key: value}; verdicts: {'<b>-<a>/<map>': 'better'|'worse'|...}."""
    try:
        mlflow_client = client()
        members = {n: find_run_id(mlflow_client, n) for names in groups.values() for n in names}
        tags = {'member_run_ids': json.dumps(members)[:5000],
                **{metric_key(f'verdict/{k}'): v for k, v in verdicts.items()}}
        run = mlflow_client.create_run(experiment_id(mlflow_client, 'FollowerLite-aggregate'),
                                       run_name='-vs-'.join(groups), tags=tags)
        run_id = run.info.run_id
        for label, names in groups.items():
            mlflow_client.log_param(run_id, f'group/{label}', ','.join(names)[:6000])
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        batch = [Metric(metric_key(k), float(v), now, 0) for k, v in metrics.items()]
        for i in range(0, len(batch), MAX_METRICS_PER_BATCH):
            mlflow_client.log_batch(run_id, metrics=batch[i:i + MAX_METRICS_PER_BATCH])
        mlflow_client.log_text(run_id, summary_text, 'summary.txt')
        mlflow_client.set_terminated(run_id)
        print(f'MLflow aggregate run {run_id}')
    except Exception as e:
        log.warning(f'MLflow aggregate logging failed: {e!r}')

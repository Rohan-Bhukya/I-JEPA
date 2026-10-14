
import os
import shlex
import sys
import modal

GPU = os.environ.get('IJEPA_GPU', 'A100-80GB')
APP_DIR = '/root/ijepa'
HERE = os.path.dirname(os.path.abspath(__file__))

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install_from_requirements(os.path.join(HERE, 'requirements.txt'))
         .add_local_dir(HERE, remote_path=APP_DIR, ignore=['data', 'runs', '**/__pycache__', '.git']))
app = modal.App('ijepa', image=image)
data_volume = modal.Volume.from_name('ijepa-data', create_if_missing=True, version=2)
runs_volume = modal.Volume.from_name('ijepa-runs', create_if_missing=True, version=2)
VOLUMES = {f'{APP_DIR}/data': data_volume, f'{APP_DIR}/runs': runs_volume}


def _enter_app_dir():
    os.chdir(APP_DIR)
    sys.path.insert(0, APP_DIR)


@app.function(gpu=GPU, cpu=16, memory=65536, volumes=VOLUMES, timeout=24 * 3600)
def train(args: str = ''):
    """Up to 24 h per call; run again with --resume to continue a longer training."""
    _enter_app_dir()
    import main as main_module
    main_module.main(shlex.split(args), on_epoch_end=runs_volume.commit)   # save each checkpoint right away
    runs_volume.commit()


@app.function(gpu=GPU, cpu=16, memory=65536, volumes=VOLUMES, timeout=24 * 3600)
def evaluate(args: str = ''):
    _enter_app_dir()
    import evaluate as evaluate_module
    return evaluate_module.main(['--num-workers', '14'] + shlex.split(args))   # your own --num-workers wins

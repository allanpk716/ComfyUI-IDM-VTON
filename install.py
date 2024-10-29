import sys
import os
import subprocess
from huggingface_hub import snapshot_download

sys.path.append("../../")
from folder_paths import models_dir

CUSTOM_NODES_PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(models_dir, "IDM-VTON")
WEIGHTS_PATH_VAE = os.path.join(models_dir, "sdxl-vae-fp16-fix")

HF_REPO_ID = "yisol/IDM-VTON"
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

os.makedirs(WEIGHTS_PATH, exist_ok=True)
os.makedirs(WEIGHTS_PATH_VAE, exist_ok=True)

def build_pip_install_cmds(args):
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        return [sys.executable, '-s', '-m', 'pip', 'install'] + args
    else:
        return [sys.executable, '-m', 'pip', 'install'] + args

def ensure_package():
    cmds = build_pip_install_cmds(['-r', 'requirements.txt'])
    subprocess.run(cmds, cwd=CUSTOM_NODES_PATH)


if __name__ == "__main__":
    ensure_package()
    snapshot_download(repo_id=HF_REPO_ID, local_dir=WEIGHTS_PATH, local_dir_use_symlinks=False)
    snapshot_download(repo_id=vae_model_id, local_dir=WEIGHTS_PATH_VAE, local_dir_use_symlinks=False)

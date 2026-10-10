"""Fine-tune the landmark model in a Kaggle notebook session.

Attach two private datasets to the notebook: the CAT crop cache (`python -m catlandmarks.prepare cat`) and this
package's `catlandmarks/` folder; for the MAE-initialised run, also a dataset holding `encoder.pt`. The script finds
them under /kaggle/input wherever Kaggle mounts them, trains, and leaves the run folder in /kaggle/working, which
Kaggle keeps as the notebook's output.

Settings are constants rather than arguments because Kaggle runs scripts without a command line.
"""

import glob
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

EPOCHS = 100
RUN_NAME = "pose_scratch"           # "pose_mae" for the MAE-initialised run
USE_MAE_ENCODER = False             # True: start from the attached encoder.pt

code = Path("/kaggle/working/src")
code.mkdir(parents=True, exist_ok=True)
for tar in glob.glob("/kaggle/input/**/catlandmarks.tar", recursive=True):   # a folder uploaded with --dir-mode tar
    with tarfile.open(tar) as t:
        t.extractall(code)
for init_file in glob.glob("/kaggle/input/**/catlandmarks/__init__.py", recursive=True):
    shutil.copytree(Path(init_file).parent, code / "catlandmarks", dirs_exist_ok=True)
sys.path.insert(0, str(code))

import torch  # noqa: E402

from catlandmarks.finetune import FinetuneConfig, evaluate, train  # noqa: E402

cache = Path(glob.glob("/kaggle/input/**/images.npy", recursive=True)[0]).parent
encoder = Path(glob.glob("/kaggle/input/**/encoder.pt", recursive=True)[0]) if USE_MAE_ENCODER else None
out = Path("/kaggle/working") / RUN_NAME
print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cache": str(cache),
                  "encoder": str(encoder)}), flush=True)

cfg = FinetuneConfig(cache=str(cache), epochs=EPOCHS)
start = time.time()
train(cfg, out, encoder)
print(json.dumps({"epochs": EPOCHS, "minutes": round((time.time() - start) / 60, 1),
                  "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}), flush=True)
print(json.dumps(evaluate(out, "val", cfg)), flush=True)

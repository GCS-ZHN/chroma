#!/usr/bin/env python3
#SBATCH -J chroma_fold_design
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o chroma_fold_design_%j.log
#SBATCH -e chroma_fold_design_%j.err
import appdirs
import os
from chroma import Chroma, conditioners
from chroma.models import graph_classifier
from pathlib import Path

out_dir = Path.cwd()
TEST_TASK_ID=os.environ.get('SLURM_JOBID', os.getpid())
print("Test task id: ", TEST_TASK_ID)
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
device = 'cuda:0'
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt",
    device=device
)
CATH = "3.40.50"  # @param {type:"string"}
length = 130  # @param {type:"slider", min:50, max:250, step:10}

proclass_model = graph_classifier.load_model(
    local_model_dir / "chroma_proclass_v1.0.pt",
    device=device)
conditioner = conditioners.ProClassConditioner("cath", CATH, model=proclass_model)
cath_conditioned_protein = chroma.sample(
    conditioner=conditioner,
    chain_lengths=[length]
)
cath_conditioned_protein.to(str(out_dir / f"fold_design_{TEST_TASK_ID}.pdb"))


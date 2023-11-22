#!/usr/bin/env python3
#SBATCH -J chroma_procap_design
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o chroma_procap_design_%j.log
#SBATCH -e chroma_procap_design_%j.err
import appdirs
import appdirs
import os
from chroma import Chroma, conditioners
from chroma.models import procap
from pathlib import Path

out_dir = Path.cwd()
SLURM_JOBID=os.environ.get('SLURM_JOBID', os.getpid())
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
device = 'cuda:0'
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt",
    device=device
)
length = 128  # @param {type:"slider", min:50, max:250, step:10}
caption = "Crystal structure of nanobody (VHH)"  # @param {type:"string"}

procap_model = procap.load_model(
    local_model_dir / "chroma_procap_v1.0.pt",
    device=device)
conditioner = conditioners.ProCapConditioner(caption, -1, model=procap_model)
protein = chroma.sample(
    steps=200, chain_lengths=[length], conditioner=conditioner
)
protein.to(str(out_dir / f"procap_design_{SLURM_JOBID}.pdb"))


#!/usr/bin/env python3
#SBATCH -J chroma_partial_resample
#SBATCH -c 8
#SBATCH -o chroma_partial_resample_%j.log
#SBATCH -e chroma_partial_resample_%j.err
import appdirs
import os
import torch
from chroma import Chroma, Protein, conditioners
from pathlib import Path

out_dir = Path.cwd()
TEST_TASK_ID=os.environ.get('SLURM_JOBID', os.getpid())
print("Test task id: ", TEST_TASK_ID)
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
protein = Protein("FC-III-AAPC.pdb", device=device)
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt"
)

conditioner = conditioners.SubstructureConditioner(
    protein,
    chroma.backbone_network,
    selection="not chain E").to(device)

protein = chroma.sample(
    protein_init=protein,
    conditioner=conditioner,
    design_selection="chain E",
    langevin_factor=1,
    inverse_temperature=8,
    sde_func='langevin',
)
protein.to(str(out_dir / f"partial_resample_{TEST_TASK_ID}.pdb"))

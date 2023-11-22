#!/usr/bin/env python3
#SBATCH -J chroma_conditional_monomer
#SBATCH -c 8
#SBATCH -o chroma_conditional_monomer_%j.log
#SBATCH -e chroma_conditional_monomer_%j.err
import appdirs
import os
from chroma import Chroma, conditioners
from pathlib import Path

out_dir = Path.cwd()
SLURM_JOBID=os.environ.get("SLURM_JOBID", "test")
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
conditioner = conditioners.SymmetryConditioner(G="C_3", num_chain_neighbors=2)
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt"
)
protein = chroma.sample(
    chain_lengths=[100],
    conditioner=conditioner,
    langevin_factor=8,
    inverse_temperature=8,
    sde_func='langevin',
    potts_symmetry_order=conditioner.potts_symmetry_order
    )
protein.to(str(out_dir / f"conditional_monomer_{SLURM_JOBID}.pdb"))

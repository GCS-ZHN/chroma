#!/usr/bin/env python3
#SBATCH -J chroma_full_redesign
#SBATCH -c 8
#SBATCH -o chroma_full_redesignr_%j.log
#SBATCH -e chroma_full_redesign_%j.err
import appdirs
import os
from chroma import Chroma, Protein
from pathlib import Path

out_dir = Path.cwd()
SLURM_JOBID=os.environ.get('SLURM_JOBID', os.getpid())
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
protein = Protein("FC-III-AAPC.pdb", device="cuda:0")
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt"
)
protein = chroma.design(protein)
protein.to(str(out_dir / f"full_redesign_{SLURM_JOBID}.pdb"))

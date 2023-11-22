#!/usr/bin/env python3
#SBATCH -J ChromaComplex
#SBATCH -c 8
#SBATCH -o complex_test_%j.log
#SBATCH -e complex_test_%j.err
import appdirs
import os
from chroma import Chroma
from pathlib import Path

out_dir = Path.cwd()
SLURM_JOBID=os.environ.get('SLURM_JOBID', os.getpid())
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt"
)
protein = chroma.sample(chain_lengths=[100, 200])
protein.to(str(out_dir / f"unconditional_complex_{SLURM_JOBID}.pdb"))

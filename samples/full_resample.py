#!/usr/bin/env python3
#SBATCH -J chroma_full_resample
#SBATCH -c 8
#SBATCH -o chroma_full_resample_%j.log
#SBATCH -e chroma_full_resample_%j.err
import appdirs
import os
from chroma import Chroma, Protein
from pathlib import Path

out_dir = Path.cwd()
TEST_TASK_ID=os.environ.get('SLURM_JOBID', os.getpid())
print("Test task id: ", TEST_TASK_ID)
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
protein = Protein("FC-III-AAPC.pdb", device="cuda:0")
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt"
)
protein = chroma.sample(
    protein_init=protein
    )
protein.to(str(out_dir / f"full_resample_{TEST_TASK_ID}.pdb"))

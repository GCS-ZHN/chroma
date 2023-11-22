#!/usr/bin/env python3
#SBATCH -J chroma_shape_design
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o chroma_shape_design_%j.log
#SBATCH -e chroma_shape_design_%j.err
import appdirs
import appdirs
import os
from chroma import Chroma, conditioners
from chroma.utility.chroma import letter_to_point_cloud
from pathlib import Path

out_dir = Path.cwd()
SLURM_JOBID=os.environ.get('SLURM_JOBID', 'test')
local_model_dir = Path(appdirs.user_cache_dir("chroma/weights"))
device = 'cuda:0'
chroma = Chroma(
    weights_backbone=local_model_dir / "chroma_backbone_v1.0.pt",
    weights_design=local_model_dir / "chroma_design_v1.0.pt",
    device=device
)
character = "G"  # @param {type:"string"}
if len(character) > 1:
    character = character[:1]
    print(f"Keeping only first character ({character})!")
length = 1000  # @param {type:"slider", min:100, max:1500, step:100}

letter_point_cloud = letter_to_point_cloud(character)
conditioner = conditioners.ShapeConditioner(
    letter_point_cloud,
    chroma.backbone_network.noise_schedule,
    autoscale_num_residues=length,
).to(device)

shaped_protein = chroma.sample(
    chain_lengths=[length], conditioner=conditioner
)

shaped_protein.to(str(out_dir / f"shape_design_{SLURM_JOBID}.pdb"))


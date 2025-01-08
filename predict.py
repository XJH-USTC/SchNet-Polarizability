import re
from ase import io
import numpy as np
from schnetpack.data import ASEAtomsData,AtomsDataModule
from schnetpack.transform import ASENeighborList
import os
from ase.db import connect
import os
import schnetpack as spk
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl    

dbfile = os.path.join('.','qm7b.db')
db = connect(dbfile)
n = len(db)
model = torch.load(os.path.join('qm7tut','best_inference_model'),map_location='cpu')
dataset = AtomsDataModule(
    datapath=dbfile,
    batch_size=1,
    num_train=n-1500,
    num_val=1000,
    num_test=500,
    transforms=[
    trn.ASENeighborList(cutoff=5.),
    trn.CastTo32()
    ]
)
dataset.prepare_data()
dataset.setup()
for batch in dataset.test_dataloader():
    result = model(batch)
    print("predict results:", result['ccsd_pol'])
    print('target results',batch['ccsd_pol'])

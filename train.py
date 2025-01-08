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

def main():
    atoms_list, property_list= read_qm7b()
    ml(atoms_list,property_list)
def trans(array=None):
    a = np.empty((3,3))
    a[0,0] = array[0]
    a[1,1] = array[1]
    a[2,2] = array[2]
    a[0,1] = a[1,0] = array[3]
    a[0,2] = a[2,0] = array[4]
    a[1,2] = a[2,1] = array[5]
    return a

def read_qm7b():
    atoms_list = io.read('qm7b_coords.xyz', index=':')  # 读取所有帧
    property_list = []
    with open('qm7b_coords.xyz') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Properties'):
                pattern = r'\w+_pol="([^"]+)"'
                matches = re.findall(pattern, line)
                ccsd = trans(np.array(list(map(float,matches[0].split()))))
                b3lpy = trans(np.array(list(map(float,matches[1].split())))) 
                scan0 = trans(np.array(list(map(float,matches[2].split()))))
                property_list.append({
                    'ccsd_pol': ccsd.reshape(1,3,3),
                    'b3lyp_pol':b3lpy.reshape(1,3,3),
                    'scan0_pol':scan0.reshape(1,3,3),
                })
    return atoms_list, property_list

def ml(atoms_list,property_list):
    dbfile = os.path.join('.','qm7b.db')
    if not os.path.exists(dbfile):
        new_dataset = ASEAtomsData.create(
            dbfile,
            distance_unit='Ang',
            property_unit_dict={
                'ccsd_pol':'a.u.', 
                'b3lyp_pol':'a.u.',
                'scan0_pol':'a.u.',
            }
        )
        new_dataset.add_systems(property_list, atoms_list)
        for p in new_dataset.available_properties:
            print('-', p)
    print()
    db = connect(dbfile)
    n = len(db)
    print(n)
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
    # print details
    rows = list(db.select())
    for row in rows[:5]:
        atoms = row.toatoms()
        print(atoms)
        for key,value in row.data.items():
            print(key,'\n',value)
    qm7tut = os.path.join('.','qm7tut')
    if not os.path.exists(qm7tut):
        os.mkdir(qm7tut)
    cutoff = 5.
    n_atom_basis = 128
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_ccsd = spk.atomistic.Polarizability(n_in=n_atom_basis, polarizability_key='ccsd_pol')

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_ccsd],
        postprocessors=[trn.CastTo64()]
    )
    output_ccsd = spk.task.ModelOutput(
        name='ccsd_pol',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_ccsd],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=qm7tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(qm7tut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm7tut,
        max_epochs=5, # for testing, we restrict the number of epochs
    )
    trainer.fit(task,dataset)

if __name__ =='__main__':
    main()

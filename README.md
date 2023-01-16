# CLAGC
Implementation for CLAGC model (Attributed Graph Clustering Under the Contrastive Mechanism with Cluster-preserving Augmentation)

# Implementation
pretrain.py: pretrain multilevel contrast to get initial parameters and node representations.

train_conclu.py: jointly train the whole model.

#### Example:

```
python train_conclu.py --dataset Cora --hidden 512 --out_dim 256 --pro_hid 1024 --activation relu --k 0 --rm 85 --mask 0.1 --lr 0.0001 --rep 0.1
```

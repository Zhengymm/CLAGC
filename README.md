# CLAGC
Implementation for CLAGC model (Attributed Graph Clustering Under the Contrastive Mechanism with Cluster-preserving Augmentation)

# Implementation
pretrain.py: pretrain multilevel contrast to get initial parameters and node representations.

train_conclu.py: jointly train the whole model.

#### Example:

--neg True: CLAGC-neg model with negative sampling strategy; --neg False: basic CLAGC model

```
python train_conclu.py --dataset Cora --hidden 512 --out_dim 256 --pro_hid 1024 --activation relu --k 0 --rm 0.85 --mask 0.1 --lr 0.0001

python train_conclu.py --dataset Amazon-Photo --activation relu --hidden 512 --out_dim 128 --pro_hid 1024 --aug knn --k 6 --rm 0.85 --mask 0 --lr 0.00003 --epochs 240 --update_p 20
```

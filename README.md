# CLAGC
Implementation for CLAGC model (Attributed Graph Clustering Under the Contrastive Mechanism with Cluster-preserving Augmentation)

# Implementation
pretrain.py: pretrain multilevel contrast to get initial parameters and node representations.

train_conclu.py: jointly train the whole model.


|    Dataset   | Encoding dimension | Projecting dimension | Activation Function | Learning rate | kNN |  p_e | p_m | Epoch |  T  |
|:------------:|:------------------:|:--------------------:|:-------------------:|:-------------:|:---:|:----:|:---:|:-----:|:---:|
|     Cora     |       512-256      |         1024         |         ReLu        |     0.0001    |  0  | 0.85 | 0.1 |  200  |  1  |
|   CiteSeer   |      1024-512      |         1024         |        PReLu        |     0.0005    |  1  | 0.65 | 0.4 |  300  |  1  |
|    PubMed    |      1024-512      |          512         |         ReLu        |     0.001     |  5  |  0.9 | 0.2 |  200  |  1  |
|    WikiCS    |      1024-1024     |          128         |        PReLu        |   0.01/0.005  |  0  | 0.01 | 0.2 |  200  |  20 |
|   AmazonCom  |       128-128      |         1024         |        PReLu        |     0.0005    |  10 | 0.65 |  0  |  200  | 200 |
| Amazon-Photo |       512-128      |         1024         |         ReLu        |    0.00003    |  6  | 0.85 |  0  |  200  |  20 |
|  Coauthor-CS |       256-256      |         1024         |        PReLu        |     0.001     |  0  |  0.5 |  0  |  200  | 200 |

#### Example:

--neg True: CLAGC-neg model with negative sampling strategy; --neg False: basic CLAGC model

```
python train_conclu.py --dataset Cora --hidden 512 --out_dim 256 --pro_hid 1024 --activation relu --k 0 --rm 0.85 --mask 0.1 --lr 0.0001

python train_conclu.py --dataset Amazon-Photo --activation relu --hidden 512 --out_dim 128 --pro_hid 1024 --aug knn --k 6 --rm 0.85 --mask 0 --lr 0.00003 --epochs 240 --update_p 20
```

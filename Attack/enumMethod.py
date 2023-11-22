from enum import Enum
import torch
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'#

class AttackMethod(Enum):
    vote = '_vote'
    indep = '_indep'
    vote_noise = '_vote_noise'
    auto = '_auto'

class RecMethod(Enum):
    bpr = 'BPR'
    fm = 'FM'
    mlp = 'MLP'
    ncf = 'NCF'
    gru = 'GRU'
    knn = 'KNN'
    svd = 'SVD'
    gnn = 'GNN'
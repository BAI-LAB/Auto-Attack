import torch
import torch.nn as nn
from functions.enumMethod import device

class FM(nn.Module):
    def __init__(self, dataset, tuner_params):
        super().__init__()
        self.u_id_embeddings = nn.Embedding(dataset.num_users, tuner_params['embed_size'])
        self.i_id_embeddings = nn.Embedding(dataset.num_items, tuner_params['embed_size'], padding_idx=dataset.PAD)

        self.u_id_bias = nn.Embedding(dataset.num_users, 1)
        self.i_id_bias = nn.Embedding(dataset.num_items, 1, padding_idx=dataset.PAD)

    def model_init(self):
        nn.init.xavier_uniform_(self.u_id_embeddings.weight)
        nn.init.xavier_uniform_(self.i_id_embeddings.weight)

        self.u_id_bias.weight = torch.nn.Parameter(torch.zeros(self.u_id_bias.weight.shape).to(device))     #nn.init.zero_(self.u_id_bias)
        self.i_id_bias.weight = torch.nn.Parameter(torch.zeros(self.u_id_bias.weight.shape).to(device))     #nn.init.zero_(self.i_id_bias)


    def forward(self, feed_dict, dataset):
        u_ids = feed_dict['user']
        
        i_ids = feed_dict['item']
        u_ids = u_ids.repeat(1, i_ids.shape[1])#u_ids.unsqueeze(-1).repeat(1, i_ids.shape[1])

        u_id_vec = self.u_id_embeddings(u_ids)
        i_id_vec = self.i_id_embeddings(i_ids)

        u_id_b = self.u_id_bias(u_ids)
        i_id_b = self.i_id_bias(i_ids)

        vecs = [u_id_vec, i_id_vec]
        biases = [u_id_b, i_id_b]

        res = torch.zeros_like(u_ids, dtype=torch.float, device=device)
        for a in range(len(vecs)):
            for b in range(a + 1, len(vecs)):
                res += torch.sum(vecs[a] * vecs[b], dim=-1)
            res += biases[a].squeeze()
        return res
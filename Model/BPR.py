import torch
import torch.nn as nn
from functions.enumMethod import device


class BPR(nn.Module):
    def __init__(self, dataset, tuner_params):
        super().__init__()
        self.u_embeddings = nn.Embedding(dataset.num_users, tuner_params['embed_size'])
        self.i_embeddings = nn.Embedding(dataset.num_items, tuner_params['embed_size'], padding_idx=dataset.PAD)
        self.user_bias = nn.Embedding(dataset.num_users, 1)
        self.item_bias = nn.Embedding(dataset.num_items, 1, padding_idx=dataset.PAD)

    def model_init(self):#某种对参数的随机初始化
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)

        # nn.init.xavier_uniform_(self.user_bias.weight)
        # nn.init.xavier_uniform_(self.item_bias.weight)
        self.user_bias.weight = torch.nn.Parameter(torch.zeros(self.user_bias.weight.shape).to(device))     #nn.init.zero_(self.u_id_bias)
        self.item_bias.weight = torch.nn.Parameter(torch.zeros(self.item_bias.weight.shape).to(device))     #nn.init.zero_(self.i_id_bias)


    def forward(self, feed_dict, dataset):#以evaluation()为例
        u_ids = feed_dict['user']#[1]
        i_ids = feed_dict['item']#[1, 用户点击item数]

        cf_u_vectors = self.u_embeddings(u_ids)#[1, emb_size]
        cf_i_vectors = self.i_embeddings(i_ids)#[1, 用户点击item数, emb_size]
        u_bias = self.user_bias(u_ids)#[1,1]
        i_bias = self.item_bias(i_ids)#[1, 用户点击item数]
        i_bias = i_bias.squeeze(-1)

        prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)#乘法为点成，结果[1,用户点击item数, emb_size]
        prediction = prediction + u_bias + i_bias
        return prediction#[1, 用户点击item数]
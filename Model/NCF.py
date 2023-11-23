import torch
import torch.nn as nn

# MLP + MF
class NCF(nn.Module):
    def __init__(self, dataset, tuner_params):
        super().__init__()
        self.mf_user_em = nn.Embedding(dataset.num_users, tuner_params['embed_size'])
        self.mf_item_em = nn.Embedding(dataset.num_items, tuner_params['embed_size'], padding_idx=dataset.PAD)

        self.mlp_user_em = nn.Embedding(dataset.num_users, tuner_params['embed_size'])
        self.mlp_item_em = nn.Embedding(dataset.num_items, tuner_params['embed_size'], padding_idx=dataset.PAD)

        factor = tuner_params['embed_size'] // 2

        hidden_num = [4 * factor, 2 * factor, factor]
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_num[i - 1], hidden_num[i])
                                                  for i in range(1, len(hidden_num))])#ModuleList：各层不共享参数，但要手动书写forward
        self.out_layer = nn.Linear(hidden_num[-1] + tuner_params['embed_size'], 1, bias=False)
        self.act = nn.ReLU()

    def model_init(self):
        nn.init.xavier_uniform_(self.mf_user_em.weight)
        nn.init.xavier_uniform_(self.mf_item_em.weight)
        nn.init.xavier_uniform_(self.mlp_user_em.weight)
        nn.init.xavier_uniform_(self.mlp_item_em.weight)

        [nn.init.xavier_uniform_(unit.weight)  for unit in self.hidden_layers]
        nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, feed_dict, dataset):
        user = feed_dict['user']
        items = feed_dict['item']
        user = user.repeat((1, items.shape[1]))

        mf_u_vectors = self.mf_user_em(user)
        mf_i_vectors = self.mf_item_em(items)
        mlp_u_vectors = self.mlp_user_em(user)
        mlp_i_vectors = self.mlp_item_em(items)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.hidden_layers:
            mlp_vector = layer(mlp_vector).relu()

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.out_layer(output_vector)
        return prediction



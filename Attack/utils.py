import os
import json

import math
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model import *
from functions.enumMethod import device


def collate(feed_dicts):#布置到GPU上，{'user':tensor, 'item':tensor}
    feed_dict = {}
    for key in feed_dicts[0]:
        stack_val = np.array([d[key] for d in feed_dicts])
        if stack_val.dtype == np.object:  # inconsistent length (e.g. history)
            feed_dict[key] = pad_sequence([torch.from_numpy(np.array(x)).long().to(device=device) for x in stack_val],
                                          batch_first=True, padding_value=PAD)#·padding_value=0.
        else:
            feed_dict[key] = torch.from_numpy(stack_val).long().to(device=device)
    return feed_dict


def get_tuner_params(dataset_name, model_name):
    with open(os.path.join('parameters', f'{dataset_name}/{model_name}.json')) as json_file:#模型超参数
        tuner_params = json.load(json_file)
    return tuner_params


def get_model(dataset, model_name, tuner_params, pretrain_path=''):#构造模型、加载预训练参数
    model = globals()[model_name]
    #· model = eval(f'{model_name}.{model_name}')
    model = model(dataset, tuner_params).to(device)
    global PAD
    PAD = dataset.PAD
    
    #加载预训练参数
    if os.path.exists(pretrain_path):
        print("读入预训练参数：", pretrain_path)
        model.load_state_dict(torch.load(pretrain_path,map_location=torch.device(device)))
    model.eval()
    return model


def loss_fn(predictions, feed_dict):
    #【改进失败】用 交叉熵
    # pos_mask = torch.ne(feed_dict['item'], PAD).long()  #真正的点击行为处 =1，其余=0
    # predictions = (predictions).softmax(dim=1)    #得分 → 概率[0,1]；
    # loss = F.cross_entropy(predictions, pos_mask, reduction='mean')   #损失 = 交叉熵，最终损失是各用户损失的均值

    pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
    neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)    #softmax时，防止neg_pred过大导致数值溢出，因而减去最大的值。
    neg_pred = (neg_pred * neg_softmax).sum(dim=1)      #neg_pred的加权和
    loss = F.softplus(-(pos_pred - neg_pred)).mean()    #max{pos_pred与neg_pred间距}
    return loss


def evaluate_method(predictions):
    try:
        predictions = predictions.cpu().data.numpy()
    except:
        pass
    topk = [5, 10, 20]
    metrics = ['HR']
    evaluations = dict()
    sort_idx = (-predictions).argsort(axis=1)
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    # 防止模型的输出为：所有item具有相同分数，因为这会导致指标为1
    idx = (predictions[:, 0] == predictions[:, 1]).nonzero()[0]
    gt_rank[idx] = np.random.randint(1, 101, idx.shape)
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    y_true = np.zeros_like(predictions)
    y_true[:, 0] = 1
    y_true = y_true.flatten()
    y_scores = predictions.flatten()
    evaluations['AUC'] = roc_auc_score(y_true, y_scores)
    return evaluations


def neg_sample_for_neg(target_user_cnt, target_item, dataset, coef=None):#coef~对item的取样数量
    """
    采样：ti + coef × tu群中没点击过的item      (ti永远在第一个， 所有tu的采样结果一致)
    input:
        target_user_cnt: {tu群点击过的item: 点击次数}
        coef: None~全局采样，(int)~负采样的item个数
    output: 
        采样结果(item列表)
    """
    # if isinstance(coef,list):   #已知负采样结果
    #     return [{
    #     'user': np.array(target_user, dtype=np.long),
    #     'item': np.array(coef, dtype=np.long),#[ti, 其他所有item随机排列]
    #     } for target_user in target_users]

    all_items = list(dataset.idx2itemid.keys())
    all_items.remove(target_item)
    if coef is None:#全局
        coef = dataset.num_items    ### 对所有的item都进行采样
        all_items.insert(0, target_item)
        return all_items

    #负采样：ti + coef × tu群中没点击过的item
    all_neg_items = list(set(all_items) - set(target_user_cnt.keys()))
    rand_idxs = np.random.randint(0, len(all_neg_items), coef)
    neg_items = [all_neg_items[idx] for idx in rand_idxs]
    items = [target_item,] + neg_items
    return items

# def neg_sample_for_coverage(target_user,dataset):
#     coef = dataset.num_items
#     items,cnt = dataset.data['test'][target_user],0
#     while cnt < coef:
#         neg_items = np.random.randint(0, dataset.num_items, coef - cnt)
#         tlist = list(filter(lambda item: item not in dataset.users_pos_adj_set[target_user], neg_items))
#         items.extend(tlist)
#         cnt += len(tlist)
#     return {
#         'user': np.array(target_user, dtype=np.long),
#         'item': np.array(items, dtype=np.long),
#     }



def get_rank(predictions):#返回每个用户第一个item的排名[50,]；输入predictions~[50, 94894]
    try:
        predictions = predictions.cpu().data.numpy()
    except:
        pass
    sort_idx = (-predictions).argsort(axis=1)#每行 把下标 降序排列
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1#每个用户，原来的第一个item的当前排名
    return gt_rank

def get_list(predictions):
    try:
        predictions = predictions.cpu().data.numpy()
    except:
        pass
    sort_idx = (-predictions).argsort(axis=1)#每行 把下标 降序排列
    #gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1#每个用户，原来的第一个item的当前排名
    print("sort:",sort_idx)
    return sort_idx

def conver_getANDupdate(predictions,target_item_set,target_pair_feed_dict,dataset):#返回各用户前20点击中正能量的数量，将前20点击加入pairs【而非feed_dict['train']】
    try:
        predictions = predictions.cpu().data().numpy()
    except:
        pass
    sort_idx = (-predictions).argsort(axis=1)#降序
    sort_idx_20 = sort_idx[...,:20]
    # add_lst = []
    hit_nums = []#用户i的前20点击中，是正能量的数量(int)
    for i in range(len(sort_idx_20)):#遍历用户
        tmp_user_feed = target_pair_feed_dict[i]
        first_item_20 = tmp_user_feed['item'][sort_idx_20[i]]#第i个用户的前20个点击item(newid)
        hit_items = target_item_set.intersection(set(first_item_20))#前20个点击中，符合目标方向（正能量）的item
        hit_nums.append(len(hit_items))
    #     for _,item in enumerate(first_item_20):
    #         #· add_lst = list(filter(lambda item: item not in dataset.feed_dict['train'][tmp_user_feed['user']]['item'], first_item_20))
    #         #· dataset.feed_dict['train'][tmp_user_feed['user']]['item'].extend(add_lst)
    #         add_lst.append([int(tmp_user_feed['user']),item])
    #将前20的点击行为，加入到训练集
    # dataset.data['train'].extend(add_lst)
        add_items = set(first_item_20) - set(dataset.feed_dict['train'][int(tmp_user_feed['user'])]['item'])#添加原来没点过的前20item
        if len(add_items) > 0:
            new_pairs = [(dataset.idx2userid[int(tmp_user_feed['user'])], dataset.idx2itemid[it])  for it in add_items]
            dataset.pairs  = np.concatenate([dataset.pairs, new_pairs], axis=0)
            # dataset.feed_dict['train'][int(tmp_user_feed['user'])]['item'].extend(list(add_items))
    return hit_nums

def effect_getANDupdate(predictions,target_item_set,target_pair_feed_dict,dataset,target_item):#返回前20点了正能量的用户占比；把除了正能量的其余前20点击行为注入（更新feed_dict['train']）
    try:
        predictions = predictions.cpu().data().numpy()
    except:
        pass
    sort_idx = (-predictions).argsort(axis=1)
    sort_idx_20 = sort_idx[...,:20]
    # add_lst = []#新交互数据
    hit_usernum = 0.
    for i in range(len(sort_idx_20)):
        tmp_user_feed = target_pair_feed_dict[i]
        first_item_20 = tmp_user_feed['item'][sort_idx_20[i]]#第i个用户的前20个点击item(newid)
        add_items = set(first_item_20)-target_item_set.intersection(set(first_item_20))#前20中不符合正能量的item
        # hit_items += (20-len(add_items))
        if 20-len(add_items) > 0:#前20存在正能量点击
            hit_usernum += 1
        #把除了正能量的其余前20点击行为注入（更新feed_dict['train']）
        add_items = add_items - set(dataset.feed_dict['train'][int(tmp_user_feed['user'])]['item'])#只添加没有点击过的行为
        if len(add_items) > 0:
            dataset.feed_dict['train'][int(tmp_user_feed['user'])]['item'].extend(list(add_items))

    # return hit_items/len(sort_idx_20)
    return hit_usernum/len(sort_idx_20)
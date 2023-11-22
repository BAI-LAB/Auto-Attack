from functions.enumMethod import AttackMethod, RecMethod
from functions.attacker import Attacker
import time
import torch
import numpy as np
import random
import warnings
import argparse
import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=Warning)
#torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(2022)
    startTime = time.time()

    rec = {'bpr': RecMethod.bpr.value, 'fm': RecMethod.fm.value, 'mlp': RecMethod.mlp.value, 'ncf': RecMethod.ncf.value,
           'gru': RecMethod.gru.value, 'knn': RecMethod.knn.value,'gnn':RecMethod.gnn.value}
    rec_method = rec[args.rec]  
    attack = Attacker(rec_method, 'MIND', dataset_path="/home/data/MIND.csv", item_path=args.item_path,
                          other_users_num=args.ou_num) 

    start_user, target_users_num = int(args.start_user), int(args.tu_num)  
    percent_inject_num = args.inj_per  # 
    inject_num = int(attack.dataset.num_users * percent_inject_num)  #
    attack_M = {'vote': AttackMethod.vote.value, 'indep': AttackMethod.indep.value,
                'noise': AttackMethod.vote_noise.value}
    attack_method = attack_M[args.attack]  # 
    target_item = int(args.target_item)  #
    if args.neg_num == 0:
        item_neg_num = None
    else:
        item_neg_num = args.neg_num  #
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(f'start user: {start_user}')
    print(f'target item: {target_item}')
    print(f'number of target users: {target_users_num}')
    print(f'number of inject users for each target users: {inject_num // target_users_num}')
    attack.cluster_attack(start_user, target_users_num, target_item, inject_num, attack_method, rec_method,
                          args.is_click, item_neg_num, percent_inject_num)  # 
    endTime = time.time()
    print(f'recommendation method: {rec_method}')
    print(f'attack method: {attack_method}')
    print(f'neg num: {item_neg_num}')
    print(f'number of inject all users: {inject_num}/({percent_inject_num * 100}%)')
    print(f'Duration time: {endTime - startTime}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_user', type=int, default=239687, help='start_user id')
    parser.add_argument('--target_item', type=int, default=122359, help='target_item id')
    parser.add_argument('--tu_num', type=int, default=5, help='target_user num')
    parser.add_argument('--ou_num', type=int, default=20, help='other_user num')
    parser.add_argument('--neg_num', type=int, default=99, help='neg_num')
    parser.add_argument('--inj_per', type=np.double, default=1, help='percent_inject_num')
    parser.add_argument('--attack', default='vote', help='attack method')
    parser.add_argument('--rec', default='bpr', help='recommendation method')
    parser.add_argument('--share', type=np.double, default=0.25, help='SHARE_SHRESHOLD')
    parser.add_argument('--item_path', default='./data/attack_item.txt', help='item_path')
    parser.add_argument('--is_click', type=int, default=0, help='is_click')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
from sklearn.utils import shuffle
import torch
import numpy as np
import random
from functions.dataset import TouTiao #·MIND
from functions.utils import *
# from functions.utils import neg_sample_for_coverage
from functions.runner import Runner
from functions.enumMethod import AttackMethod
from model import *
from functions.enumMethod import device
import json
import copy


SHARE_SHRESHOLD = 0.25 #！共同爱好的判别标准：SHARE_SHRESHOLD*总数 个tu点击过的item
EFFECTIVE_THRESHOLD = 1./20 #！有效用户比例>=EFFECTIVE_THRESHOLD时，攻击仍有效
class Attacker():
    def __init__(self, model_name, dataset_name, dataset_path,item_path, other_users_num):
        self.device = device
        self.model_name = model_name#RecMethod
        self.other_users_num = other_users_num
        # prepare dataset
        self.tuner_params = get_tuner_params(dataset_name, model_name)#读入模型超参数
        self.dataset = TouTiao(df_path=dataset_path,items_path=item_path)
        
        # prepare model【暂时放弃】
        self.pretrain_path = ''#f'model_saved/{model_name}-_pretrain^Large-30.param'#
        # if not os.path.exists(self.pretrain_path):
        #     raise FileExistsError("!!pretrain_path "+ self.pretrain_path + " doesn't exist.")
        model = get_model(self.dataset, self.model_name, self.tuner_params, pretrain_path=self.pretrain_path)#搭建模型
        # self.model存原始模型
        self.model = model

    def _get_random_other_users(self, start_user,target_users, target_item):
        dataset = self.dataset
        other_users = []
        other_user_cnt = {}#{item_idx: ou群的点击总数}
        sim_metric = 0
        for possible_user in range(dataset.num_users):
            #if possible_user not in dataset.items_adj_set[target_item] and possible_user not in target_users: ##########
                other_users.append(possible_user)
        random.shuffle(other_users)
        other_users = other_users[:self.other_users_num]
        print("other_users",other_users)
        #计算others和tu的相似度
        lst = list(range(dataset.num_users))
        lst.remove(start_user)
        similarity = {tu: len(dataset.users_pos_adj_set[tu].intersection(dataset.users_pos_adj_set[start_user]))
                      for tu in lst}
        unsim_users, _ = zip(*sorted(similarity.items(), key=lambda item: item[1]))
        for other_user in other_users:
            #sim_metric += similarity[other_user]/similarity[unsim_users[-1]] ######################
            #求other_user_cnt
            for item in dataset.users_pos_adj_set[other_user]:
                if item not in other_user_cnt:
                    other_user_cnt[item] = 1
                else:
                    other_user_cnt[item] += 1
        sim_metric = sim_metric / self.other_users_num
        return other_users, other_user_cnt, sim_metric


    def inject(self, inject_users_list):
        dataset = self.dataset
        #print(inject_users_list)
        fu = dataset.inject_users(inject_users_list)#更新dataset.pairs、用户数量dataset.num_users
        dataset.get_adj_list()#根据pairs更新出边表dataset.users_pos_adj_set
        dataset.ratio_split()#划分train、dev、test
        dataset.re_sample()#更新dataset.feed_dict。【·为每个用户（包括fu）增加train_neg_sum个随机点击。】
        return fu

    def evalution(self, model, target_pair_feed_dict, other_user_feed_dict,flag=False):
        with torch.no_grad():
            prediction = np.stack([model(collate([fd]), self.dataset).squeeze().cpu().numpy() for fd in target_pair_feed_dict], axis=0)#target_pair_feed_dict~{'user':userid, 'item':item列表(所有94894)}*50；predictions~[50, 94894]
            #print("prediction",prediction)
            #print(prediction.shape)
            target_item_rank = get_rank(prediction)#每个用户第一个item（ti）的排名
            target_item_mean_rank = target_item_rank.mean()
            target_item_metric = evaluate_method(prediction)
            print("other_user_feed_dict",other_user_feed_dict)
            prediction = np.stack([model(collate([fd]), self.dataset).squeeze().cpu().numpy() for fd in other_user_feed_dict], axis=0)
            other_users_metric = evaluate_method(prediction)

            return target_item_mean_rank,target_item_metric,other_users_metric


    def _get_similar_tu(self, start_user, target_item, target_users_num,is_click):#依据相似度，取tu群
        """
        input: start_user~最初选中的tu、target_itrm~目标item ti、target_users_num~tu群的规模
        return：target_users~tu群、target_user_cnt[item]~tu群对item的总点击次数
        """
        dataset = self.dataset
        target_users = [start_user] #·[]
        target_user_cnt = {}
        lst = list(range(dataset.num_users))
        lst.remove(start_user)
        similarity = {tu: len(dataset.users_pos_adj_set[tu].intersection(dataset.users_pos_adj_set[start_user]))#取交集
                      for tu in lst}#其他所有用户与tu的相似度 = item集的交集长度
        sim_metric = 0
        possible_users, jiaoji_lens = zip(*sorted(similarity.items(), key=lambda item: -item[1]))
        for possible_user in possible_users:#依次取与tu相似度最高且没有点击ti的用户，组成tu群
            if is_click == 0 and possible_user not in dataset.items_adj_set[target_item]:
                target_users.append(possible_user)
                for item in dataset.users_pos_adj_set[possible_user]:
                    if item not in target_user_cnt:
                        target_user_cnt[item] = 1
                    else:
                        target_user_cnt[item] += 1
                sim_metric += similarity[possible_user]/similarity[possible_users[0]]
                if len(target_users) == target_users_num: break
            elif is_click == 1 and possible_user  in dataset.items_adj_set[target_item]:
                target_users.append(possible_user)
                for item in dataset.users_pos_adj_set[possible_user]:
                    if item not in target_user_cnt:
                        target_user_cnt[item] = 1
                    else:
                        target_user_cnt[item] += 1
                sim_metric += similarity[possible_user]/similarity[possible_users[0]]
                if len(target_users) == target_users_num: break
        sim_metric = sim_metric / target_users_num
        return target_users, target_user_cnt, sim_metric
    
    def _get_similar_tu_click(self, start_user, target_item, target_users_num,is_click):#依据相似度，取tu群
        """
        input: start_user~最初选中的tu、target_itrm~目标item ti、target_users_num~tu群的规模
        return：target_users~tu群、target_user_cnt[item]~tu群对item的总点击次数
        """
        dataset = self.dataset
        target_users = [start_user] #·[]
        target_user_cnt = {}
        lst = list(range(dataset.num_users))
        lst.remove(start_user)
        similarity = {tu: len(dataset.users_pos_adj_set[tu].intersection(dataset.users_pos_adj_set[start_user]))#取交集
                      for tu in lst}#其他所有用户与tu的相似度 = item集的交集长度

        lst = list(range(dataset.num_items))
        lst.remove(target_item)
        item_sim = {item: len(dataset.items_adj_set[item].intersection(dataset.items_adj_set[target_item]))#取交集
                      for item in lst}#其他所有用户与tu的相似度 = item集的交集长度
        item_list,item_lens = zip(*sorted(item_sim.items(), key=lambda item: -item[1]))
        pos_items = item_list[:14]
        sim_metric = 0
        possible_users, jiaoji_lens = zip(*sorted(similarity.items(), key=lambda item: -item[1]))
        for possible_user in possible_users:#依次取与tu相似度最高且没有点击ti的用户，组成tu群
            if possible_user not in dataset.items_adj_set[target_item] and is_click == 1:
                flag = 0
                for k in range(len(pos_items)):
                    if possible_user in dataset.items_adj_set[pos_items[k]]:
                        flag = 1
                        break
                if flag == 1:
                    target_users.append(possible_user)
                    for item in dataset.users_pos_adj_set[possible_user]:
                        if item not in target_user_cnt:
                            target_user_cnt[item] = 1
                        else:
                            target_user_cnt[item] += 1
                    sim_metric += similarity[possible_user]/similarity[possible_users[0]]
                    if len(target_users) == target_users_num: break
            
            if possible_user not in dataset.items_adj_set[target_item] and is_click == 0:
                flag = 0
                for k in range(len(pos_items)):
                    if possible_user in dataset.items_adj_set[pos_items[k]]:
                        flag = 1
                        break
                if flag == 0:
                    target_users.append(possible_user)
                    for item in dataset.users_pos_adj_set[possible_user]:
                        if item not in target_user_cnt:
                            target_user_cnt[item] = 1
                        else:
                            target_user_cnt[item] += 1
                    #sim_metric += similarity[possible_user]/similarity[possible_users[0]] ######################################
                    if len(target_users) == target_users_num: break
            
        sim_metric = sim_metric / target_users_num
        return target_users, target_user_cnt, sim_metric

    def _prepare_tuou(self, start_user,target_users_num,target_item,is_click,item_neg_num=None):#准备tu、ou数据，保存为文件以保证每次取的tu和ou一样
        save_path = f'./save/TouTiao/ti{self.dataset.idx2itemid[target_item]}_tu{self.dataset.idx2userid[start_user]}_tunum{target_users_num}_ounum{self.other_users_num}_neg{item_neg_num}_clicksim{is_click}.json'
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as fin:
                tuou_json = json.load(fin)

            target_users, target_user_cnt, tu_similar = tuou_json['tus'],tuou_json['tus_cnt'],tuou_json['tu_similarity']
            other_users, other_user_cnt,ou_similar = tuou_json['ous'],tuou_json['ous_cnt'], tuou_json['ou_similarity']
            neg_samples = tuou_json['neg_sample']
        else:
            #target_users, target_user_cnt, tu_similar = self._get_similar_tu(start_user, target_item, target_users_num,is_click)#tu群、tu群对item的总点击次数

            target_users, target_user_cnt, tu_similar = self._get_similar_tu_click(start_user, target_item, target_users_num,is_click)#tu群、tu群对item的总点击次数
            other_users, other_user_cnt,ou_similar = self._get_random_other_users(start_user,target_users,target_item)
            tuou_user_cnt = target_user_cnt
            tuou_user_cnt.update(other_user_cnt)

            #neg_samples = neg_sample_for_neg(tuou_user_cnt, target_item, self.dataset, coef=item_neg_num) #ti永远是每个tu的第一个item（便于计算排名），其余所有item被每个tu随即次序点击；第一个用户是start_user
            neg_samples =[398,3151,1,30,10,100,2356,789,932,4,233,678,3089,2571,2,3,4,5,222,333,444,555,666,777,888,999,1001]
            #保存为文件
            tuou_json = {'tu_similarity': tu_similar, 'ou_similarity': ou_similar, 'tus':target_users, 'tus_cnt':target_user_cnt, 'ous':other_users, 'ous_cnt':other_user_cnt, 'neg_sample':neg_samples}
            with open(save_path, 'w', encoding='utf-8') as fout:
                json.dump(tuou_json,fout)
        target_pair_feed_dict = [{
                'user': np.array(target_user, dtype=np.long),
                'item': np.array(neg_samples, dtype=np.long),#[ti, 其他所有item随机排列]
            } for target_user in target_users]
        other_user_feed_dict = [{
                'user': np.array(target_user, dtype=np.long),
                'item': np.array(neg_samples, dtype=np.long),#[ti, 其他所有item随机排列]
            } for target_user in other_users]
        #all_item = neg_sample_for_neg(20, target_item, self.dataset, coef=None) #####
        other_i = list(range(self.dataset.num_items))
        #print(other_i)
        other_i.insert(0,'398')
        all_item = list(set(other_i))
        #print("all_item",all_item)
        target_pair_global = [{
                'user': np.array(target_user, dtype=np.long),
                'item': np.array(all_item, dtype=np.long),#[ti, 其他所有item随机排列]
            } for target_user in target_users]
        return target_users,target_user_cnt,target_pair_feed_dict,other_user_feed_dict,target_pair_global,all_item

    def cluster_attack(self,start_user,target_users_num,target_item,inject_num,method,rec,is_click,item_neg_num=None, fu_rate=None):
        ### 数据预处理
        dataset = self.dataset
        start_user = dataset.user2idx[start_user]
        target_item = dataset.item2idx[target_item]

        target_users, target_user_cnt, target_pair_feed_dict, other_user_feed_dict,target_pair_global,all_item = self._prepare_tuou(start_user, target_users_num, target_item,is_click, item_neg_num)

         ### 攻击之前的排序结果
        ti_mean_rank_bf = []
        ### 0.注入攻击之前的数据
        print(f"0. dataset length before attack: {len(self.dataset.data['train'])}\n")
        prediction = np.stack([self.model(collate([fd]), self.dataset).squeeze().cpu().detach().numpy() for fd in target_pair_feed_dict], axis=0)
        sort_idx = (-prediction).argsort(axis=1)
        list_idx = sort_idx[:,:20]
        acc_before = 0
        for i in range(2):
            user = target_users[i]
            #print("user",user)
            adj = dataset.users_pos_adj_set[user]

            for j in range(10):
                item = all_item[list_idx[i][j]]
                #print("item",dataset.idx2itemid[item])
                if dataset.idx2itemid[item] in adj:
                    acc_before += 1
        acc_before /= 400
        idx_before = list_idx

        ti_mean_rank_bf_1,_,_= self.evalution(self.model, target_pair_feed_dict, other_user_feed_dict,True)
        ti_mean_rank_bf.append(ti_mean_rank_bf_1)


        # inject 注入fu
        if method==AttackMethod.vote.value:
            fu = self.vote_inject(target_users_num,target_item,inject_num,target_users,target_user_cnt)
        elif method==AttackMethod.indep.value:
            fu = self.indep_inject(target_users_num,target_item,inject_num,target_users)
        elif method==AttackMethod.vote_noise.value:
            fu = self.vote_noise_inject(target_users_num,target_item,inject_num,target_users,target_user_cnt)
        save_path = f'/home/pris/gsh/recommender/fake_users2/ti{self.dataset.idx2itemid[target_item]}_tu{self.dataset.idx2userid[start_user]}_attack{method}_rec{rec}_tunum{target_users_num}_ounum{self.other_users_num}_funum{inject_num}_neg{item_neg_num}.json'
        with open(save_path,'w') as f:
            json.dump(fu,f)
        # re-train预训练
        ### 1.注入攻击数据之后的数据长度
        print('Re-training ......')
        model = get_model(self.dataset, self.model_name, self.tuner_params)#搭建模型
        #· model = self.get_model()
        log = f'tunum{target_users_num}_ou{self.other_users_num}_tu{start_user}_ti{target_item}_share{SHARE_SHRESHOLD}_inject{inject_num}'
        runner = Runner(model, self.model_name, self.dataset, self.tuner_params, method,fu_rate,log, epoch_num=3000000)#！
        runner.run()

        
        ti_mean_rank_1,ti_metric_1,ou_metric_1 = \
            self.evalution(model, target_pair_feed_dict, other_user_feed_dict,True)
        ti_mean_rank_af, ti_metric_af, ou_metric_af = [],[],[]
        ti_mean_rank_af.append(ti_mean_rank_1)
        ti_metric_af.append(ti_metric_1)
        ou_metric_af.append(ou_metric_1)
        

        success_rate = [self.evalution_success(model,target_pair_feed_dict)]
        times = 4
        for idx in range(1,times):
            model.model_init()
            ti_mean_rank_tmp_bf,_,_ = \
                self.evalution(model,target_pair_feed_dict,other_user_feed_dict)
            ti_mean_rank_bf.append(ti_mean_rank_tmp_bf)

            runner.run()

            success_rate.append(self.evalution_success(model,target_pair_feed_dict))
            ti_mean_rank_tmp_af,ti_metric_tmp_af,ou_metric_tmp_af = \
                self.evalution(model,target_pair_feed_dict,other_user_feed_dict)
            ti_mean_rank_af.append(ti_mean_rank_tmp_af)
            ti_metric_af.append(ti_metric_tmp_af)
            ou_metric_af.append(ou_metric_tmp_af)
            
            prediction = np.stack([self.model(collate([fd]), self.dataset).squeeze().cpu().detach().numpy() for fd in target_pair_feed_dict], axis=0)
            sort_idx = (-prediction).argsort(axis=1)
            list_idx = sort_idx[:,:20]
            acc_after = 0
            same = 0
            for i in range(2):###########
                user = target_users[i]
                #print("user",user)
                adj = dataset.users_pos_adj_set[user]
                #print("adj",adj)
                for j in range(10):###############
                    item = all_item[list_idx[i][j]]
                    #print("item",item)
                    if dataset.idx2itemid[item] in adj:
                        acc_after += 1
                    if list_idx[i][j] in  idx_before[i]:
                        same += 1
            acc_after /= 400
            same /= 400




        effective_num = self.evalution_effective(model,target_pair_feed_dict,target_item_set,target_item)
        print(f'Time 1: start effective user rate = {effective_num}')
        if effective_num < EFFECTIVE_THRESHOLD:
            day_sum_best, day_sum = 0,0
        else:
            day_sum_best, day_sum = 1,1
        EARLY_STOP = 3
        MAX_EFFECT_DAY = 20
        for day in range(2, MAX_EFFECT_DAY+1):
            if effective_num < EFFECTIVE_THRESHOLD:
                EARLY_STOP -=1
                if day_sum > day_sum_best:
                    day_sum_best = day_sum
                day_sum = 0
            else:
                EARLY_STOP = 3
                day_sum += 1
            if EARLY_STOP==0 :
                print("···contiue 3 times, no target set, stop···")
                break

            runner.run()
            effective_num = self.evalution_effective(model,target_pair_feed_dict,target_item_set,target_item)
            print(f'Time {day}: effective user rate = {effective_num}')
        
        if day_sum > day_sum_best:
            day_sum_best = str(day_sum) + ' +'
        print(f'Final length of effective time = {day_sum_best}')
        

        self.dataset.feed_dict['train'] = copy.deepcopy(before_fd_train)
        model.model_init()
        runner.run()
        

        before_hit_nums = self.evalution_converage(model,target_pair_feed_dict,target_item_set)
        tmp_hit_nums = copy.deepcopy(before_hit_nums)
        is_conver = [0 for i in range(target_users_num)]   #用户i是否收敛（0，1）
        time_consumes = [0 for i in range(target_users_num)]    #用户i收敛的耗时（is_conver=1时>0）
        increase_times = [0 for i in range(target_users_num)]   #用户i hit@20连续上升的次数
        print(f'Time 1: start counting converage, average hit@20= : {float(sum(before_hit_nums))/target_users_num/20}')
        CONVERAGE_THRESHOLD = 3     #！
        MAX_CONVER_TIME = 10        #！最多尝试次数
        # best_con = coverage_num
        for time in range(2, MAX_CONVER_TIME+1):
            for u in range(target_users_num):#遍历用户
                if is_conver[u] != 0:#已经收敛的用户
                    continue
                if tmp_hit_nums[u] >= before_hit_nums[u]:
                    increase_times[u] += 1
                    if increase_times[u] >= CONVERAGE_THRESHOLD:#达到收敛
                        is_conver[u] = 1
                        time_consumes[u] = time
                else:
                    increase_times[u] = 0

            if method==AttackMethod.vote.value:
                self.vote_inject(target_users_num,target_item,inject_num,target_users,target_user_cnt)
            elif method==AttackMethod.indep.value:
                self.indep_inject(target_users_num,target_item,inject_num,target_users)
            elif method==AttackMethod.vote_noise.value:
                self.vote_noise_inject(target_users_num,target_item,inject_num,target_users,target_user_cnt)
            
            print(f'Train on new dataset (Time {time})')
            model = get_model(self.dataset, self.model_name, self.tuner_params)
            runner = Runner(model, self.model_name, self.dataset, self.tuner_params, method,fu_rate,log, epoch_num=30)
            runner.run()
            before_hit_nums = copy.deepcopy(tmp_hit_nums)
            tmp_hit_nums = self.evalution_converage(model,target_pair_feed_dict,target_item_set)
            print(f'Time {time}: coverage user rate: {sum(is_conver)/float(target_users_num)}')
            if 0 not in is_conver:
                break



    def evalution_effective(self,model,target_pair_feed_dict,target_item_set,target_item):
        with torch.no_grad():
            prediction = np.stack([model(collate([fd]), self.dataset).squeeze().cpu().numpy() for fd in target_pair_feed_dict], axis=0)
            effective_num = effect_getANDupdate(prediction,target_item_set,target_pair_feed_dict,self.dataset,target_item)
        return effective_num

    def evalution_converage(self,model,target_pair_feed_dict,target_item_set):#返回当前正能量在前20点击中的数量(对多个tu求平均)
        with torch.no_grad():
            prediction = np.stack([model(collate([fd]), self.dataset).squeeze().cpu().numpy() for fd in target_pair_feed_dict], axis=0)
            hit_nums = conver_getANDupdate(prediction,target_item_set,target_pair_feed_dict,self.dataset)#list(int) 用户i在前20点击中，正能量的数量
        return hit_nums

    def evalution_success(self,model,target_pair_feed_dict):#当前可达率。成功干预=ti出现在前20
        with torch.no_grad():
            prediction = np.stack([model(collate([fd]), self.dataset).squeeze().cpu().numpy() for fd in target_pair_feed_dict], axis=0)
            print("pre:",prediction)##########
            list = get_list(prediction)
            one_rank = get_rank(prediction)
            success_time = sum(i < 20 for i in one_rank)
            if success_time>0:
                return success_time/len(one_rank)   #可达率=ti在前20的tu数量÷tu人数 #·1
            else:
                return 0

    def vote_inject(self,target_users_num,target_item,inject_num,target_users,target_user_cnt):#投票
        target_user_list = [int(k) for k, v in target_user_cnt.items() if v >= SHARE_SHRESHOLD * target_users_num]
        print("//共同爱好长度=",len(target_user_list))
        inject_user_list = [target_item] + target_user_list
        inject_users_list = inject_num * [inject_user_list]
        fu = self.inject(inject_users_list)
        return fu
    def indep_inject(self,target_users_num,target_item,inject_num,target_users):#独立攻击
        dataset = self.dataset
        cnt = 0
        i = 0 
        inject = []
        for target_user in target_users:#对每个tu生成数个fu
            target_user_list = list(dataset.users_pos_adj_set[target_user])
            inject_user_list = [target_item] + target_user_list #fu点击 = ti + tu点击
            cnt += inject_num // target_users_num
            inject_users_list = inject_num // target_users_num * [inject_user_list]
            fu = dataset.inject_users(inject_users_list)
            inject.append(fu)
        #print(f'fu for every tu: {cnt // len(target_users)}')
        dataset.get_adj_list()
        dataset.ratio_split()
        dataset.re_sample()
        return inject

    def vote_noise_inject(self,target_users_num,target_item,inject_num,target_users,target_user_cnt):#噪声投票
        target_user_list = [int(k) for k, v in target_user_cnt.items() if v >= SHARE_SHRESHOLD * target_users_num]#筛选4/5的tu群都点击的item
        print("//共同爱好长度=",len(target_user_list))
        inject_users_list = [
            [target_item] + list(np.random.choice(target_user_list, len(target_user_list) // 2, replace=False))
            for _ in range(inject_num)]#fu点击的item = ti+随机选1/2的target_user_list
        fu = self.inject(inject_users_list)
        return fu


import torch
from torch.utils.data import DataLoader
from functions.utils import collate, loss_fn, evaluate_method
from functions.enumMethod import device
# save_path = ''

class Runner():
    def __init__(self, model, model_name, dataset, tuner_params, suffix, fu_rate,log, epoch_num):
        self.device = device
        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.tuner_params = tuner_params
        self.epoch = epoch_num
        self.best_HR5 = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=tuner_params['lr'], weight_decay=tuner_params['l2'])
        self.ld = {key: DataLoader(dataset.feed_dict[key], collate_fn=collate,
                                   batch_size=tuner_params['batch_size'], shuffle=True, drop_last=False)
                   for key in ['train', 'dev', 'test']}
        self.save_path = f'model_saved/{self.model_name}-{suffix}-{self.epoch}-{fu_rate}-{log}.param'


    def run(self):#训练迭代self.epoch次，验证+测试，保存并加载最佳模型（save_path）。为防止多进程混淆，每次启动Runner，赋予唯一save_path
        ### 每次重新训练之前，应该重新初始化一下
        # self.model.load_state_dict(torch.load(f'model_saved/{model_name}_train.param',
        #                                  map_location=torch.device(self.device)))
        # #加载预训练参数
        # global save_path
        # if pretrained and os.path.exists(save_path):
        #     print("读入预训练参数：", save_path)
        #     self.model.load_state_dict(torch.load(save_path,map_location=torch.device(device)))
        # else:
        # self.model.model_init()
        self.best_HR5 = 0
        self.ld['train'] = DataLoader(self.dataset.feed_dict['train'], collate_fn=collate,
                            batch_size=self.tuner_params['batch_size'], shuffle=True, drop_last=False)


        for i in range(self.epoch):
            # print(f"before length feed_dict:{len(self.dataset.feed_dict['train'])}")
            # self.dataset.re_sample()
            loss_avg = self._run_train()#训练迭代*1
            print(f'Epoch: {i}, loss:{loss_avg} train')
            self._run_vt(i)#多指标 测试性能，保存模型

        #取dev上性能最好的模型
        #print(self.best_scores['dev'], "dev")
        #print(self.best_scores['test'], "test")
        self.model.load_state_dict(torch.load(self.save_path, map_location=torch.device(device)))


    def _run_train(self):#训练一轮，返回平均损失值
        self.model.train()
        loss_avg = 0
        for i,feed_dict in enumerate(self.ld['train']):
            self.optimizer.zero_grad()
            prediction = self.model(feed_dict, self.dataset)
            loss = loss_fn(prediction, feed_dict)
            loss_avg += loss.item()
            loss.backward()
            self.optimizer.step()
        loss_avg /= len(self.ld['train'])
        return loss_avg

    def _run_vt(self, i):#多个指标上测试，保存最佳模型参数（用HR@5衡量）
        metrics = ['HR@5', 'HR@10', 'HR@20', 'AUC']
        with torch.no_grad():
            self.model.eval()
            phases = ['dev', 'test']
            vt_loss = {key: 0 for key in phases}
            vt_res, vt_prediction, vt_res_tmp = {}, {}, {}
            for phase in phases:
                vt_res[phase] = {key: 0 for key in metrics}
                for feed_dict in self.ld[phase]:
                    vt_prediction[phase] = self.model(feed_dict, self.dataset)
                    vt_res_tmp[phase] = evaluate_method(vt_prediction[phase])
                    vt_res[phase] = {key: vt_res[phase][key] + vt_res_tmp[phase][key] for key in metrics}
                    vt_loss[phase] += loss_fn(vt_prediction[phase], feed_dict)
                vt_loss[phase] /= len(self.ld[phase])
                vt_res[phase] = {key: vt_res[phase][key] / len(self.ld[phase]) for key in metrics}
                print(f'{phase}——Epoch: {i}, loss:{vt_loss[phase]}, {vt_res[phase]}')
        # save best
        if vt_res['dev']['HR@5'] > self.best_HR5:
            self.best_HR5 = vt_res['dev']['HR@5']
            self.best_scores = vt_res
            # global save_path
            torch.save(self.model.state_dict(), self.save_path)

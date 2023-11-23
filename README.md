# CIKM2023-AutoAttack
Implementation of AutoAttack.<br />
Guo, Sihan, Ting Bai et al. "Targeted Shilling Attacks on GNN-based Recommender Systems." Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 2023.<br />
# Run the model: python main.py
Parameters:<br />
start_user: The userID of the target user<br />
target_item: The itemID of the target item<br />
inj_per: The injection percent of fake users<br />
rec: The recommender system to attack<br />
attack: The attack method<br />
# File description
**Model**: Provide the implementation of diferrent recommender algorithm inlcuding LightGCN<br />
**GNNAttack**: The attack model of AutoAttack<br />
**generate**: The generation of fake users<br />
**runner**: Train the attack model<br />
**utils**: Custom packages<br />
By modifying the parameters, you can try the effects of different attack methods on a variety of recommendation models.<br /><br />
# Requirement
Python 3.7.0<br />
Pytorch 1.9.0<br />
# Cite
Please cite our paper if you use this code in your own work:<br />
@inproceedings{guo2023targeted,<br />
  title={Targeted Shilling Attacks on GNN-based Recommender Systems},<br />
  author={Guo, Sihan and Bai, Ting and Deng, Weihong},<br />
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},<br />
  pages={649--658},<br />
  year={2023}<br />
}

import os
import numpy as np
import matplotlib.pyplot as plt
from fcd_torch.utils import SmilesDataset
from fcd_torch import FCD
from sklearn.manifold import TSNE  # 导入t-SNE

# 文件夹路径
data_folder = "/home/lk/project/mol_generate/GDSS/evaluation/eval_regression/datas"
# 初始化FCD类
fcd = FCD(device='cuda:0', n_jobs=8)

# 存储所有数据的mu和sigma
all_mus = []
all_sigmas = []

# 遍历数据文件
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    # 跳过非文件的项目
    if not os.path.isfile(file_path):
        continue
    
    # 读取SMILES列表
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file.readlines() if line.strip()]

    # 获取mu和sigma
    result = fcd.precalc(smiles_list)
    mu = result['mu']
    sigma = result['sigma']

    # 存储到列表中
    all_mus.append(mu)
    all_sigmas.append(sigma)

mean_mus  = np.mean(np.array(all_mus), axis=1)
mean_sigmas = np.mean(np.array(all_sigmas), axis=1)
mean_sigmas = np.mean(np.array(mean_sigmas), axis=1)
# 将所有数据合并
# all_mus = np.concatenate(all_mus, axis=0)
# all_sigmas = np.concatenate(all_sigmas, axis=0)

import pdb;pdb.set_trace()
# 对所有数据进行降维
mus_2d = TSNE(perplexity=5, n_components=2).fit_transform(mean_mus)
sigmas_2d = TSNE(perplexity=5, n_components=2).fit_transform(mean_sigmas)

# 可视化mu
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Mu Distribution')
plt.scatter(mus_2d[:, 0], mus_2d[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 可视化sigma
plt.subplot(1, 2, 2)
plt.title('Sigma Distribution')
plt.scatter(sigmas_2d[:, 0], sigmas_2d[:, 1])
plt.xlabel('Sigma Feature 1')
plt.ylabel('Sigma Feature 2')

plt.tight_layout()


# 保存图片
save_path = "./visualization.png"  # 请替换为你想要保存的路径
plt.savefig(save_path)
plt.show()
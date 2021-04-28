import os, pickle, glob
from collections import defaultdict
import pandas as pd
# from sklearn.metrics import roc_curve, auc

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# from DNN2 import DSCNN
from DNN1 import DSCNN

model_path = 'data/DNN1'
# model_path = 'data/DNN2'

latest_file = max(glob.glob(model_path + '/*'), key=os.path.getctime)
print(latest_file)

trained_model = DSCNN.load_from_checkpoint(latest_file)
trained_model = trained_model.eval()

#############################
pkl_file = 'data/data_mntjulip.pkl'  ###############################

(test_splice_site_introns_dict, conditions, test_groups_batches,
 test_results_batches, test_coords_batches, test_genes) = pickle.load(open(pkl_file, 'rb'))

# test_introns_list
test_introns_list = []
for introns_batches in test_coords_batches:
    for introns in introns_batches:
        test_introns_list.append(introns)

test_results = []
for results in test_results_batches:
    for _, _, ratios in results:
        test_results.append(np.array(ratios))


def generate_tissue_avg_counts(counts):
    pre = 0
    avg_counts = []
    for i, j in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 24, 36, 40, 52, 64, 75, 87, 95, 102, 114, 120]):
        avg_counts.append(counts[:, :, pre:j].mean(axis=-1, keepdims=True))
        pre = j
    return torch.cat(avg_counts, axis=-1)


model_preditions = []
tissue_avg_counts = []
pkl_file = 'data/data.pkl'  ###############################
train_set, test_set = pickle.load(open(pkl_file, "rb"))
for batch in DataLoader(test_set, collate_fn=trained_model.collate_fn, batch_size=1, num_workers=2):
    X, Y = batch
    size = X.size(1)
    X = X.view(-1, X.size(2), X.size(3))
    x = X.unsqueeze(1)  # shape: (batch, 1, 4, seq_length)
    embedded = trained_model.embedding_forward(x)  # shape: (batch, embed_size)
    out = trained_model.output_layer(embedded)  # shape: (batch, dims)
    out = out.view((-1, size, trained_model.hparams.dims))
    out = out.softmax(axis=1)
    model_preditions.append(out)
    tissue_avg_counts.append(generate_tissue_avg_counts(Y))

pred_results = []
test_counts = []
for predictions, counts in zip(model_preditions, tissue_avg_counts):
    for i in range(predictions.size(0)):
        pred_results.append(predictions[i].detach().numpy())
        test_counts.append(counts[i].detach().numpy())


size_introns_dict = defaultdict(list)
for site, introns in test_splice_site_introns_dict.items():
    size = len(introns)
    size_introns_dict[size].append(introns)

orderted_introns = []
for size, introns_list in size_introns_dict.items():
    orderted_introns += introns_list


def calculate_auc(df):
    class_count = df['label'].value_counts()
    pos_count = class_count[1]
    neg_count = class_count[0]
    tp = fp = 0
    coords = []

    for label in df['label']:
        if label == 1:
            tp += 1
        else:
            fp += 1
        coords.append((fp, tp))
    fp, tp = map(list, zip(*coords))
    tpr = tp / pos_count
    fpr = fp / neg_count
    return auc(fpr, tpr)


for k in range(12):
    res_low, res_med, res_high = [], [], []
    for i, introns in enumerate(orderted_introns):
        index = test_introns_list.index(introns)
        mntjulip_result = test_results[index]
        pred_result = pred_results[i]
        test_count = test_counts[i]
        test_count_sum = test_count.sum(axis=0)
        if test_count_sum[k] < 5:
            continue

        n, m = pred_result.shape
        for j in range(n):
            l_low = int(0 <= mntjulip_result[j, k] <= 1 / 3)
            res_low.append((pred_result[j, k], l_low))

            l_med = int(1 / 3 <= mntjulip_result[j, k] <= 2 / 3)
            res_med.append((pred_result[j, k], l_med))

            l_high = int(2 / 3 <= mntjulip_result[j, k] <= 1)
            res_high.append((pred_result[j, k], l_high))


    results = [0, 0, 0]
    ##########
    df = pd.DataFrame(res_low, columns =['pred', 'label'])
    df2 = df.sort_values(by=['pred'])
    results[0] = calculate_auc(df2)

    ##########
    df = pd.DataFrame(res_med, columns =['pred', 'label'])
    df2 = df.sort_values(by=['pred'], key=lambda l: abs(l - 0.5))
    results[1] = calculate_auc(df2)

    ##########
    df = pd.DataFrame(res_high, columns =['pred', 'label'])
    df2 = df.sort_values(by=['pred'], ascending=False)
    results[2] = calculate_auc(df2)
    print(results)


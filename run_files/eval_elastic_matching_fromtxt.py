import torch
import os
import numpy as np
import json
from sklearn.metrics import auc, roc_curve


if __name__ == "__main__":
    # eval_result_path = "eval_logs/baseline/elastic_matching_0.4_results.txt"
    eval_result_path = "eval_logs/sampleStep_cls_blank/elastic_matching_0.4_results.txt"
    ground_truth_path = "Datasets/Elastic-Matching/elastic_pair_0.4.json"
    
    pred_dict = {}
    with open(eval_result_path, 'r') as f:
        for line in f.readlines():
            video1, label1, video2, label2, distance = line.strip().split(' ')
            distance = float(distance)
            pred_key = "{}-{}-{}-{}".format(label1, video1, label2, video2)
            pred_dict[pred_key] = distance
            
    with open(ground_truth_path, 'r') as gt_f:
        gt_dict = json.load(gt_f)
    
    auc_list = []
    for query_key in gt_dict:
        preds = []
        labels = []
        task = gt_dict[query_key]
        for candidate_key in task:
            pred_key = "{}-{}".format(query_key, candidate_key)
            distance = pred_dict[pred_key]
            pair_label = task[candidate_key]['label']
            preds.append(distance)
            labels.append(pair_label)
        
        labels = np.array(labels)
        preds = np.array(preds)
        fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=0)
        auc_value = auc(fpr, tpr)
        auc_list.append(auc_value)
    
    average_auc = np.array(auc_list).mean()
    print("Average AUC: ", average_auc)
    
        
        
        
    # fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    # auc_value = auc(fpr, tpr)
    # wdr_value = compute_WDR(preds, labels1_all, labels2_all)
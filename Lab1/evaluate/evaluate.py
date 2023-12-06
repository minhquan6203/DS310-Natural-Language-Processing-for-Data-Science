from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def compute_score(labels,preds):
    acc=[]
    f1=[]
    precision=[]
    recall=[]
    for i in range(len(labels)):
      acc.append(accuracy_score(labels[i],preds[i][1:len(labels[i])]))
      f1.append(f1_score(labels[i],preds[i][1:len(labels[i])],average='macro',zero_division=1))
      precision.append(precision_score(labels[i],preds[i][1:len(labels[i])],average='macro',zero_division=1))
      recall.append(recall_score(labels[i],preds[i][1:len(labels[i])],average='macro',zero_division=1))
    return np.mean(acc),np.mean(f1),np.mean(precision),np.mean(recall)
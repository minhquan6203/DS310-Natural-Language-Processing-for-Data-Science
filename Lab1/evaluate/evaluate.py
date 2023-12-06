from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def compute_score(labels,preds,Tag_space):
    acc=[]
    f1=[]
    precision=[]
    recall=[]
    for i in range(len(labels)):
      label=[Tag_space[n] for n in labels[i] if n < len(Tag_space)]
      pred=[Tag_space[n] for n in preds[i] if n < len(Tag_space)]
      acc.append(accuracy_score(label,pred[:len(label)]))
      f1.append(f1_score(label,pred[:len(label)],average='macro',zero_division=1))
      precision.append(precision_score(label,pred[:len(label)],average='macro',zero_division=1))
      recall.append(recall_score(label,pred[:len(label)],average='macro',zero_division=1))
    return np.mean(acc),np.mean(f1),np.mean(precision),np.mean(recall)
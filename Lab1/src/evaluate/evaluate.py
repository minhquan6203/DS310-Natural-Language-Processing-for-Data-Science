from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_score(labels,preds,tag_space):
    label_list=[]
    pred_list=[]
    for i in range(len(labels)):
      label=[n for n in labels[i] if n < len(tag_space)]
      label_list.extend(label)
      pred_list.extend(preds[i][1:len(label)+1])

    acc=accuracy_score(label_list,pred_list)
    f1_macro=f1_score(label_list,pred_list,average='macro',zero_division=1)
    f1_micro=f1_score(label_list,pred_list,average='micro',zero_division=1)
    return acc,f1_macro,f1_micro
import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data_lstm import Get_Loader
from data_utils.load_data_bert import Get_Loader_Bert
from evaluate.evaluate import compute_score
from tqdm import tqdm
from model.build_model import build_model
from data_utils.vocab import create_ans_space
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
class Inference:
    def __init__(self,config):
        self.save_path=os.path.join(config['train']['output_dir'],config['model']['type_model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.POS_space,self.Tag_space=create_ans_space(config)
        self.base_model = build_model(config).to(self.device)
        if config['model']['type_model']=='lstm':
            self.dataloader = Get_Loader(config)
        if config['model']['type_model']=='bert':
            self.dataloader = Get_Loader_Bert(config) 
        # self.POS_space,_=create_ans_space(config)
    def predict(self):
        test_data = self.dataloader.load_data_test()
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test hả')
        self.base_model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for it,item in enumerate(tqdm(test_data)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.base_model(item['inputs'])
                preds = logits.argmax(axis=-1).cpu().numpy()
                for i in range(len(item['labels'])):
                    label=[n for n in item['labels'][i] if n < len(self.Tag_space)]
                    true_labels.extend(label)
                    pred_labels.extend(preds[i][1:len(label)+1])
        test_acc=accuracy_score(true_labels,pred_labels)
        test_f1_macro=f1_score(true_labels,pred_labels,average='macro',zero_division=1)
        test_f1_micro=f1_score(true_labels,pred_labels,average='micro',zero_division=1)
        print(f"test acc: {test_acc:.4f} | test f1_macro: {test_f1_macro:.4f} | test f1_micro: {test_f1_micro:.4f}")
        self.Tag_space.extend(['[PAD]','[CLS]','[SEP]'])
        label=[self.Tag_space[n] for n in true_labels]
        pred = [self.Tag_space[n] for n in pred_labels]
        print('classification report:\n',classification_report(label,pred,zero_division=1))

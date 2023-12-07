from model.transformer_model import createTrans_Model,Trans_Model
from model.baseline_model import createBaseline_Model, Baseline_Model

def build_model(config, answer_space):
    if config['model']['type_model']=='trans':
        return createTrans_Model(config, answer_space)
    if config['model']['type_model']=='baseline':
        return createBaseline_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='trans':
        return Trans_Model(config, num_labels)
    if config['model']['type_model']=='baseline':
        return Baseline_Model(config,num_labels)

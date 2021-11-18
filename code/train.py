import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *

LABEL_LIST = [
    'no_relation', 'org:hometown', 'org:rival', 'org:counterpart', 'org:member_of', 
    'org:founded', 'org:stadium', 'org:members', 'evt:happened', 'per:role'
]

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""

    no_relation_label_idx = LABEL_LIST.index("no_relation")
    label_indices = list(range(len(LABEL_LIST)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(len(LABEL_LIST))[labels]

    score = np.zeros((len(LABEL_LIST),))
    for c in range(len(LABEL_LIST)):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
        
    return num_label

def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # wandb setting
    # wnadb entity is automatically set up after wandb login. If you want to change it, then relogin.
    os.environ["WANDB_ENTITY"] = "annotation_soccer"
    os.environ["WANDB_PROJECT"] = "EVAL_MAIN"

    # load dataset
    dataset = load_data("./dataset.csv")

    # train_test_split
    train_dataset, dev_dataset, _, _ = train_test_split(dataset, dataset['label'], test_size=0.2, shuffle=True, stratify=dataset['label'], random_state=42)
    
    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = len(LABEL_LIST)

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)
    
    # training arguments
    lr_rate=5e-5
    epochs=3
  
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        seed=42,
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=50,                 # model saving step.
        num_train_epochs=epochs,              # total number of training epochs
        learning_rate=lr_rate,               # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        #warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=50,              # log saving step.
        evaluation_strategy='steps', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        eval_steps = 50,            # evaluation step.
        load_best_model_at_end = True,
        report_to='wandb',
        run_name=f'{MODEL_NAME}_EPOCHS_{epochs}_LR_{lr_rate}'
    )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained('./best_model')
    
def main():
    train()

if __name__ == '__main__':
    main()

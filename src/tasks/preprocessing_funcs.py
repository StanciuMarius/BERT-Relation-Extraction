#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging
import json 

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(text, mode='train'):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]
        
        # check entries
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1
        
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks


def process_qald_json(file):
    examples = json.load(file)
    texts = [example['text'] for example in examples]
    relations = [example['relation'] for example in examples]
    return texts, relations
    
def preprocess_qald(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = args.train_data
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as file:
        texts, relations = process_qald_json(file)

    df_train = pd.DataFrame(data={'sents': texts, 'relations': relations})
    
    data_path = args.test_data
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as file:
        texts, relations = process_qald_json(file)
    
    df_test = pd.DataFrame(data={'sents': texts, 'relations': relations})
    
    rm = Relations_Mapper(pd.concat([df_train['relations'], df_test['relations']]))
    save_as_pickle(args, 'relations.pkl', rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle(args, 'df_train.pkl', df_train)
    save_as_pickle(args, 'df_test.pkl', df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm
    
def preprocess_semeval2010_8(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = args.train_data
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    data_path = args.test_data
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle(args, 'relations.pkl', rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle(args, 'df_train.pkl', df_train)
    save_as_pickle(args, 'df_test.pkl', df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
                                                             axis=1)
        self.df[self.df.apply(lambda x: len(x['input']) < 500, axis=1)] # Filter out sequences that are too long

        def get_e1e2_start(x, e1_id, e2_id):
            e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                            [i for i, e in enumerate(x) if e == e2_id][0])
            return e1_e2_start
        
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])
    
def load_dataloaders(args):
    if args.model_no == 0:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model = 'bert-base-uncased'
        lower_case = True
        model_name = 'BERT'
    elif args.model_no == 1:
        from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
        model = 'albert-base-v2'
        lower_case = False
        model_name = 'ALBERT'
        
    if os.path.isfile(os.path.join(args.temp_folder_path, "%s_tokenizer.pkl" % model_name)):
        tokenizer = load_pickle(args, "%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        tokenizer = Tokenizer.from_pretrained(model, do_lower_case=lower_case)

        additional_tokens = []
        if args.additional_tokens_path:
            with open(args.additional_tokens_path, 'r') as file:
                additional_tokens = [token.strip() for token in file.read().split(',')]
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'] + additional_tokens)

        save_as_pickle(args, "%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info("Saved %s tokenizer at ./temp_folder_path/%s_tokenizer.pkl" %(model_name, model_name))
    
    relations_path = os.path.join(args.temp_folder_path, 'relations.pkl')
    train_path = os.path.join(args.temp_folder_path, 'df_train.pkl')
    test_path = os.path.join(args.temp_folder_path, 'df_test.pkl')
    if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
        rm = load_pickle(args, 'relations.pkl')
        df_train = load_pickle(args, 'df_train.pkl')
        df_test = load_pickle(args, 'df_test.pkl')
        logger.info("Loaded preproccessed data.")
    else:
        df_train, df_test, rm = preprocess_qald(args)
    
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    train_length = len(train_set); test_length = len(test_set)
    PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                      label_pad_value=tokenizer.pad_token_id,\
                      label2_pad_value=-1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                              num_workers=0, collate_fn=PS, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                              num_workers=0, collate_fn=PS, pin_memory=False)
    return train_loader, test_loader, train_length, test_length
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019

@author: weetee
"""
import os
import math
import torch
import torch.nn as nn
from .misc import save_as_pickle, load_pickle
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=self.lm_ignore_idx)
        self.BCE_criterion = nn.BCELoss(reduction='mean')
    
    def p_(self, f1_vec, f2_vec):
        p = 1/(1 + math.exp(torch.dot(f1_vec, f2_vec)))
        return p
    
    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, probabilities)
        blank_labels: (batch_size, 0 or 1)
        '''
        lm_loss = self.LM_criterion(lm_logits, lm_labels)
        blank_loss = self.BCE_criterion(blank_logits, blank_labels)
        total_loss = lm_loss + blank_loss
        return total_loss

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle(args, "test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle(args, "test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=True):
    '''
    evaluate must be called after loss.backward()
    '''
    # lm_logits
    lm_logits_pred_ids = torch.softmax(lm_logits, dim=-1).max(1)[1]
    lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum().float()/len(masked_for_pred)).item()
    
    if print_:
        print("Predicted masked tokens: \n")
        print(tokenizer.decode(lm_logits_pred_ids.cpu().numpy() if lm_logits_pred_ids.is_cuda else \
                               lm_logits_pred_ids.numpy()))
        print("\nMasked labels tokens: \n")
        print(tokenizer.decode(masked_for_pred.cpu().numpy() if masked_for_pred.is_cuda else \
                               masked_for_pred.numpy()))
        
    # blanks
    blanks_diff = ((blanks_logits - blank_labels)**2).detach().cpu().numpy().sum() if blank_labels.is_cuda else\
                    ((blanks_logits - blank_labels)**2).detach().numpy().sum()
    blanks_mse = blanks_diff/len(blank_labels)
    
    if print_:
        print("Blanks MSE: ", blanks_mse)
    return lm_accuracy, blanks_mse
    
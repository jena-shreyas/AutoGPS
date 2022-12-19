#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
from tqdm import tqdm
from dataset import ConstrDataset

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BartTokenizerFast, BartForConditionalGeneration, get_linear_schedule_with_warmup


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


def collate_fn(batch):
    input, output = zip(*batch)

    input = [torch.LongTensor(i) for i in input]
    output = [torch.LongTensor(i) for i in output]
    input = pad_sequence(input, batch_first=True, padding_value=1)
    output = pad_sequence(output, batch_first=True, padding_value=1)
    return input, output


if __name__ == '__main__':

    MAX_EPOCH = 5

    diagram_logic_file = '../data/new/logic_forms/diagram_logic_forms_annot.json'
    text_logic_file = '../data/new/logic_forms/text_logic_forms_annot_dissolved.json'
    pred_file = 'results/train/pred_constr_reqd.json'

    output_path = "models/"
    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cpu')

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')

    train_set = ConstrDataset('train', diagram_logic_file, text_logic_file, pred_file, tokenizer)
    # val_set = GeometryDataset('val', diagram_logic_file, text_logic_file, pred_file, tokenizer)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_set, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), 3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, len(train_loader) * 20)
    best_loss = 1e6

    for epoch in tqdm(range(MAX_EPOCH)):
        total_loss = 0
        model.train()
        for idx, (input, output) in enumerate(train_loader):
            optimizer.zero_grad()
            res = model(input.to(device), labels = output.to(device))
            loss = res.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print('\nepoch: ', epoch, " train_loss ", total_loss)
        torch.save(model.state_dict(), output_path+"/tp_model_" + str(epoch) + ".pt")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), output_path+"/tp_model_best.pt")


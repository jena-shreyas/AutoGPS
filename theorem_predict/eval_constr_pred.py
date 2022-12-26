#!/usr/bin/env python
# coding: utf-8

import json
import ast
from tqdm import tqdm

import torch
from transformers import BartForSequenceClassification, BartTokenizerFast


def evaluate(diagram_logic_file, text_logic_file, tokenizer_name, model_name, check_point):

    test_lst = range(24, 29)

    ## read logic form files
    with open(diagram_logic_file) as f:
        diagram_logic_forms = json.load(f)
    with open(text_logic_file) as f:
        text_logic_forms = json.load(f)

    combined_logic_forms = {}
    for pid in test_lst:
        combined_logic_forms[pid] = diagram_logic_forms[str(pid)]['diagram_logic_forms'] + \
                                    text_logic_forms[str(pid)]['text_logic_forms']

    ## build tokenizer and model
    tokenizer = BartTokenizerFast.from_pretrained(tokenizer_name) # 'facebook/bart-base'
    model = BartForSequenceClassification.from_pretrained(model_name).to(device) # 'facebook/bart-base'
    model.load_state_dict(torch.load(check_point))

    final = dict()
    for pid in tqdm(test_lst):
        print(f"\n\nPID : {pid}\n\n")
        input = str(combined_logic_forms[pid])
        tmp = tokenizer.encode(input)
        if len(tmp) > 1024:
            tmp = tmp[:1024]
        input = torch.LongTensor(tmp).unsqueeze(0).to(device)

        # output = model.generate(input, bos_token_id=0, eos_token_id=2,
        #                      max_length=2, num_beams=10, num_return_sequences=1)

        output = model(input)
        print(output)

        print("Output data type : ", type(output))
        # print(out.size())

        # ## refine output sequence
        # seq = []
        # for j in range(seq_num):
        #     res = tokenizer.decode(output[j].tolist())
        #     res = res.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
        #     # print(res)
        #     try:
        #         res = ast.literal_eval(res) # string class to list class
        #     except Exception as e:
        #         res = []
        #     seq.append(res)

        final[str(pid)] = {"id": str(pid), "output": output}

    return final


if __name__ == '__main__':

    diagram_logic_file = '../data/new/logic_forms/diagram_logic_forms_annot.json'
    text_logic_file = '../data/new/logic_forms/text_logic_forms_annot_dissolved.json'

    check_point = 'models/tp_model_best.pt'
    output_file = 'results/test/constr_pred_outputs.json'

    tokenizer_name = 'facebook/bart-base'
    model_name = 'facebook/bart-base'

    # SEQ_NUM = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(diagram_logic_file, text_logic_file, tokenizer_name, model_name, check_point)
    # print("Type of result : ", type(result))

    # with open(output_file, 'w') as f:
    #     json.dump(result, f)


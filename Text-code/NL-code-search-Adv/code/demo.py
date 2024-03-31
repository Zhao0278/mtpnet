from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import sys
import os

from model import Model


def single_tokenize(text, tokenizer, block_size=256):
    tokens = tokenizer.tokenize(text)[:block_size - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = block_size - len(ids)
    ids += [tokenizer.pad_token_id] * padding_length
    return torch.tensor([ids])


if __name__ == "__main__":
    config =RobertaConfig.from_pretrained("microsoft/codebert-base")
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", do_lower_case=True)
    model = RobertaModel.from_pretrained("roberta-base", config=config)
    model = Model(model, config, tokenizer, args=None)
    model.load_state_dict(torch.load("../model/epoch_2/subject_model.pth"))


    query = "print hello world"
    code_1 = """
    import numpy as np
    """
    code_2 = """
    a = 'hello world'
    """
    code_3 = """
    cout << "hello world" << endl;
    """
    code_4 = '''
    print('hello world')
    '''
    codes = []
    codes.append(code_1)
    codes.append(code_2)
    codes.append(code_3)
    codes.append(code_4)
    scores = []
    nl_inputs = single_tokenize(query, tokenizer)
    for code in codes:
        code_inputs = single_tokenize(code, tokenizer)
        score = model(code_inputs, nl_inputs, return_scores=True)
        scores.append(score)
    print("Query:", query)
    for i in range(len(codes)):
        print('------------------------------')
        print("Code:", codes[i])
        print("Score:", float(scores[i]))
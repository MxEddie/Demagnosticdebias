import argparse
import regex as re
import nltk
import json 
import torch
import string 
from torch.utils.data import Dataset

from transformers import *

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def parse_args():
    parser = argparse.ArgumentParser()
    tp = lambda x:list(x.split(','))

    parser.add_argument('--input', type=str, required=True,
                        help='Data')
    parser.add_argument('--stereotypes', type=tp)
    parser.add_argument('--attributes', type=tp, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['bert', 'roberta', 'electra', 'albert', 'dbert'])

    args = parser.parse_args()

    return args

def prepare_transformer(args):
    if args.model_type == 'bert':
        pretrained_weights = 'bert-base-cased'
        model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    return model, tokenizer

def encode_to_is(tokenizer, data, add_special_tokens):
    if type(data) == list:
        data = [tuple(tokenizer.encode(sentence, add_special_tokens=add_special_tokens)) for sentence in data]
    elif type(data) == dict:
        data = {tuple(tokenizer.encode(key, add_special_tokens=add_special_tokens)): tokenizer.encode(value, add_special_tokens=add_special_tokens)
                for key, value in data.items()}

    return data

def split_data(input, dev_rate, max_train_data_size):
    if max_train_data_size > 0:
        train = input[:max_train_data_size]
        dev = input[max_train_data_size:]
    else:
        train = input[int(dev_rate*len(input)):]
        dev = input[:int(dev_rate*len(input))]

    return train, dev

def main(args):
    #data = [l.strip() for l in open(args.input)]
    #print("data list compiled")
    stereotypes_l = []
    all_stereotypes_set = set()
    for stereotype in args.stereotypes:
        l = [word.strip() for word in open(stereotype)]
        #l = [word.lower().strip() for word in open(stereotype)]
        stereotypes_l.append(set(l))
        all_stereotypes_set |= set(l)
    print(all_stereotypes_set)
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    attributes_l = []
    all_attributes_set = set()
    for attribute in args.attributes:
        l = [word.strip() for word in open(attribute)]
        #l = [word.lower().strip() for word in open(attribute)]
        attributes_l.append(set(l))
        all_attributes_set |= set(l)
    print(all_attributes_set) 
    
    model, tokenizer = prepare_transformer(args)

    #if args.stereotypes:
    #    tok_stereotypes = encode_to_is(tokenizer, all_stereotypes_set, add_special_tokens=False)

    neutral_examples = [[] for _ in range(len(stereotypes_l))]
    if args.stereotypes:
        neutral_labels = [[] for _ in range(len(stereotypes_l))]
    attributes_examples = [[] for _ in range(len(attributes_l))]
    attributes_labels = [[] for _ in range(len(attributes_l))]

    other_num = 0

    #for line in data:
    with open(file=args.input,encoding='utf-8') as f:
        for idx,lline in enumerate(f):
            if idx < 45000000:
                if idx % 1000000 == 0:
                    print(idx)
                comment_line = json.loads(lline)
                comment = comment_line["body"]
                if comment != '[deleted]' and comment != '[removed]':
                    comment = comment.replace("&gt;"," ")
                    comment = comment.replace("&amp;"," ")
                    comment = comment.replace("&lt;"," ")
                    comment = comment.replace("&quot;"," ")
                    comment = comment.replace("&apos;"," ")
                    line = comment.translate(translator)
                    neutral_flag = True
                    line = line.strip()
                    if len(line) < 1:
                        continue
                    leng = len(line.split())
                    if leng > args.block_size or leng <= 1:
                        continue
                    tokens_orig = [token.strip() for token in re.findall(pat, line)]
                    token_set = set(tokens_orig)
                    #tokens_lower = [token.lower() for token in tokens_orig]
                    #token_set = set(tokens_lower)


                    attribute_other_l = []
                    for i, _ in enumerate(attributes_l):
                        a_set = set()
                        for j, attribute in enumerate(attributes_l):
                            if i != j:
                                a_set |= attribute
                        attribute_other_l.append(a_set)

                    stereotype_other_l = []
                    for i, _ in enumerate(stereotypes_l):
                        a_set = set()
                        for j, stereotype in enumerate(stereotypes_l):
                            if i != j:
                                a_set |= stereotype
                        stereotype_other_l.append(a_set)


                    for i, (attribute_set, other_set) in enumerate(zip(attributes_l, attribute_other_l)):
                        if attribute_set & token_set:
                            neutral_flag = False
                            if not other_set & token_set:
                                orig_line = line
                                line = tokenizer.encode(line, add_special_tokens=True,max_length=512,truncation=True)
                                labels = attribute_set & token_set
                                for label in list(labels):
                                    idx = tokens_orig.index(label)
                                #idx = tokens_lower.index(label)
                                label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True,max_length=512,truncation=True))[1:-1]
                                line_ngram = list(nltk.ngrams(line, len(label)))
                                if label not in line_ngram:
                                    label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=False,max_length=512,truncation=True))
                                    line_ngram = list(nltk.ngrams(line, len(label)))
                                    if label not in line_ngram:
                                        label = tuple(tokenizer.encode(f'a {tokens_orig[idx]} a',max_length=512,truncation=True))[1:-1]
                                        line_ngram = list(nltk.ngrams(line, len(label)))
                                        if label not in line_ngram:
                                            label = tuple([tokenizer.encode(f'{tokens_orig[idx]}2',max_length=512,truncation=True)[0]])
                                            line_ngram = list(nltk.ngrams(line, len(label)))
                                idx = line_ngram.index(label)
                                attributes_examples[i].append(line)
                                attributes_labels[i].append([idx + j for j in range(len(label))])
                            break
                     

                    if neutral_flag:
                        if args.stereotypes:
                            for i, (stereotype_set, other_set) in enumerate(zip(stereotypes_l, stereotype_other_l)):       
                                if stereotype_set & token_set:
                                    if not other_set & token_set:
                                        orig_line = line
                                        line = tokenizer.encode(line, add_special_tokens=True,max_length=512,truncation=True)
                                        labels = stereotype_set & token_set
                                        for label in list(labels):
                                            idx = tokens_orig.index(label)
                                            #idx = tokens_lower.index(label)
                                            label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True,max_length=512,truncation=True))[1:-1]
                                            line_ngram = list(nltk.ngrams(line, len(label)))
                                            if label not in line_ngram:
                                                label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=False,max_length=512,truncation=True))
                                                line_ngram = list(nltk.ngrams(line, len(label)))
                                                if label not in line_ngram:
                                                    label = tuple(tokenizer.encode(f'a {tokens_orig[idx]} a',max_length=512,truncation=True))[1:-1]
                                                    line_ngram = list(nltk.ngrams(line, len(label)))
                                                    if label not in line_ngram:
                                                        label = tuple([tokenizer.encode(f'{tokens_orig[idx]}2',max_length=512,truncation=True)[0]])
                                                        line_ngram = list(nltk.ngrams(line, len(label)))
                                            idx = line_ngram.index(label)
                                            neutral_examples[i].append(line)
                                            neutral_labels[i].append([idx + i for i in range(len(label))])
                        else:
                            neutral_examples.append(tokenizer.encode(line, add_special_tokens=True,max_length=512,truncation=True))

    for i, examples in enumerate(neutral_examples):
        print(f'neutral{i}:', len(examples))
    for i, examples in enumerate(attributes_examples):
        print(f'attributes{i}:', len(examples))

    data = {'attributes_examples': attributes_examples,
            'attributes_labels': attributes_labels,
            'neutral_examples': neutral_examples}

    if data:
        print("data not empty") 
    if args.stereotypes:
        data['neutral_labels'] = neutral_labels

    torch.save(data, args.output + '/data.bin')
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

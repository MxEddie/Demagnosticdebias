import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch

# import spacy
from transformers import BertModel, BertTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel,GPT2Model
# from allennlp.commands.elmo import ElmoEmbedder
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import logging
logging.getLogger('transformers.tokenization_utils').disabled = True
import numpy as np
import json
import pickle
import datetime
# import spacy
# from allennlp.commands.elmo import ElmoEmbedder

print(torch.cuda.is_available())


#tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')
#model_bert = BertModel.from_pretrained('bert-base-cased')
tokenizer_bert = BertTokenizer.from_pretrained("/rds/user/hpcungl1/hpc-work/debiased_models/21/bert/0.00005/0.2/checkpoint-best")
model_bert = BertModel.from_pretrained("/rds/user/hpcungl1/hpc-work/debiased_models/21/bert/0.00005/0.2/checkpoint-best")
model_bert.eval()
model_bert.to('cuda')

#weat 

EA = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett',  'Matthew',
'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie',  'Meredith', 'Sarah','Neil']

AA= ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed',
 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya',   'Latoya', 'Tamika',
'Tanisha','Lakisha']

MF = ['Maria','Yesenia','Adriana','Liset','Mayra','Alma',
'Carolina','Iliana','Sonia','Karina','Alejandra','Brenda']

EAM = ['Roger', 'Neil', 'Geoffrey',
 'Brad','Peter', 'Josh', 'Andrew',  'Jack',
'Matthew', 'Jonathan',  'Greg']

#warm = ['social', 'open', 'understanding', 'popular', 'civil', 'nice', 'warm', 'friendly','good', 'right', 'kind', 'moral', 'fair', 'responsible', 'honest', 'innocent']

#cold = ['cold', 'rough', 'distant', 'unpleasant', 'shy', 'disliked', 'unfriendly', 'rude','mean','bad', 'wrong', 'evil', 'criminal', 'hostile', 'vicious', 'unfair']

#competent = ['able', 'effective', 'bright', 'practical', 'capable', 'critical', 'brilliant', 'creative','determined', 'active', 'independent', 'dominant', 'secure', 'dedicated', 'aggressive', 'confident']

#incompetent = ['unable', 'stupid', 'naive', 'foolish', 'dumb', 'ignorant', 'incapable', 'irrational', 'dependent', 'anxious', 'doubtful', 'helpless', 'dominated', 'vulnerable', 'cautious', 'meek']

warm = ['social', 'open', 'understanding', 'popular', 'civil', 'nice', 'warm', 'friendly', 'sensitive', 'liked', 'funny', 'pleasant', 'sympathetic', 'helpful', 'humorous', 'sentimental','good', 'right', 'kind', 'moral', 'fair', 'responsible', 'honest', 'innocent', 'genuine', 'generous', 'reliable', 'cooperative', 'loyal', 'sincere', 'honorable', 'faithful']

cold = ['cold', 'rough', 'distant', 'unpleasant', 'shy', 'disliked', 'unfriendly', 'rude', 'unpopular', 'aloof', 'boring', 'nasty', 'timid', 'insensitive', 'unsympathetic', 'antisocial','mean','bad', 'wrong', 'evil', 'criminal', 'hostile', 'vicious', 'unfair', 'fake', 'irresponsible', 'selfish', 'corrupt', 'brutal', 'treacherous', 'immoral', 'cunning']

competent = ['able', 'effective', 'bright', 'practical', 'capable', 'critical', 'brilliant', 'creative', 'wise', 'logical', 'efficient', 'competitive', 'skilled', 'intelligent', 'rational', 'competent','determined', 'active', 'independent', 'dominant', 'secure', 'dedicated', 'aggressive', 'confident', 'persistent', 'ambitious', 'daring', 'energetic', 'conscientious', 'motivated', 'resolute', 'industrious']

incompetent = ['unable', 'stupid', 'naive', 'foolish', 'dumb', 'ignorant', 'incapable', 'irrational', 'inefficient', 'clumsy', 'impractical', 'unwise', 'incompetent', 'inept', 'uneducated', 'unimaginative','dependent', 'anxious', 'doubtful', 'helpless', 'dominated', 'vulnerable', 'cautious', 'meek', 'lazy', 'careless', 'inactive', 'sporadic', 'submissive', 'docile', 'insecure', 'negligent']

pleasant = [ 'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation']

unpleasant = ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster',
'hatred', 'pollute', 'tragedy', 'bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'evil', 'kill', 'rotten', 'vomit']

mf_bias = ['feisty','curvy', 'loud',  'cook', 'darkskinned', 'uneducated', 'hardworker' ,'promiscuous','unintelligent','short','sexy', 'maids']

eam_bias = ['rich', 'arrogant', 'status', 'blond', 'racist', 'American', 'leader', 'privileged',  'tall', 'sexist', 'successful'] #remove intelligent as overlap with SCM words


def short_sen(sen,wd):
    """
    shorten the raw comment, take only 9 words including the target word
    """
    wds = sen.split()
    #wds = sen.lower().split()
    wd_idx = wds.index(wd)
    if len(wds) >=9:
        if wd_idx < 4:
            wds_used = wds[:9]
        elif (len(wds) - wd_idx - 1 < 4):
            wds_used = wds[-9:]
        else:
            wds_used = wds[(wd_idx-4):(wd_idx+4)]
        new_sen = ' '.join(wds_used)
    else:
        new_sen = sen
    return new_sen

def bert(wd_lst,out_name):
    # load
    sen_dict = pickle.load(open('/rds/user/hpcungl1/hpc-work/sen_dic_1_final.pickle','rb'))
    wd_idx_dict = {wd:[] for wd in wd_lst}
    out_dict = {wd:[] for wd in wd_lst}

    # generate wd index dictionary
    for wd in wd_lst:
        #wd = wd.lower()
        current_idx = torch.tensor(tokenizer_bert.encode(wd,add_special_tokens=False)).unsqueeze(0).tolist()[0]
        wd_idx_dict[wd] = current_idx
    
    # generate embeddings
    i = 0
    for wd in wd_lst:
        #wd = wd.lower()
        target = wd_idx_dict[wd][-1]
        tem = []
        for idx,sen in enumerate(sen_dict[wd]):
            i += 1
            if i%5000 == 0:
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(str(i)+' finished.')
            if idx == 1000:
                break
            # try:
            #     input_ids = torch.tensor(tokenizer_bert.encode(sen, add_special_tokens=False)).unsqueeze(0) 
            #     input_ids = input_ids.to('cuda')
            #     exact_idx = input_ids.tolist()[0].index(target)
            #     outputs = model_bert(input_ids)
            #     exact_state_vector = outputs[0][0,int(exact_idx),:].cpu().detach().numpy() 
            #     out_dict[wd].append(exact_state_vector)
            # except:
            #     # error_sen_dict[wd].append(sen)

            sen = short_sen(sen,wd)
            input_ids = torch.tensor(tokenizer_bert.encode(sen, add_special_tokens=False)).unsqueeze(0) 
            input_ids = input_ids.to('cuda')
            exact_idx = input_ids.tolist()[0].index(target)
            outputs = model_bert(input_ids)
            exact_state_vector = outputs[0][0,int(exact_idx),:].cpu().detach().numpy()  
            out_dict[wd].append(exact_state_vector)
        # out_dict.append(tem)
    n = '/rds/user/hpcungl1/hpc-work/bert_'+out_name+'.pickle'
    pickle.dump(out_dict,open(n,'wb'))
    # pickle.dump(error_sen_dict,open('bert_error_sen.pickle','wb'))


warm_race = EA + AA + warm + cold
competent_race = EA + AA + competent + incompetent 
warm_inter = EAM + MF + warm + cold
competent_inter = EAM + MF + competent + incompetent 
pleasant_race = EA + AA + pleasant + unpleasant
pleasant_inter = EAM + MF + pleasant + unpleasant 
inter_race = EA + AA + eam_bias + mf_bias  
inter_inter = EAM + MF + eam_bias + mf_bias 

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
bert(warm_race,'weat_warm_race')
bert(competent_race,'weat_competent_race')
bert(warm_inter,'weat_warm_inter')
bert(competent_inter,'weat_competent_inter')
bert(pleasant_race,'weat_pleasant_race')
bert(pleasant_inter,'weat_pleasant_inter')
bert(inter_race,'weat_inter_race')
bert(inter_inter,'weat_inter_inter')


print("bert finish")


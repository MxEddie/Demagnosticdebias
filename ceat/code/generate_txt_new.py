
"""
generate text for each word group and save as txt file separately
"""

import string
import datetime
import json
import random
import pickle
import csv

print("code open",flush=True)

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


#warm8 = ['social', 'open', 'understanding', 'popular','good', 'right', 'kind','moral']
#cold8 = ['cold', 'rough', 'distant', 'unpleasant','mean','bad', 'wrong','evil']
#competent8 = ['able', 'effective', 'bright', 'practical','determined', 'active', 'independent','dominant']
#incompetent8 = ['unable', 'stupid', 'naive', 'foolish','dependent', 'anxious', 'doubtful','helpless']
#pleasant8 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
#unpleasant8 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']



now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

N = 10000

wd_lst = list(set(pleasant+unpleasant+EA+AA+MF+EAM+warm+cold+competent+incompetent+mf_bias+eam_bias))

count_d = {i:0 for i in wd_lst}
sen_d = {i:[] for i in wd_lst}


file_path = '/rds/user/hpcungl1/hpc-work/RC_2014'

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
    
print(file_path)

with open(file=file_path,mode='r',encoding='utf-8') as f:
    #json_list = list(f)
    #for i in range(0,len(json_list)-1):
    for idx,line in enumerate(f):
        if idx % 1000000 == 0:
            print(idx,flush=True)
        comment_line = json.loads(line)
        #json_str = json_list[i]
        #comment_line = json.loads(json_str)
        comment = comment_line["body"]
        if comment != '[deleted]' and comment != '[removed]':
            comment = comment.replace("&gt;"," ")
            comment = comment.replace("&amp;"," ")
            comment = comment.replace("&lt;"," ")
            comment = comment.replace("&quot;"," ")
            comment = comment.replace("&apos;"," ")
            comment = comment.translate(translator)
            for wd in wd_lst:
                wwd = " "+wd+" "
                if wwd in comment:
                    count_d[wd] += 1
                    if count_d[wd] <= N:
                        sen_d[wd].append(comment)
                    elif (count_d[wd]>N) and (random.random() < N/float(count_d[wd]+1)):
                        replace = random.randint(0,len(sen_d[wd])-1)
                        sen_d[wd][replace] = comment
        if idx == 250000000:
           break
            # if idx == 5:
            #     break

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("pickle dump",flush=True)

pickle.dump(sen_d,open('/rds/user/hpcungl1/hpc-work/sen_dic_1_final.pickle','wb'))
pickle.dump(count_d,open('/rds/user/hpcungl1/hpc-work/count_dic_1_final.pickle','wb'))

with open('/rds/user/hpcungl1/hpc-work/count_1_final.csv', 'w') as f:
    for key in count_d.keys():
        f.write("%s,%s\n"%(key,count_d[key]))

with open('/rds/user/hpcungl1/hpc-work/count_sample_1_final.csv', 'w') as f:
    for key in sen_d.keys():
        f.write("%s,%s\n"%(key,len(sen_d[key])))
     

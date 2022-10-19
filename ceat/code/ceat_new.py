import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy.stats
import time as t
import pickle
import random
import datetime

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


weat_groups = [[EA,AA,warm,cold],[EA,AA,competent,incompetent],[EAM,MF,warm,cold],[EAM,MF,competent,incompetent],[EA,AA,pleasant,unpleasant],[EAM,MF,pleasant,unpleasant],[EA,AA,eam_bias,mf_bias],[EAM,MF,eam_bias,mf_bias]]

#weat_groups = [[european,african,warm,cold],[european,african,competent,incompetent],[european,african,pleasant,unpleasant],[male,female,warm8,cold8],[male,female,competent8,incompetent8],[male,female,pleasant8,unpleasant8]]


def associate(w,A,B):
    return cosine_similarity(w.reshape(1,-1),A).mean() - cosine_similarity(w.reshape(1,-1),B).mean()

def difference(X,Y,A,B):
    # return np.sum(np.apply_along_axis(associate,1,X,A,B)) - np.sum(np.apply_along_axis(associate,1,Y,A,B))

    return np.sum([associate(X[i,:],A,B) for i in range(X.shape[0])]) - np.sum([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

def effect_size(X,Y,A,B):
    # delta_mean = np.mean(np.apply_along_axis(associate,1,X,A,B)) - np.mean(np.apply_along_axis(associate,1,Y),A,B)
    delta_mean =  np.mean([associate(X[i,:],A,B) for i in range(X.shape[0])]) - np.mean([associate(Y[i,:],A,B) for i in range(Y.shape[0])])

    # s = np.apply_along_axis(associate,1,np.concatenate((X,Y),axis=0),A,B)
    XY = np.concatenate((X,Y),axis=0)
    s = [associate(XY[i,:],A,B) for i in range(XY.shape[0])]

    std_dev = np.std(s,ddof=1)
    var = std_dev**2

    return delta_mean/std_dev, var

def inn(a_huge_key_list):
    L = len(a_huge_key_list)
    i = np.random.randint(0, L)
    return a_huge_key_list[i]


def sample_statistics(X,Y,A,B,num = 100):
    XY = np.concatenate((X,Y),axis=0)
   
    def inner_1(XY,A,B):
        X_test_idx = np.random.choice(XY.shape[0],X.shape[0],replace=False)
        Y_test_idx = np.setdiff1d(list(range(XY.shape[0])),X_test_idx)
        X_test = XY[X_test_idx,:]
        Y_test = XY[Y_test_idx,:]
        return difference(X_test,Y_test,A,B)
    
    s = [inner_1(XY,A,B) for i in range(num)]

    return np.mean(s), np.std(s,ddof=1)

def p_value(X,Y,A,B,num=100):
    m,s = sample_statistics(X,Y,A,B,num)
    d = difference(X,Y,A,B)
    p = 1 - scipy.stats.norm.cdf(d,loc = m, scale = s)
    return p

def ceat_meta(weat_groups = weat_groups,group="warm_race",model='bert',test=1,N=10000):
    nm = "/rds/user/hpcungl1/hpc-work/bert_weat_{}.pickle".format(group)
    print(nm)
    weat_dict = pickle.load(open(nm,'rb'))
    # nm_1 = "name_{}_vector_new.pickle".format(model)
    # name_dict = pickle.load(open(nm_1,'rb'))  

    e_lst = [] #effect size
    v_lst = [] #variance

    len_list = [len(weat_groups[test-1][i]) for i in range(4)]
    for i in range(N):

        X = np.array([weat_dict[wd][np.random.randint(0,len(weat_dict[wd]))] for wd in weat_groups[test-1][0]])
        Y = np.array([weat_dict[wd][np.random.randint(0,len(weat_dict[wd]))] for wd in weat_groups[test-1][1]])
        for wd in weat_groups[test-1][3]:
            try:
                weat_dict[wd][np.random.randint(0,len(weat_dict[wd]))]
            except:
                print(wd)
        A = np.array([weat_dict[wd][np.random.randint(0,len(weat_dict[wd]))] for wd in weat_groups[test-1][2]])
        B = np.array([weat_dict[wd][np.random.randint(0,len(weat_dict[wd]))] for wd in weat_groups[test-1][3]])
        e,v = effect_size(X,Y,A,B)
        e_lst.append(e)
        v_lst.append(v)

    e_nm = "/rds/user/hpcungl1/hpc-work/{0}_{1}_es.pickle".format(model,test)
    v_nm = "/rds/user/hpcungl1/hpc-work/{0}_{1}_v.pickle".format(model,test)
    pickle.dump(e_lst,open(e_nm,'wb'))
    pickle.dump(v_lst,open(v_nm,'wb'))
    
    #calculate Q (total variance)
    e_ary = np.array(e_lst)
    w_ary = 1/np.array(v_lst)

    q1 = np.sum(w_ary*(e_ary**2))
    q2 = ((np.sum(e_ary*w_ary))**2)/np.sum(w_ary)
    q = q1 - q2

    df = N - 1

    if q>df:
        c = np.sum(w_ary) - np.sum(w_ary**2)/np.sum(w_ary)
        tao_square = (q-df)/c
        print("tao>0")
    else:
        tao_square = 0

    v_ary = np.array(v_lst)
    v_star_ary = v_ary + tao_square
    w_star_ary = 1/v_star_ary

    # calculate combiend effect size, variance
    pes = np.sum(w_star_ary*e_ary)/np.sum(w_star_ary)
    v = 1/np.sum(w_star_ary)

    # p-value
    z = pes/np.sqrt(v)
    # p_value = 1 - scipy.stats.norm.cdf(z,loc = 0, scale = 1)
    p_value = scipy.stats.norm.sf(z,loc = 0, scale = 1)
    #p_value = p_value/ np.log(10)


    return pes, p_value


if __name__ == '__main__':

    e_lst = []
    p_lst = []
    group_list = ["warm_race","competent_race","warm_inter","competent_inter","pleasant_race","pleasant_inter","inter_race","inter_inter"]
    for e in range(1,len(weat_groups)+1):
        # group = weat_groups[(e - 1)]
        e_lst.append([])
        p_lst.append([])
        print(e)
        g = group_list[e-1]
        print(g)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

        pes,  p_value = ceat_meta(weat_groups = weat_groups,group=g, model='bert',test=e,N=10000)
        print("PES is {}:".format(pes))
        print("P-value is {}:".format(p_value))
        e_lst[e-1].append(pes)
        e_lst[e-1].append(p_value)
        print(" ")
    
    e_ary = np.array(e_lst)
    p_ary = np.array(p_lst)

    np.savetxt("/rds/user/hpcungl1/hpc-work/e_10000.csv", e_ary, delimiter=",")

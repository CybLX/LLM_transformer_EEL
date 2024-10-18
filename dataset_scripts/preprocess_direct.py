#%%
from datetime import datetime
start_time = datetime.now()
print('START AT: {}'.format(start_time))


from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell() 
sym_spell.create_dictionary('/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/dicionarios/br-sem-acentos.txt')
slangs = eval(open('./dataset/dicionarios/slang.txt', 'r').read())



import pandas as pd
import re
import unidecode
import os

@staticmethod
def preprocess_text(text):
    text = re.sub('&quot;', ' ', text)
    text = re.sub('&amp;','',text)
    text = re.sub('039;','',text)
    text = re.sub('[||]','',text)
    text = re.sub("[*)@;#%'(&-?!:_^]", ' ', text, flags=re.MULTILINE)
    text = text.split()
    text = [unidecode.unidecode(word.replace(word,slangs[word]).lower()) if word in slangs.keys() else unidecode.unidecode(word.lower()) for word in text]
    text = [sym_spell.lookup(token, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0].term for token in text]
    return text

#%%
                 
#######################################################
#                B2W
print('PROCESS B2W CORPUS...')
#######################################################

b2w = pd.read_csv('/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/datasets_train/Portugues/B2W/B2W-Reviews01.csv', low_memory = False)
b2w = b2w[['overall_rating','site_category_lv1','review_text']]
nans_b2w = b2w['review_text'].isna().sum()/len(b2w) * 100
b2w = b2w.dropna(axis = 0)

polarity = []
for r in b2w['overall_rating']:
    if r < 3:
        polarity.append('negative')
    elif r > 3:
        polarity.append('positive')
    else:
        polarity.append('neutral')
b2w['polarity'] = polarity



b2w['review_text'] = [preprocess_text(token) for token in b2w['review_text']]
b2w['site_category_lv1'] =[preprocess_text(token) for token in b2w['site_category_lv1']]
b2w = b2w.rename(columns = {'site_category_lv1':'product','review_text' : 'comentario','overall_rating':'rating'})

# Statisticas
polarity_count = b2w['polarity'].value_counts()
pol_statis = pd.DataFrame(polarity_count).rename(columns = {'polarity' : 'b2w'})



             

#######################################################
#               Buscape
print('PROCESS BUSCAPE CORPUS...')
#######################################################
dir_main = '/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/datasets_train/Portugues/Buscape/reviews'

arq = os.listdir(dir_main)

buscape = []
for title in arq:
    dir_title = dir_main+'/'+title
    scores = os.listdir(dir_title)
    
    title_comments = []
    for score in scores:
        review_scores = dir_title+'/'+score
        files = [f for f in os.listdir(review_scores) if f.endswith('.txt')]
        
        for_score = []
        for f in files:  
            txt = open(review_scores+'/'+f).read().splitlines()
            review = ''.join(i for i in txt)
            review = preprocess_text(review)
            review = ' '.join(i for i in review)

            try:
                like_index = review.index('o que gostei')
            except:
                like_index = None
            
            try:
                notlike_index = review.index('o que nao gostei')
            except:
                notlike_index = None

            if like_index and notlike_index != None:
                comment = review[: like_index].split()
                like = review[like_index+13 : notlike_index].split()
                notlike = review[notlike_index+17: ].split()

            elif like_index != None and notlike_index == None:
                comment = review[: like_index].split()
                like = review[like_index+13:].split()
                notlike = [pd.NA]

            elif like_index == None and notlike_index != None:
                comment = review[: notlike_index].split()
                like = [pd.NA]
                notlike = review[notlike_index+17:].split()
            else:
                comment = review.split()
                like = [pd.NA]
                notlike = [pd.NA]

            review_file = pd.DataFrame([[comment],[like],[notlike]], index = ['comentario','gostou','nao_gostou']).T
            for_score.append(review_file)
       
        for_score = pd.concat(for_score, axis = 0).reset_index(drop = True)
        for_score['rating'] = int(float(score))
        title_comments.append(for_score)
        
    title_comments = pd.concat(title_comments, axis = 0).reset_index(drop = True)
    title_comments['product']= title.lower()
    buscape.append(title_comments)

buscape = pd.concat(buscape, axis = 0).reset_index(drop = True)

polarity = []
for r in buscape['rating']:
    if r < 3:
        polarity.append('negative')
    elif r > 3:
        polarity.append('positive')
    else:
        polarity.append('neutral')
buscape['polarity'] = polarity

# Statisticas
nans_buscape = buscape[['gostou','nao_gostou']].isna().sum()/len(buscape) * 100
prop_nasn_buscape = len(buscape.dropna(axis = 0))/len(buscape)*100
pol_statis = pd.concat([pd.DataFrame(buscape['polarity'].value_counts()).rename(columns = {'polarity':'buscape'}),pol_statis], axis = 1)




#######################################################
#               OLIST
print('PROCESS OLIST CORPUS...')
#######################################################
olist = pd.read_csv('/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/datasets_train/Portugues/Olist/olist.csv')
olist = olist[['review_text','rating']]
olist['review_text'] =[preprocess_text(token) for token in olist['review_text']]

polarity = []

for r in olist['rating']:
    if r<3:
        polarity.append('negative')
    elif r>3:
        polarity.append('positive')
    else:
        polarity.append('neutral')
olist['polarity'] = polarity
olist = olist.rename(columns = {'review_text':'comentario'})

# Statisticas
pol_statis = pd.concat([pd.DataFrame(olist['polarity'].value_counts()).rename(columns = {'polarity':'olist'}),pol_statis], axis = 1)




#######################################################
#               twiter
print('PROCESS TWITTER CORPUS...')
#######################################################
dir_path = '/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/datasets_train/Portugues/Tweet'
files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]


twites = []
for arq in files:
    arq_path = dir_path+'/'+arq
    dt = pd.read_csv(arq_path)
    dt['product'] = arq[:-4]
    twites.append(dt)
twites = pd.concat(twites, axis = 0).reset_index(drop = True)
twites = twites[['tweet_text','sentiment','product']]
twites['sentiment'] = twites['sentiment'].replace({'Positivo': 'positive','Negativo':'negative','Neutro':'neutral'})

rashs = []
for i in twites['tweet_text']:
    i = i.split()
    i = [ x for x in i if '@' not in x ]
    i = ' '.join(i)
    rashs.append(i)
twites['tweet_text'] = rashs
twites['tweet_text'] =[preprocess_text(token) for token in twites['tweet_text']]
twites = twites.rename(columns = {'tweet_text': 'comentario','sentiment': 'polarity'})

# Statisticas
pol_statis = pd.concat([pd.DataFrame(twites['polarity'].value_counts()).rename(columns = {'polarity':'twitter'}),pol_statis], axis = 1)

#%%

        
#######################################################
#               UTL CORPUS
print('PROCESS UTL CORPUS...')
#######################################################
dir_path = '/home/lua_lex/Documentos/ic _/comentarios_eel/code/dataset/datasets_train/Portugues/UTLCORPUS'
files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

utl = []
for arq in files:
    arq_path = dir_path+'/'+arq
    dt = pd.read_csv(arq_path)
    dt = dt[['review_text','rating']]
    dt['product'] = arq[5:-4]
    utl.append(dt)
utl = pd.concat(utl, axis = 0).reset_index(drop =True)

polarity = []
for r in utl['rating']:
    if r<3:
        polarity.append('negative')
    elif r>3:
        polarity.append('positive')
    else:
        polarity.append('neutral')
utl['polarity'] = polarity

utl['review_text'] =[preprocess_text(token) for token in utl['review_text']]
utl =utl.rename(columns = {'review_text':'comentario'})
# Statisticas
pol_statis = pd.concat([pd.DataFrame(utl['polarity'].value_counts()).rename(columns = {'polarity':'utl'}),pol_statis], axis = 1)

#%%

# Compilando estatisticas
pol_statis['total'] = pol_statis.sum(axis = 1)
pol_statis.loc['total'] = pol_statis.sum(axis= 0)
pol_statis.index.name = 'Amostras'

prop = []
for i in pol_statis.columns:
    x = pol_statis[i]/pol_statis['total']*100
    x = pd.DataFrame(x, columns = [i])
    prop.append(x)
prop = pd.concat(prop, axis = 1)
prop.index.name = 'Classes do total(%)'

props = []
for i in pol_statis.columns:
    x = pol_statis[i]/pol_statis[i].loc['total']*100
    x = pd.DataFrame(x, columns = [i])
    props.append(x)
props = pd.concat(props, axis = 1)
props.index.name = 'classes dos corpus(%)'


end_time = datetime.now()
print('Finish Duration: {}'.format(end_time - start_time))


col = ['comentario','polarity']
fs = pd.concat([b2w[col], buscape[col],utl[col],twites[col],olist[col]], axis = 0)
ss = []
for i in fs['comentario']:
    ss.append(' '.join([j for j in i]))
fss = fs.copy()
fss['comentario'] = ss
fss.to_csv('./captions.csv')
# %%

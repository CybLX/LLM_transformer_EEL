#%%
import pandas as pd
import re
import unidecode
import os
from datetime import datetime
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import spacy
from nltk.stem import RSLPStemmer




class ProcessCorpus:

    def __init__(self):
        self.sym_spell = SymSpell() 
        self.sym_spell.create_dictionary('./dataset/dicionarios/br-sem-acentos.txt')
        self.slangs = eval(open('./dataset/dicionarios/slangs.txt', 'r').read())

        self.stemmer = RSLPStemmer()
        self.nlp = spacy.load("pt_core_news_lg")

    def preprocess_text(self,text):
        text = re.sub('&quot;', ' ', text)
        text = re.sub('&amp;','',text)
        text = re.sub('039;','',text)
        text = re.sub('[||]','',text)
        text = re.sub("[*)@;#%'(&-?!:_^]", ' ', text, flags=re.MULTILINE)
        text = text.split()
        text = [unidecode.unidecode(word.replace(word,self.slangs[word]).lower()) if word in self.slangs.keys() else unidecode.unidecode(word.lower()) for word in text]
        text = [self.sym_spell.lookup(token, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0].term for token in text]        
        return text

    def pos_tags(self,data):
        lem_ = []
        pos_ = []
        tag_ = []
        dep_ = []
        for sentence in data['comentario']:
            sentence = self.nlp(sentence)

            lem_.append(' '.join(i for i in [token.lemma_ for token in sentence]))
            pos_.append(' '.join(i for i in [token.pos_ for token in sentence]))
            tag_.append(' '.join(i for i in [token.tag_ for token in sentence]))
            dep_.append(' '.join(i for i in [token.dep_ for token in sentence]))
        return lem_,pos_,tag_,dep_

    def stemizer(self,data):
        stem_ = []
        for sentence in data['comentario']:
            stem_.append(' '.join(i for i in [self.stemmer.stem(token) for token in sentence.split() if token not in stopwords.words('portuguese')]))
        return stem_
    
    def B2Wcorpus(self):
        b2w = pd.read_csv('./dataset/datasets_train/Portugues/B2W/B2W-Reviews01.csv', low_memory = False)
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

        b2w['review_text'] = [' '.join(i for i in self.preprocess_text(token)) for token in b2w['review_text']]
        b2w['site_category_lv1'] =[self.preprocess_text(token) for token in b2w['site_category_lv1']]
        b2w = b2w.rename(columns = {'site_category_lv1':'product','review_text' : 'comentario','overall_rating':'rating'})
        
        b2w['tokens_stem'] = self.stemizer(b2w)
        b2w['tokens_lemma'],b2w['simple_POS'],b2w['detailed_POS'],b2w['Syntactic_dependency'] = self.pos_tags(b2w)
         

        # Statisticas
        polarity_count = b2w['polarity'].value_counts()
        pol_statis = pd.DataFrame(polarity_count).rename(columns = {'polarity' : 'b2w'})
        
        return b2w, pol_statis
    
    def BUSCAPEcorpus(self):
        dir_main = './dataset/datasets_train/Portugues/Buscape/reviews'
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
                    review = self.preprocess_text(review)
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
                        comment = review[: like_index]
                        like = review[like_index+13 : notlike_index]
                        notlike = review[notlike_index+17: ]

                    elif like_index != None and notlike_index == None:
                        comment = review[: like_index]
                        like = review[like_index+13:]
                        notlike = [pd.NA]

                    elif like_index == None and notlike_index != None:
                        comment = review[: notlike_index]
                        like = [pd.NA]
                        notlike = review[notlike_index+17:]
                    else:
                        comment = review
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
        buscape['tokens_stem'] = self.stemizer(buscape)
        buscape['tokens_lemma'],buscape['simple_POS'],buscape['detailed_POS'],buscape['Syntactic_dependency'] = self.pos_tags(buscape)
        
        # Statisticas
        nans_buscape = buscape[['gostou','nao_gostou']].isna().sum()/len(buscape) * 100
        prop_nasn_buscape = len(buscape.dropna(axis = 0))/len(buscape)*100
        pol_statis = pd.DataFrame(buscape['polarity'].value_counts()).rename(columns = {'polarity':'buscape'})

        return buscape,pol_statis
    
    def OLISTcorpus(self):
        olist = pd.read_csv('./dataset/datasets_train/Portugues/Olist/olist.csv')
        olist = olist[['review_text','rating']]
        olist['review_text'] = [' '.join(i for i in self.preprocess_text(token)) for token in olist['review_text']]
        olist = olist.rename(columns = {'review_text':'comentario'})
        polarity = []

        for r in olist['rating']:
            if r<3:
                polarity.append('negative')
            elif r>3:
                polarity.append('positive')
            else:
                polarity.append('neutral')

        olist['polarity'] = polarity
        olist['tokens_stem'] = self.stemizer(olist)
        olist['tokens_lemma'],olist['simple_POS'],olist['detailed_POS'],olist['Syntactic_dependency'] = self.pos_tags((olist))
        
        
        pol_statis = pd.DataFrame(olist['polarity'].value_counts()).rename(columns = {'polarity':'olist'})

        return olist, pol_statis
    
    def TWITTERcorpus(self):
        dir_path = './dataset/datasets_train/Portugues/Tweet'
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
        twites['tweet_text'] = [' '.join(i for i in self.preprocess_text(token)) for token in twites['tweet_text']]
        twites = twites.rename(columns = {'tweet_text': 'comentario','sentiment': 'polarity'})

        twites['tokens_stem'] = self.stemizer(twites)
        twites['tokens_lemma'],twites['simple_POS'],twites['detailed_POS'],twites['Syntactic_dependency'] = self.pos_tags(twites)
        pol_statis = pd.DataFrame(twites['polarity'].value_counts()).rename(columns = {'polarity':'twitter'})

        return twites, pol_statis
    
    def UTLcorpus(self):
        dir_path = './dataset/datasets_train/Portugues/UTLCORPUS'
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
        
        utl['review_text'] = [' '.join(i for i in self.preprocess_text(token)) for token in utl['review_text']]
        utl = utl.rename(columns = {'review_text':'comentario'})  

        utl['tokens_stem'] = self.stemizer(utl)
        utl['tokens_lemma'],utl['simple_POS'],utl['detailed_POS'],utl['Syntactic_dependency'] = self.pos_tags(utl)
        
             
        pol_statis = pd.DataFrame(utl['polarity'].value_counts()).rename(columns = {'polarity':'utl'})
        return utl, pol_statis


def start():
    start_time = datetime.now()
    
    print("\n"*2)
    print("*"*20)
    print('START AT: {}'.format(start_time))
    print("*"*20,'\n'*2)
    
    pc = ProcessCorpus()
    
    print('PROCESS B2W CORPUS... \n')
    b2w,pol0 = pc.B2Wcorpus()
    print(b2w.info(),'\n',f"qta NAN: {b2w.isna().sum()}",'\n'*2)

    print('PROCESS BUSCAPE CORPUS... \n')
    buscape,pol1 = pc.BUSCAPEcorpus()
    print(buscape.info(),'\n',f"qta NAN: {buscape.isna().sum()}",'\n'*2)

    print('PROCESS OLIST CORPUS... \n')
    olist,pol2 = pc.OLISTcorpus()
    print(olist.info(),'\n',f"qta NAN: {olist.isna().sum()}",'\n'*2)

    print('PROCESS TWITTER CORPUS... \n')
    twites,pol3 = pc.TWITTERcorpus()
    print(twites.info(),'\n',f"qta NAN: {twites.isna().sum()}",'\n'*2)

    print('PROCESS UTL CORPUS... \n')
    utl,pol4 = pc.UTLcorpus()
    print(utl.info(),'\n',f"qta NAN: {utl.isna().sum()}",'\n'*2)

    pol_statis = pd.concat([pol0,pol1,pol2,pol3,pol4],axis=1)
    
    print('SAVING THE RESULTS... \n')
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
    prop.to_csv('./Classes_total.csv')
    
    props = []
    for i in pol_statis.columns:
        x = pol_statis[i]/pol_statis[i].loc['total']*100
        x = pd.DataFrame(x, columns = [i])
        props.append(x)
    props = pd.concat(props, axis = 1)
    props.index.name = 'classes dos corpus(%)'
    props.to_csv('./classes_corpus.csv')
    

    # Select Features

    col = ['comentario','polarity','tokens_stem','tokens_lemma','simple_POS','detailed_POS','Syntactic_dependency']
    fs = pd.concat([b2w[col], buscape[col],utl[col],twites[col],olist[col]], axis = 0)
    fs = fs.astype({'comentario' : 'string',
                'tokens_stem' : 'string',
                'tokens_lemma' : 'string',
                'simple_POS': 'string',
                'detailed_POS': 'string',
                'Syntactic_dependency': 'string',
                'polarity':'string'})
    fs.to_csv('./corpus.csv')

    end_time = datetime.now()
    print("*"*20)
    print('Finish Duration: {}'.format(end_time - start_time))
    print("*"*20,'\n'*2)
    print(fs.head(2), fs.info(),f"qta NAN: {fs.isna().sum()}",'\n'*2)

# 6 horas de duracao

start()




# %%

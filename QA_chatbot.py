import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
df = pd.read_csv("D:/web/sophia/model/QA/question_answer_132.csv")
df1 = pd.read_csv("D:/web/sophia/model/QA/question_concat.csv",header=None)

class chatbot:

    def chatbot_text(self,text):
        em_result = model.encode(text)

        co_result = []
        for temp in range(len(df1)):
            data = df1.iloc[temp]
            co_result.append(cosine_similarity([data],[em_result])[0][0] )

        df['cos'] = co_result
        df_result = df.sort_values('cos',ascending=False)
        r = random.randint(0,2)
        # df_result['A']

        print(df_result.iloc[r]['A'])
        return df_result.iloc[r]['A']


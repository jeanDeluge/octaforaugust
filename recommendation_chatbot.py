import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import numpy as np
import os

df=pd.read_csv('D:/web/sophia/model/recommend/newfile_concat.csv')
df1 = pd.read_csv('D:/web/sophia/model/recommend/embeding.csv')
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# 추천해주는 함수
def recommend(text):
    em_result = model.encode(text)
    co_result = []
    for temp in range(len(df1)):
        data = df1.iloc[temp]
        co_result.append(cosine_similarity([data], [em_result])[0][0])

    df['cos'] = co_result
    df_result = df.sort_values('cos', ascending=False)
    r = random.randint(0, 5)

    word = f"책 제목: {df_result.iloc[r]['title']}\n저자: {df_result.iloc[r]['author']}\n출판사: {df_result.iloc[r]['publisher']}\n리뷰 수: {int(df_result.iloc[r]['reviewnum'])}\n집중돼요/도움돼요/쉬웠어요/최고에요/추천해요 비율: {df_result.iloc[r]['score']}"
    #print(word)
    return word


recommend('자유')
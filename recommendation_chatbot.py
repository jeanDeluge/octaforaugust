import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np


model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
df = pd.read_csv("D:/web/sophia/model/recommend/recommend_final.csv") ##책 크롤링 내용
df1 = pd.read_csv("D:/web/sophia/model/recommend/recommend_embedding.csv", header=None) ##책 크롤링 벡터화

class Recommend:

    def recommend_book(self,text):
        em_result = model.encode(text)

        co_result = []
        for temp in range(len(df1)):
            data = df1.iloc[temp]
            co_result.append( cosine_similarity([data],[em_result])[0][0] )

        df['cos'] = co_result
        df_result = df.sort_values('cos',ascending=False)
        r = random.randint(0,5)

        word = f"책 제목: {df_result.iloc[r]['title']}\n저자: {df_result.iloc[r]['author']}\n출판사: {df_result.iloc[r]['publisher']}\n리뷰 수: {int(df_result.iloc[r]['reviewnum'])}\n집중돼요/도움돼요/쉬웠어요/최고에요/추천해요 비율: {df_result.iloc[r]['score']}"
        print(word)
        return word


# a = Recommend()
# a.recommend_book("머신러닝") 

## 키워드를 자유롭게 입력하면, 키워드 코사인 유사도를 분석해서 가장 비슷한 책을 추천해줌.
## 자유주의 국가 정치    <<-- 와 같은 형태로 띄어쓰기로 구분되어있음 입력 받을때 해당 형태로 입력받으면 가장 좋음
# a.recommend_book('자유 행정 가난 ')


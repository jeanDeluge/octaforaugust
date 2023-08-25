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
        # r = random.randint(0,2
        r = random.randint(0,3)
        # df_result['A']

        print(df_result.iloc[r]['A'])
        return df_result.iloc[r]['A']

##도서관에 관한 질문 예) chatbot_text("회원가입")
## 결과 -> 회원카드를 분실하신 경우 핸드폰에서 광진구통합도서관 어플리케이션을 다운로드 받아 실물 회원증 대신 사용하시거나, 
## 도서관에 방문하시어 재발급하셔야 합니다. 회원카드 재발급 시 수수료 2,000원이 발생합니다. 
## (단, 회원카드를 분실로 인해 발생하는 책임은 이용자에게 있습니다.)
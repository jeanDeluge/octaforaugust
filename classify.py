from ultralytics import YOLO
import cv2
from easyocr import Reader
from PIL import Image
from glob import glob
import re
from collections import Counter
import os
import datetime as dt
import pandas as pd


modelpath = 'd:/web/sophia/model_ai/book_detection/book_detection.pt'
img_paths = 'd:/web/sophia/tmp/received/'
img_paths = glob(img_paths+'*.jpg')

class Classify:
    def __init__(self,imagepath) -> None :
        self.ocr = Reader(['ko'])
        self.model = YOLO(modelpath)
        self.predict_result = self.model.predict(imagepath)
        self.imagepath = imagepath



    def get_unsorted_book(self):

        path = 'd:/web/sophia/tmp/cropped_images/{}/'.format(dt.datetime.now().strftime(("%Y%m%d%H%M%S")))
        os.makedirs(path, exist_ok=True)

        img = cv2.imread(self.imagepath)
        result = self.model.predict(img, conf=0.05)
        count = 1
        book_label_boxes = []
        book_label_reversed_boxes = []
        for cnt,i in enumerate(result[0].boxes.data):
            x1,y1 = int(result[0].boxes.data[cnt][0]),int(result[0].boxes.data[cnt][1])
            x2,y2 = int(result[0].boxes.data[cnt][2]),int(result[0].boxes.data[cnt][3])
            if i[-1] == 2:
                crop_img = img[y1:y2, x1:x2]
                cv2.imwrite(path+str(count).zfill(4)+'.jpg', crop_img)
                count+=1
                book_label_boxes.append([x1,y1,x2,y2])
            if i[-1] ==3:
                book_label_reversed_boxes.append([x1,y1,x2,y2])

        PATH = path

        img_paths = glob(PATH+'*.jpg')

        ##OCR모델 초기화
        reader = Reader(['ko'])
        a = []
        a_filtered = []
        a_combined = []
        invalid = []
        valid =[]

        for i in img_paths:
            result = reader.readtext(i,detail=0)
            result = ''.join(result)
            a.append(result)

        x_label = ['000','100','200','300','400','500','600','700''800','900']

        remove_dot = []
        for cnt, i in enumerate(a) :
            i = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', i)
            i = re.sub(' ', '', i)

            remove_dot.append((i,cnt))

        a_combined = remove_dot

        except_threenum_combined=[] ##청구기호 갖춘 것들 모음
        for i in a_combined:
            if i[0][:3] in x_label:
                a = i[0][3:]
                except_threenum_combined.append((a,i[1]))
            else: 
                except_threenum_combined.append(i)

        a_combined = except_threenum_combined
        

        ##중복검출
        dup_list = []
        for cnt,i in enumerate(a_combined): ##바운딩 박스의 x,y 좌표값 range가 다른 바운딩 박스의 좌표값과 90프로 이상 일치할 때, 중복으로 파악.
            for cnt2,j in enumerate(a_combined):
                if cnt!=cnt2:           
                    a = [k for k in range(book_label_boxes[i[1]][0],book_label_boxes[i[1]][2],1)] ##x좌표 range
                    b =  [k for k in range(book_label_boxes[j[1]][0],book_label_boxes[j[1]][2],1)]##비교대상 x좌표 range
                    c =  [k for k in range(book_label_boxes[i[1]][1],book_label_boxes[i[1]][3],1)]##y좌표 range
                    d = [k for k in range(book_label_boxes[j[1]][1],book_label_boxes[j[1]][3],1)]##비교대상 y좌표 range

                    count_x = 0 ##x좌표값이 동일한 숫자
                    count_y = 0 ##y좌표값이 동일한 숫자

                    for p in a:
                        if p in b:
                            count_x +=1

                    for p in c:
                        if p in d:
                            count_y +=1


                    if (count_x/len(a)*100>90 or count_x/len(b)*100 >90) and (count_y/len(c)*100>90 or count_y/len(d)*100>90): ##바운딩박스의 길이, 높이를 구해 x,y좌표값이 같은 숫자(count_x,count_y) 를 이용하여 비율이 90프로 이상일 경우 중복으로 인식
                        dup_list.append((i,j))




        dup_unique_set =[]  ##set으로 중복제거 
        for i in dup_list:
            for j in dup_list:
                if set(i)==set(j):
                    if set(i) not in dup_unique_set:
                        dup_unique_set.append(set(i)) ##그러나 아직도 중복원소가 중복되어 나옴 [(('200219', '캠44물', 0), ('200219', '캠44물', 19)),(('200219', '캠44물', 19), ('200219', '캠44물', 0))]

        dup_unique_value = []
        for j in dup_unique_set:
            if sorted(list(j))[0] not in dup_unique_value:
                dup_unique_value.append(sorted(list(j))[0]) ##그 중에서 순서 상관없이 중복되지 않는 세트만 가져옴 (('200219', '캠44물', 0), ('200219', '캠44물', 19))만 남게됨


        dup_value_list =[]
        for i in dup_list:
            if i[0] not in dup_value_list: ##중복되지 않는 세트 중 중복되지 않는 원소 리스트.
                dup_value_list.append(i[0])
        

        except_dup_sorted_list =[]  ##원래 중복된 적이 없던 원소 리스트
        for i in a_combined:
            if i not in dup_value_list:
                except_dup_sorted_list.append(i)

        undup_list = except_dup_sorted_list+dup_unique_value ##  ##둘을 결합, 
        a_combined = undup_list

        a_combined = list(set(a_combined))
        a_combined = sorted(a_combined) ##결합 과정에서 순서가 바뀌므로 순서대로 다시 정렬

        except_threenum_combined=[]

        for i in a_combined:
            if i[0][:3] in x_label:
                a = i[0][3:]
                except_threenum_combined.append((a,i[1]))
            else: 
                except_threenum_combined.append(i)

        a_combined = except_threenum_combined

        reg = re.compile('\d{3}\D*\d*\D\d*\D.*')
        book_valid_list = []
        invalid = []
        for i in a_combined:
            #OCR결과가 청구기호 형식을 갖춘 책 찾는 코드
            if reg.search(i[0]) :
                result = reg.search(i[0])
                foward, back = result.span()
                word = i[0][foward:back]
                book_valid_list.append((word,i[1]))
            # 그렇지 않은 책 찾는 코드
            if reg.search(i[0]) is None:
                invalid.append(i)

        unrecog_dict = []
        for i in invalid:
            
            vid = re.compile('\d+\D+')
            if vid.search(i[0]):
                cmd = vid.search(i[0])
                foward, back = cmd.span()
                numbers = i[0][foward:back]
                sentence = i[0][back:]
                unrecog_dict.append((numbers,sentence,i[1]))



        # book_valid_list

        new_dict = []
        for i in book_valid_list:
            vid = re.compile('\d*')
            cmd = vid.search(i[0])
            foward, back = cmd.span()
            numbers = i[0][foward:back]
            sentence = i[0][back:]
            new_dict.append((numbers,sentence,i[1]))


        book_sorted_list = sorted(new_dict, key=lambda x: (x[0], x[1]))


        book_diff =[] ##책 분류기호 [000,100,200....900]
        book_diff_list=[] ##기준서가가 아닌 곳에 잘못 분류된 책 위치
        book_right_list=[] ##기준 서가에 제대로 분류된 책 



        for i in book_sorted_list:
            book_diff.append((int(i[0][:3])//100*100)) ##분류기호 검출

        count = Counter(book_diff)

        if count:
            most_book_diff = count.most_common()[0][0] ##가장 많이 나온 책 분류기호를 기준으로 잡음

        for i in book_sorted_list:
            if (int(i[0][:3])//100*100) != most_book_diff:
                book_diff_list.append(i)
                # if int(i[0][:3])//100*100 == 0 :
                #     print(i[1],  "000번 서가 위치로 이동하세요")
                # else:
                #     print(i[1], str(int(i[0][:3])//100*100)+"번 서가 위치로 이동하세요")
            else: 
                book_right_list.append(i)




        ## 텐서 dtype을 int로 변경 (해도 되고 안해도 됨)    

        # book_label_boxes = book_label_boxes.type(torch.int16)
        # book_label_reversed_boxes = book_label_reversed_boxes.type(torch.int16)


        # try:
        #     for i in invalid : ##식별 안되는 책(검정색)
        #         original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), (int(book_label_boxes[i[1]][2]), int(book_label_boxes[i[1]][3])), (0,0,0), 3)  
        #         original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[1]][0]),int(book_label_boxes[i[1]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,0), 2)
        # except:
        #     pass
        original_image = cv2.imread(self.imagepath)

        if unrecog_dict:
            for i in unrecog_dict : ##식별 안되는 책(검정색)
                original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), (int(book_label_boxes[i[2]][2]), int(book_label_boxes[i[2]][3])), (0,0,0), 2)
                original_image = cv2.putText(original_image, "unrecog", (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,0), 2)


        for cnt, i in enumerate(book_sorted_list,1): ##같은 서가 위치 다른책 위치변경(파란색)

            original_image = cv2.rectangle(original_image, (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), (int(book_label_boxes[i[2]][2]), int(book_label_boxes[i[2]][3])), (255,0,0), 2)
            original_image = cv2.putText(original_image, "num"+str(cnt), (int(book_label_boxes[i[2]][0]),int(book_label_boxes[i[2]][1])), cv2.FONT_HERSHEY_COMPLEX,1 , (255,0,0), 2)
            print(cnt)

        try:
            for j in book_diff_list: ##다른 서가에 위치한 책 알려줌(초록색)
                original_image = cv2.rectangle(original_image, (int(book_label_boxes[j[2]][0]),int(book_label_boxes[j[2]][1])+20), (int(book_label_boxes[j[2]][2]), int(book_label_boxes[j[2]][3])-20), (0,255,0), 3)
                if int(i[0][:3])//100*100 ==0 :
                    original_image = cv2.putText(original_image, "move to 000th", (int(book_label_boxes[j[2]][0]+3),int(book_label_boxes[j[2]][1]+3)), cv2.FONT_HERSHEY_COMPLEX,1 , (0,255,0), 1)
                else:
                    original_image = cv2.putText(original_image, "move to "+str(int(j[0][:3])//100*100), (int(book_label_boxes[j[2]][0]+3),int(book_label_boxes[j[2]][1]+3)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1 , (0,255,0), 1)
        except:
            pass

        try:
            for i in book_label_reversed_boxes: ##뒤집힌 책 (빨간색)
                original_image = cv2.rectangle(original_image, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), (0,0,255), 3)
                original_image = cv2.putText(original_image, "reversed" , (int(i[0]),int(i[1])), cv2.FONT_HERSHEY_COMPLEX,1 , (0,0,255), 2)
        except:
            pass

        book_result = {}
        book_result['bookshelf'] = self.imagepath.split('/')[-1]
        book_result['unrecog_list'] =  [len(unrecog_dict)]
        book_result['reversed_list:'] = [len(book_label_reversed_boxes)]
        book_result['diff_list'] = [len(book_diff_list)]
        book_result_df = pd.DataFrame(book_result)
        # 최초 생성 이후 mode는 append
        if not os.path.exists("D:/octaforaugust/web/sophia/static/result/output.csv"):
            book_result_df.to_csv("D:/octaforaugust/web/sophia/static/result/output.csv", index=False, mode='w', encoding='utf-8-sig')
        else:
            book_result_df.to_csv("D:/octaforaugust/web/sophia/static/result/output.csv", index=False, mode='a', encoding='utf-8-sig', header=False)

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_image)
        original_image_path = 'd:/web/sophia/tmp/predicted/'
        precise = 'classified_book_{}.jpg'.format(dt.datetime.now().strftime(("%Y%m%d%H%M%S")))
        original_image_path = "".join([original_image_path, precise])
        original_image.save(original_image_path, "JPEG")
        
        
        return precise


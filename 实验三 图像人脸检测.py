# -*- coding: GBK -*-

import cv2
import dlib

path = "D:\\PycharmProjects\\deep_learning\\Image analysis\\homework\img\\xingye1.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#����������
detector = dlib.get_frontal_face_detector()
# ��ȡ���������
predictor = dlib.shape_predictor(
    "C:\\Users\\yizhuo\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\dlib-data\\shape_predictor_68_face_landmarks.dat"
)

dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)  # Ѱ��������68���궨��
    # �������е㣬��ӡ�������꣬��Ȧ����
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


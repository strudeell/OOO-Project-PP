import cv2
import numpy as np
from collections import defaultdict

# работа с изображением

original_img = cv2.imread("test(1).jpg")
img = original_img

new_img = np.zeros(img.shape, dtype='uint8')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 20, 20)

# создаём список прямоугольных границ

contours = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
boxes = []
for cnt in contours:
    rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
    box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.intp(box)  # округление координат
    if len(cv2.approxPolyDP(cnt, 10, True)) == 4:
        boxes.append(box)

# считаем частоту встречаемости значений по y (верхние и нижноие границы)

counts_y1 = defaultdict(int)
counts_y2 = defaultdict(int)

for box in boxes:
    for point in box[:2]:
        counts_y1[point[1]] += 1
    for point in box[2:]:
        counts_y2[point[1]] += 1

# удаление дубликатов

i = 0
while i < len(boxes) - 1:
    coords_curr = [boxes[i][x][y] for x in range(4) for y in range(2)]
    coords_next = [boxes[i + 1][x][y] for x in range(4) for y in range(2)]
    if set(coords_curr) == set(coords_next):
        boxes.pop(i)
    i += 1

# поиск внешних контуров

counts_y1 = {k: v for k, v in sorted(counts_y1.items(), key=lambda item: item[1], reverse=True)}
counts_y2 = {k: v for k, v in sorted(counts_y2.items(), key=lambda item: item[1], reverse=True)}
lst_y1 = list(counts_y1)
lst_y2 = list(counts_y2)
y1, y2 = min(lst_y1[:2]), max(lst_y2[:2])  # y1- верхняя граница, y2 - нижняя

# удаление областей, не являющихся границами
i = 0
while i < len(boxes):
    box = boxes[i]
    coords = [box[i][1] for i in range(4)]
    if not (y1 in set(coords) and y2 in set(coords)):
        boxes.pop(i)
    i += 1
# удаление внутренних границ
i = 0
alpha = 1  # погрешность в пикселях

all_y1 = [y1 + i for i in range(-alpha, alpha + 1)]
all_y2 = [y2 + i for i in range(-alpha, alpha + 1)]

while i < len(boxes):
    box = boxes[i]
    coords = [box[i][1] for i in range(4)]
    flag1 = flag2 = False
    for y in all_y1:
        if y in set(coords):
            flag1 = True
    for y in all_y2:
        if y in set(coords):
            flag2 = True
    if not (flag1 and flag2):
        boxes.pop(i)
    i += 1

# отрисовка найденных областей

for box in boxes:
    cv2.drawContours(original_img, [box], 0, (255, 0, 0), 1)
cv2.imshow('contours', original_img)  # вывод обработанного кадра в окно
cv2.waitKey()
cv2.destroyAllWindows()
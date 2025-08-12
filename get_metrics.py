import cv2
import numpy as np
import os

# 预测结果路径
pred_path = r'F:\SEG\SegResults\VM-Unet'
# 标签路径
lab_path = r'F:\SEG\SegResults\Label'


def tpcount(imgp, imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgp[i, j] == 255 and imgl[i, j] == 255:
                n = n + 1
    return n


def fncount(imgp, imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i, j] == 255 and imgp[i, j] == 0:
                n = n + 1
    return n


def fpcount(imgp, imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i, j] == 0 and imgp[i, j] == 255:
                n += 1
    return n


def tncount(imgp, imgl):
    n = 0
    for i in range(WIDTH):
        for j in range(HIGTH):
            if imgl[i, j] == 0 and imgp[i, j] == 0:
                n += 1
    return n



imgs = os.listdir(pred_path)
a = len(imgs)
TP = 0
FN = 0
FP = 0
TN = 0
c = 0
for name in imgs:
    imgp = cv2.imread(pred_path + '/' + name, 0)
    imgp = np.array(imgp)
    imgp = np.where(imgp > 128, 255, 0)

    imgl = cv2.imread(lab_path + '/' + name, 0)
    imgl = np.array(imgl)
    imgl = np.where(imgl > 128, 255, 0)

    WIDTH = imgl.shape[0]
    HIGTH = imgl.shape[1]

    TP += tpcount(imgp, imgl)
    FN += fncount(imgp, imgl)
    FP += fpcount(imgp, imgl)
    TN += tncount(imgp, imgl)

    c += 1
    print('已经计算：' + str(c) + ',剩余数目：' + str(a - c))

print('TP:' + str(TP))
print('FN:' + str(FN))
print('FP:' + str(FP))
print('TN:' + str(TN))

# 准确率
zq = (int(TN) + int(TP)) / (int(WIDTH) * int(HIGTH) * int(len(imgs)))
# 精确率
jq = int(TP) / (int(TP) + int(FP))
# 召回率
zh = int(TP) / (int(TP) + int(FN))
# F1
f1 = int(TP) * 2 / (int(TP) * 2 + int(FN) + int(FP))
#IoU
iou = int(TP) / (int(TP) + int(FP) + int(FN))

print('F1值：' + str(f1))
print('IoU值：' + str(iou))
print('Precision：' + str(jq))
print('Accuracy：' + str(zq))
print('Recall：' + str(zh))



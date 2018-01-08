# coding=utf-8
# 设置图片路径
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

picture_savePath = "/home/gengjia/gengjia/orl_faces/"

# 图片数据的n维数组
data = []
# 标签数组
label = []


# 将图片信息转化为可处理的向量
def ImageConvert():
    for i in range(1, 41):
        for j in range(1, 11):
            path = picture_savePath + "s" + str(i) + "/" + str(j) + ".pgm"
            # 单通道读取图片
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            img_col = img.reshape(h * w)
            data.append(img_col)
            label.append(i)


ImageConvert()
C_data = np.array(data)
C_label = np.array(label)
print(C_data.shape)
# 划分数据集与测试集
x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.2, random_state=256)
# 对训练集进行pca分析
pca = PCA(n_components=15, svd_solver='randomized').fit(x_train)

# 得到pca分析后的训练集和测试集
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# 进行svm分类，这里用的线性核函数
svc = SVC(kernel='linear')
svc.fit(x_train_pca, y_train)

# svm_predict = svc.predict(x_test_pca)
print('%.5f' % svc.score(x_test_pca, y_test))

# SVM+PCA----人脸识别
### Tools：
* opencv
* numpy
* sklearn
### 一、首先获取数据集
这里使用英国剑桥大学的AT&T人脸数据集：【http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html】
此数据集共有40×10=400张图片，图片大小为112*92，已经经过灰度处理。

s1-s40分别对应着每类样本：
![](https://github.com/gengjia007/SVM/blob/master/dir.png)

其中每类样本下有10张图片：
![](https://github.com/gengjia007/SVM/blob/master/dir_detail.png)

### 二、将图片转化为可处理的n维向量
由于每张图片的大小为112*92,所以这里每张图片共有10304个像素点，则每张图片转化为一个10304维向量。
这时需要一个图片转化函数ImageConvert():
```python
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
```
此时data中存储了每个图片的10304维信息,data格式（list）：[[...],[...],....,[...]]
此时label中存储了每个图片的类别标签1-40

生成特征向量矩阵：
```python
C_data = np.array(data)
C_label = np.array(label)
```
为了验证矩阵构建完成，我们输出矩阵规模：
```python
print(C_data.shape)
```
若输出（400，10304），则表示400*10304的特征矩阵构建完成。

### 三、分割数据集
```python
x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.2, random_state=256)
```
这里我指定了测试集占20%.

### 四、PCA主成分分析，降维处理
这里pca分析是通过计算协方差矩阵，找出影响特征的主要因素，em....举个简单例子来说明吧：
。
。
。
这里引入sklearn工具进行pca处理：
```python
pca = PCA(n_components=15, svd_solver='randomized').fit(x_train)
```
这里对PCA（）的主要参数进行一下介绍：
（1）n_components：这里指我们希望经过pca处理后保留的特征维度。
（2）svd_solver：指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}
（3）whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.

这里我指定了pca分析后特征向量保留了15个主要维度。









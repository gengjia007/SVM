# SVM+PCA----人脸识别
### Tools：
* opencv
* numpy
* sklearn
### 一、首先获取数据集
这里使用英国剑桥大学的AT&T人脸数据集：【http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html】
此数据集共有40×10=400张图片，图片大小为112*92，已经经过灰度处理。
![](https://github.com/gengjia007/SVM/blob/master/face.png)

s1-s40分别对应着每类样本：
![](https://github.com/gengjia007/SVM/blob/master/dir.png)

其中每类样本下有10张图片：
![](https://github.com/gengjia007/SVM/blob/master/dir_detail.png)

### 二、将图片转化为可处理的n维向量
由于每张图片的大小为112*92,所以这里每张图片共有10304个像素点，则每张图片转化为一个10304维向量。
这时需要一个图片转化函数ImageConvert():
```
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



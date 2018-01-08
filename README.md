# SVM+PCA----人脸识别
### Tools：
* opencv
* numpy
* sklearn
### 一、首先获取数据集
这里使用英国剑桥大学的AT&T人脸数据集：【http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html】
此数据集共有40×10=400张图片，图片大小为112*92，已经经过灰度处理。
![](https://github.com/gengjia007/SVM/blob/master/face.png)


### 二、将图片转化为可处理的n维向量
由于每张图片的大小为112*92,所以这里每张图片共有10304个像素点，则每张图片转化为一个10304维向量。


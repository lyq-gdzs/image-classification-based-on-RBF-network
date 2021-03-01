# image-classification-based-on-RBF-network
应用RBF网络对CIFAR-10进行分类。

运行环境：MATLAB R2016a。
运行步骤：
1.在matlab中导入所有文件
2.在rbfpiccla_simple中自行设定M值和N值的大小（需要大于2，值越大训练速度越快，准确率越低）
3.运行rbfpiccla_simple

注：
preprocess中为数据集的预处理部分
simprbf为对matlab自带的newrbe()进行简化
LMgist为gistdescriptor工具包中用于提取GIST特征的函数

关于CIFAR-10：
1.数据集已经对数据和标签做好分类，载入后分别放在data和labels中
2.共5个训练集data_batch,一个测试集test_batch，每个10000样本

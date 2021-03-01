%名称：RBF神经网络对cifar10数据集进行分类识别
clear;clc;
%读取数据
train=[];%训练集
trainlabel=[];%训练集label
test=[];%测试集
testlabel=[];%测试集label
for j=1:5,%读取训练集数据
     load(['data_batch_' num2str(j) '.mat']);
     train=[train;data];%训练集样本
     trainlabel=[trainlabel;labels];%训练集label
end
load 'test_batch.mat';%加载测试集数据
test=[test;data];%测试集样本
testlabel=[testlabel;labels];%测试集label
%================================================================================================================
%预处理
M=2;  %训练集数量=50000/M
[sptrain,Class]=preprocess(train,trainlabel,M);%sptrain为经过预处理后的训练集，Class为经过预处理后的训练集label
N=2; %测试集数量=10000/N
[sptest,testClass]=preprocess(test,testlabel,N);%sptest为经过预处理后的测试集，testClass为经过预处理后的测试集label
%================================================================================================================
%RBF网络的建立和训练
Class=Class';
testClass=testClass';
net = simprbf(sptrain,Class,1.4);
 %net = newrb(sptrain,Class,1)
 disp('训练完成，正在测试');
%RBF网络的测试
Rbfoutput = sim (net,sptest);
%计算识别率
[s1,s2] = size(Rbfoutput);
count = 0;
for i = 1:s2
    [m ,index] = max(Rbfoutput(:,i)); %m为当前列的最大值，index为该列最大值所处的行数，即最大值位置为（index,i）
    [l,std] = max(testClass(:,i));  %L为当前列的最大值，std为该列最大值所处的行数，即最大值位置为（std,i）
    if(index==std)
        count = count + 1;
    end
end
sprintf('识别率是%3.3f%%',100*count/s2)
for i = 1:uint8(s2/100)
    [m ,index] = max(Rbfoutput(:,i)); %m为当前列的最大值，index为该列最大值所处的行数，即最大值位置为（index,i）
    [l,std] = max(testClass(:,i));  %L为当前列的最大值，std为该列最大值所处的行数，即最大值位置为（index,i）
    if(index==std)
        plot(i,2,'b*');
        axis([0,uint8(s2/100),0,3]);
    else
        plot(i,1,'r*');
    end
    hold on;
end
%%roc
plotroc(testClass,Rbfoutput);%ROC
% plotroc(testClass(1,:),Rbfoutput(1,:),'1',testClass(2,:),Rbfoutput(2,:),'2',testClass(3,:),Rbfoutput(3,:),'3',testClass(4,:),Rbfoutput(4,:),'4',testClass(5,:),Rbfoutput(5,:),'5',testClass(6,:),Rbfoutput(6,:),'6',testClass(7,:),Rbfoutput(7,:),'7',testClass(8,:),Rbfoutput(8,:),'8',testClass(9,:),Rbfoutput(9,:),'9',testClass(10,:),Rbfoutput(10,:),'10');

for i=1:10 %ROC&AUC
[X,Y,T,AUC] = perfcurve(testClass(i,:),Rbfoutput(i,:),1);
subplot(3,4,i);
plot(X,Y);
AUC=roundn(AUC,-2);
title(['Class',num2str(i),' AUC:',num2str(AUC)]);
end

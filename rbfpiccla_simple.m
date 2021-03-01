%���ƣ�RBF�������cifar10���ݼ����з���ʶ��
clear;clc;
%��ȡ����
train=[];%ѵ����
trainlabel=[];%ѵ����label
test=[];%���Լ�
testlabel=[];%���Լ�label
for j=1:5,%��ȡѵ��������
     load(['data_batch_' num2str(j) '.mat']);
     train=[train;data];%ѵ��������
     trainlabel=[trainlabel;labels];%ѵ����label
end
load 'test_batch.mat';%���ز��Լ�����
test=[test;data];%���Լ�����
testlabel=[testlabel;labels];%���Լ�label
%================================================================================================================
%Ԥ����
M=2;  %ѵ��������=50000/M
[sptrain,Class]=preprocess(train,trainlabel,M);%sptrainΪ����Ԥ������ѵ������ClassΪ����Ԥ������ѵ����label
N=2; %���Լ�����=10000/N
[sptest,testClass]=preprocess(test,testlabel,N);%sptestΪ����Ԥ�����Ĳ��Լ���testClassΪ����Ԥ�����Ĳ��Լ�label
%================================================================================================================
%RBF����Ľ�����ѵ��
Class=Class';
testClass=testClass';
net = simprbf(sptrain,Class,1.4);
 %net = newrb(sptrain,Class,1)
 disp('ѵ����ɣ����ڲ���');
%RBF����Ĳ���
Rbfoutput = sim (net,sptest);
%����ʶ����
[s1,s2] = size(Rbfoutput);
count = 0;
for i = 1:s2
    [m ,index] = max(Rbfoutput(:,i)); %mΪ��ǰ�е����ֵ��indexΪ�������ֵ�����������������ֵλ��Ϊ��index,i��
    [l,std] = max(testClass(:,i));  %LΪ��ǰ�е����ֵ��stdΪ�������ֵ�����������������ֵλ��Ϊ��std,i��
    if(index==std)
        count = count + 1;
    end
end
sprintf('ʶ������%3.3f%%',100*count/s2)
for i = 1:uint8(s2/100)
    [m ,index] = max(Rbfoutput(:,i)); %mΪ��ǰ�е����ֵ��indexΪ�������ֵ�����������������ֵλ��Ϊ��index,i��
    [l,std] = max(testClass(:,i));  %LΪ��ǰ�е����ֵ��stdΪ�������ֵ�����������������ֵλ��Ϊ��index,i��
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

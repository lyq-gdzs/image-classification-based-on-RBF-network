function [dataset,labelclass]=preprocess(data,label,num)
for i=1:size(data,1)/num  
%data预处理
    R=reshape(data(i,1:1024),32,32);%R通道矩阵重整为32*32矩阵
    G=reshape(data(i,1025:2048),32,32);%G通道矩阵重整为32*32矩阵
    B=reshape(data(i,2049:end),32,32);%B通道矩阵重整为32*32矩阵
    RGBpic(:,:,1)=R;%将RGB三个矩阵合成一张RGB图片
    RGBpic(:,:,2)=G;
    RGBpic(:,:,3)=B;
    
    %LMGIST特征提取
    clear param;
    param.orientationsPerScale=[8 8 8 8];
    param.numberBlocks=4;
    param.fc_prefilt=4;
    [featureVectorgist,param]=LMgist(RGBpic,'',param);
    
    %HOG特征提取
    [featureVectorhog] = extractHOGFeatures(RGBpic);
    
    %LBP特征提取
    GRAYpic=rgb2gray(RGBpic);
    [featureVectorlbp] = extractLBPFeatures(GRAYpic);
    
    %将特征提取后获得的矩阵合为一个矩阵，作为训练网络的输入样本
     dataset(:,i)=double(real([featureVectorhog,featureVectorlbp,featureVectorgist]));%57% num=5:69.15%   num=4:71.12%   num=3.2:72.930%
     %dataset(:,i)=double(real([featureVectorhog,featureVectorgist]));%59.1%
     %dataset(:,i)=double(real([featureVectorlbp,featureVectorgist]));%61.3%
     %dataset(:,i)=double(featureVectorgist);%51.5%
     %dataset(:,i)=double(real(featureVectorlbp));%21.2%
     %dataset(:,i)=double(real(featureVectorhog));%53.1%
     %dataset(:,i)=double(real([featureVectorhog,featureVectorlbp]));%53.4%;num=4:63.88%
 %================================================================================================
 
 %将label预处理为one hot编码
   switch label(i);
       case 0 ;   
           labelclass(i,:)=[1 0 0 0 0 0 0 0 0 0]; 
       case 1;
           labelclass(i,:)=[0 1 0 0 0 0 0 0 0 0];
       case 2;
           labelclass(i,:)=[0 0 1 0 0 0 0 0 0 0];
       case 3;
           labelclass(i,:)=[0 0 0 1 0 0 0 0 0 0];
       case 4;
           labelclass(i,:)=[0 0 0 0 1 0 0 0 0 0];
       case 5;
           labelclass(i,:)=[0 0 0 0 0 1 0 0 0 0];
       case 6;
           labelclass(i,:)=[0 0 0 0 0 0 1 0 0 0];
       case 7;
           labelclass(i,:)=[0 0 0 0 0 0 0 1 0 0];
       case 8;
           labelclass(i,:)=[0 0 0 0 0 0 0 0 1 0];
       otherwise 9;
           labelclass(i,:)=[0 0 0 0 0 0 0 0 0 1];
   end
   %进度条
    if mod(i,(size(data,1)/num)/100)==0
        clc;
        disp(['预处理进度：',num2str(100*i/(size(data,1)/num)),'%']);
    end
end

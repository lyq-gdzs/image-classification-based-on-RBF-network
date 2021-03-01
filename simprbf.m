function net=simprbf(P,T,spread)
[R,Q]=size(P);%RΪP��������QΪP������
[S,Q]=size(T);
net=network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);%1input,   2layers,   layer1��layer2����ֵ�� input1��layer1��Ȩ�أ� layer1��layer2��Ȩ�أ� layer2�����
net.inputs{1}.size=R;%�����sizeΪR
net.layers{1}.size=Q;%������sizeΪQ
net.inputWeights{1,1}.weightFcn='dist';%input��layer1ӳ�䷽ʽΪ����ӳ��
net.layers{1}.netInputFcn='netprod';%layer1����ΪȨֵ����ֵ�Ĺ�ϵʽ
net.layers{1}.transferFcn='radbas';%inputͨ�����������ӳ�䵽layer1
net.layers{2}.size=S;%����cifar10�����sizeΪ10
net.outputs{2}.exampleOutput=T;%T��Ϊѵ����label
[w1,b1,w2,b2]=designrbe(P,T,spread);
net.b{1}=b1;%����layer1����ʱ������ֵ
net.iw{1,1}=w1;%input��layer1��Ȩ��/������������ģ�
net.b{2}=b2;%����layer2����ʱ������ֵ
net.lw{2,1}=w2;%layer1��layer2��Ȩ��
end

function [w1,b1,w2,b2]=designrbe(P,T,spread)
[r,q]=size(P);
[s2,q]=size(T);
w1=P';%w1Ϊ������������ģ�
b1=ones(q,1)*sqrt(-log(.5))/spread;%��������Ԫ��Ӧ��ֵ0.8326/spread��spreadΪ�����������չ�ٶ�
%b1=ones(q,1)*0.8326/spread;
a1=radbas(dist(w1,P).*(b1*ones(1,q)));%��������Ԫ�����
%a1=exp(-dist(w1,P)).*(b1*ones(1,q));
x=T/[a1;ones(1,q)];%TΪ�����x=[w2 b2],[w2 b2]��[A;I]=T
w2=x(:,1:q);
b2=x(:,q+1);
end
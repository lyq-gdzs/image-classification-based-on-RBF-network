function net=simprbf(P,T,spread)
[R,Q]=size(P);%R为P的行数，Q为P的列数
[S,Q]=size(T);
net=network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);%1input,   2layers,   layer1和layer2有阈值， input1到layer1有权重， layer1到layer2有权重， layer2有输出
net.inputs{1}.size=R;%输入层size为R
net.layers{1}.size=Q;%隐含层size为Q
net.inputWeights{1,1}.weightFcn='dist';%input到layer1映射方式为距离映射
net.layers{1}.netInputFcn='netprod';%layer1数据为权值和阈值的关系式
net.layers{1}.transferFcn='radbas';%input通过径向基函数映射到layer1
net.layers{2}.size=S;%对于cifar10输出层size为10
net.outputs{2}.exampleOutput=T;%T作为训练集label
[w1,b1,w2,b2]=designrbe(P,T,spread);
net.b{1}=b1;%计算layer1数据时所用阈值
net.iw{1,1}=w1;%input到layer1的权重/径向基函数中心？
net.b{2}=b2;%计算layer2数据时所用阈值
net.lw{2,1}=w2;%layer1到layer2的权重
end

function [w1,b1,w2,b2]=designrbe(P,T,spread)
[r,q]=size(P);
[s2,q]=size(T);
w1=P';%w1为径向基函数中心？
b1=ones(q,1)*sqrt(-log(.5))/spread;%隐含层神经元对应阈值0.8326/spread，spread为径向基函数扩展速度
%b1=ones(q,1)*0.8326/spread;
a1=radbas(dist(w1,P).*(b1*ones(1,q)));%隐含层神经元的输出
%a1=exp(-dist(w1,P)).*(b1*ones(1,q));
x=T/[a1;ones(1,q)];%T为输出，x=[w2 b2],[w2 b2]·[A;I]=T
w2=x(:,1:q);
b2=x(:,q+1);
end
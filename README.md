# CW_Attack_on_MNIST
Implemention of cw attack on pytorch with corresponding MNIST model

## MNIST model
根据[Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) p6的TABLE1搭建  
由四个卷积层、两个池化层、两个全连接层、激活函数ReLU组成。论文中提到的softmax不出现在model结构中，而是在loss中体现，因此此处选用CrossEntropyLoss函数

## Carlini and Wagner L2

论文中给出了三种攻击方式 L0, L2, L-inf,此处采用了L2攻击方式，可以修改函数

```
def loss1_func(w,x,d,c):
    return torch.dist(x,(torch.tanh(w)*d+c),p=2)

```
以及对应的7种f(x')函数，在```def f(output,tlab,target,k=0)```中修改


# CW_Attack_on_MNIST
Implemention of cw attack on pytorch with corresponding MNIST model

## MNIST model
Based on [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) TABLE1
Consist of four convolution layer, two pooling layers, tow FC layers and ReLU.
Notice: softmax shouldn't be put into model. What we need is the result from FC.That's why I use crossEntropyLoss as loss function.

## Carlini and Wagner L2
L0,L2,L-inf be used for generating adversarial examples. In this repo, I choose L2 norm as the objective function.
```
def loss1_func(w,x,d,c):
    return torch.dist(x,(torch.tanh(w)*d+c),p=2)

```
You can define different objective function f in ```def f(output,tlab,target,k=0)```

## Blog(Written in Chinese)
[CW Attack](https://blog.andrmapper.cn/2020/08/17/CWAttack论文阅读与复现/)

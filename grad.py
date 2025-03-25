import torch
import math



def update_w(loss1, loss2,loss3 ,**kwargs):
    """
    计算每一次loss的视角权重
    """
    w1=(math.e**(-loss1/1))/((math.e**(-loss1/1))+(math.e**(-loss2/1))+(math.e**(-loss3/1)))
    print(w1)
    w2=(math.e**(-loss2/1))/((math.e**(-loss1/1))+(math.e**(-loss2/1))+(math.e**(-loss3/1)))
    print(w2)
    w3=(math.e**(-loss3/1))/((math.e**(-loss1/1))+(math.e**(-loss2/1))+(math.e**(-loss3/1)))
    print(w3)

    l=w1*loss1+w2*loss2+w3*loss3
    print(l)
    return l











if __name__ == '__main__':
    update=update_w(1.5843,1.7079,1.4717)

    #print(update())
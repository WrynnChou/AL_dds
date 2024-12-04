import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
def kernel(u: np.ndarray, v: np.ndarray, criterion: str = "CD"):
    """
    Kernel function
    """

    res = 0
    d1 = u.shape[1]
    d2 = v.shape[1]
    assert d1 == d2, "u, v have different dimensions"
    if criterion == "CD":
        u -= 0.5
        v -= 0.5
        for i in range(d1):
            res *= 1 + 1 / 2 * np.abs(u[i]) + 1 / 2 * np.abs(v[i]) - 1 / 2 * np.abs(u[i] - v[i])

    elif criterion == "WD":
        for i in range(d1):
            res *= 3 / 2 - np.abs(u[i] - v[i]) + (u[i] - v[i]) ** 2

    else:
        assert criterion == "MD", "Only support CD, WD and MD."
        for i in range(d1):
            res *= 15 / 8 - 1 / 4 * np.abs(u[i] - 1 / 2) - 1 / 4 * np.abs(v[i] - 1 / 2) - 3 / 4 * np.abs(u[i] - v[i]) + \
                   1 / 2 * (u[i] - v[i]) ** 2
    return res



def gefd(data: np.ndarray, subdata: np.ndarray):
    """
    General empirical F-discrepancy
    :param data: Whole data
    :param subdata: sub data
    :return: FEFD of subdata.
    """

    dim1 = data.shape[1]
    dim2 = subdata.shape[1]
    assert dim1 == dim2, "Dimensions don't match please check the data."
    N = data.shape[0]
    n = subdata.shape[0]
    T = Tx(data)

    d1 = 0
    for i in range(N):
        for j in range(N):
            Tu = T(data[i, :])
            Tv = T(data[j, :])
            d1 += kernel(Tu, Tv)

    d2 = 0
    for i in range(N):
        for j in range(n):
            Tu = T(data[i, :])
            Tv = T(subdata[j, :])
            d2 += kernel(Tu, Tv)

    d3 = 0
    for i in range(n):
        for j in range(n):
            Tu = T(subdata[i, :])
            Tv = T(subdata[j, :])
            d3 += kernel(Tu, Tv)

    res = 1 / (N ** 2) * d1 - 2 / (N * n) * d2 + 1 / (n ** 2) * d3
    return res



def fast_gefd(data, subdata):
    """
    Fast General empirical F-discrepancy compare method.
    For the same data, we want to compare the GEFD of two subdata, they share the same d1.
    And d1 is the most time-cost part. Thus, we could only calculate d2 and d3 to save time.
    :param data: The whole data
    :param subdata: subdata
    :return: The sum of second part and third part of GEFD between data and subdata.
    """

    dim1 = data.shape[1]
    dim2 = subdata.shape[1]
    assert dim1 == dim2, "Dimensions don't please check the data."
    N = data.shape[0]
    n = subdata.shape[0]
    T = Tx(data)
    Tdata = T(data)
    Tsubdata = T(subdata)

    d2 = 0
    for i in range(N):
        for j in range(n):
            Tu = T(data[i, :])
            Tv = T(subdata[j, :])
            d2 += kernel(Tu, Tv)

    d3 = 0
    for i in range(n):
        for j in range(n):
            Tu = T(subdata[i, :])
            Tv = T(subdata[j, :])
            d3 += kernel(Tu, Tv)

    print(d2)
    print(d3)
    res = - 2 / (N * n) * d2 + 1 / (n ** 2) * d3
    return res


def Tx(x: np.ndarray):
    """
    Generate transformation T_{\mathcal{x}}
    :param x: whole samples
    :return: transformation function Tx
    """
    n = x.shape[0]
    if len(x.shape) == 1:
        dim = 1
        x = x.reshape([n, 1])
    else:
        dim = x.shape[1]
    f_ = []
    for i in range(dim):
        f = ecdf_1dimension(x[:, i])
        f_.append(f)

    def T(u):
        if len(u.shape) == 1:
            u = u.reshape([u.shape[0], 1])
        res = []
        for i in range(dim):
            res_ = f_[i](u[:,i])
            res.append(res_)
        res = np.array(res)
        return res.transpose()
    return T

def ecdf_1dimension(x: np.ndarray):
    n = x.shape[0]
    assert len(x.shape) == 1, "This function only support one dimension"

    def ecdf(u: np.ndarray):
        u = np.array(u)
        if len(u.shape) == 0:
            u = u.reshape(1)
        res = np.zeros_like(u).astype("float32")
        for i in range(u.shape[0]):
            s = np.sum(x <= u[i])
            res[i] = s / n
        return res

    return ecdf

def cat_of_dataset(subdata, acc):
    cat = []
    for x, y in subdata:
        cat.append(y)
    cat.sort()
    xx = np.arange(5, 200, 10)
    plt.rcParams['font.size'] = 16  # 设置全局字体大小为16

    # plot histogram chart for var1
    plt.hist(cat, bins=20, label="Number of samples")
    plt.plot(xx, acc, label="Acc")
    plt.legend()
    plt.title('Tiny-Imagenet with DDS ' + str(len(cat))+ ' samples.')
    plt.show()
    c = Counter(cat)
    a = []
    for i in range(200):
        if c[i] == 0:
            tmp = -10
        else:
            tmp = c[i]
        a.append(tmp)
    aa = np.array(a).reshape((10, 20))
    plt.title('Heatmap of Tiny-Imagenet with DDS ' + str(len(cat))+ ' samples.')
    plt.imshow(aa)
    plt.colorbar()



print('Have a nice day!')











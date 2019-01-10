"""
内容：
1.机器学习-回归分析
2.标准回归函数
3.局部加权线性回归
4.岭回归
5.前向逐步回归
姓名：pcb
日期：2019.1.6
"""
from numpy import *
import matplotlib.pyplot as plt


#加载数据
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    datMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        datMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return datMat,labelMat

#1.---------------计算最佳拟合直线---------------------------------------------
def standRegres(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr)
    xTx=xMat.T*xMat               #计算X.T*X
    if linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat.T)       #xTx.I为xTx的逆
    return ws

#画出线性回归的散点图以及最佳拟合曲线
def plotScatter(xArr,yArr,ws):
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat=xMat*ws
    coef=corrcoef(yHat.T,yMat)      #计算相关系数
    print("相关系数为：",coef)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],c='r',s=10)  #flatten函数将矩阵降成一维，.A表示将矩阵转换成数组
    xCopy=xMat.copy()
    xCopy.sort(0)                 #按列排序结果
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.savefig('最佳拟合直线示意图.png')
    plt.show()

#-----------------------------------------------------------------------------

#2.----------------局部加权线性回归---------------------------------------------
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr);yMat=mat(yArr)
    m=shape(xMat)[0]
    weights=mat(eye((m)))                             #创建对角矩阵(方阵)，阶数等于样本点的个数，即为每个样本点初始化一个权重
    for j in range(m):                                #算法遍历数据集，计算每一个样本点对应的权重值
        diffMat=testPoint-xMat[j,:]                   #计算样本点与待预测样本点的距离
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2)) #随着距离的增加，权重值的大小以指数级衰减
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return
    ws=xTx.I*(xMat.T*(weights*yMat.T))
    return testPoint*ws                               #得到回归系数ws的一个估计

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)          #给定空间中的任意一点，计算出对应的预测值yHat
    return yHat

def plotScatterlwlr(xArr,yArr,yHat):
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], c='r', s=2)  # flatten函数将矩阵降成一维
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]                                               #返回的是数组值从小到大的索引值
    ax.plot(xSort[:,1], yHat[srtInd])
    plt.savefig('k=0.01时局部加权线性回归示意图.png')
    plt.show()

#-----------------------------------------------------------------------------

#3.----------使用岭回归---------------------------------------------------------


#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print('This matrix is singular,connot do inverse')
        return
    ws=denom.I*(xMat.T*yMat.T)
    return  ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr)
    yMean=mean(yMat.T,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def plotRidge(ridgeWeights):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.savefig('岭回归的回归系数变化图.png')
    plt.show()

#-----------------------------------------------------------------------------

#-----------前向逐步回归-------------------------------------------------------
"""
伪代码：
    数据标准化，使其分布满足0均值和单位方差
    在每轮迭代过程中：
        设置当前最小误差lowestError为正无穷
        对每个特征：
            增大或者缩小：
               改变一个系数得到一个新的w
               计算新的w下的误差
               如果误差Error小于最小当前误差lowestError:
                   设置wbest等于当前的w
            将w设置为新的wbest

"""
def stageWise(xArr,yArr,eps,numIt):

    #数据标准化
    xMat=mat(xArr);yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()

    #每轮迭代过程
    for i in range(numIt):
        print(ws.T)
        lowesError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowesError:
                    lowesError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#分析预测误差大小
def rssError(yArr,yHatArr):
    return((yArr-yHatArr)**2).sum()

def plotlasso(returnWeights):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(returnWeights)
    plt.savefig('前向逐步回归系数变化图.png')
    plt.show()
#----------------------------------------------------------------------------


def main():
# #1.---------线性回归---------------------------
#     xArr,yArr=loadDataSet('ex0.txt')
#     ws=standRegres(xArr,yArr)
#     plotScatter(xArr,yArr,ws)
# #---------------------------------------------
# #2.--------局部加权线性回归---------------------
#     xArr,yArr=loadDataSet('ex0.txt')
#     #ws=lwlr(xArr[0],xArr,yArr,0.001)
#     #print(ws)
#     yHat=lwlrTest(xArr,xArr,yArr,0.01)
#     plotScatterlwlr(xArr,yArr,yHat)
# #---------------------------------------------

# #3.---------岭回归------------------------------
#     abX,abY=loadDataSet('abalone.txt')
#     ridgeWeights=ridgeTest(abX,abY)
#     plotRidge(ridgeWeights)
#
# #----------------------------------------------

#4.---------前向逐步回归--------------------------
    xArr,yArr=loadDataSet('abalone.txt')
    returnMat=stageWise(xArr,yArr,0.005,1000)
    plotlasso(returnMat)
    print(returnMat)
#-----------------------------------------------


if __name__=='__main__':
    main()

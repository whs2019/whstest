import os
import numpy as np
import operator


def createdataset():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels

def classify0(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat = np.tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount ={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    group, labels = createdataset()
    test = [60,55]
    test_class = classify0(test, group, labels, 3)
    print(test_class)

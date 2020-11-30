# 实证角度总结的很好的一篇文章
https://blog.csdn.net/qq_22210253/article/details/85229988

CrossEntropy = Log(softmax) + NLLLoss  

NLLLoss: # The negative log likelihood loss.
# 简单的理解，NLLLoss就是对log(softmax)的结果，按照真实标签target作为下标，按行分别提取出相应矩阵相应元素，提取出来并全部取负数再相加求和，除以取出的个数(也即标签数)，取平均数的结果。


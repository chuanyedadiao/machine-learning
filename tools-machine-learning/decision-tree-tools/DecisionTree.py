import pandas as pd  # 加载数据，以及针对One-Hot编码

from sklearn.preprocessing import LabelEncoder

'''将字符串转换为编码'''

import numpy as np  # 计算平均值和标准差等等
import matplotlib.pyplot as plt  # 画画

from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型
from sklearn.tree import plot_tree  ##用于画决策树
from sklearn.model_selection import train_test_split  # 训练集，测试集
from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

'''
读取数据
params:{
    filename:文件名/文件路径
}
'''


def readData(filename):
    df = pd.read_csv(filename, sep=';')
    return df


'''
读取数据
'''

'''
缺失值先不管
'''

'''
one hot encoder
编码
'''


def labelEncodeTemp(df, column):
    encoder = LabelEncoder()
    encoder.fit(df[column].values.tolist())
    temp = df[column].values.tolist()
    return encoder.transform(temp)


def labelEncoder(df):
    for column in list(df):
        if (not (np.issubdtype(df[column].dtype, np.number))):
            data = labelEncodeTemp(df, column)
            df.drop([column], axis=1)
            df[column] = data
    return df


'''
初步建立一个树的模型
参数对应于DecisionTreeClassifier
'''


def buildTree(X_train, y_train, criterion="gini", splitter="best",
                   max_depth=None, min_samples_split=2,
                   min_samples_leaf=1, min_weight_fraction_leaf=0,
                   max_features=None, random_state=None,
                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                   class_weight="balanced", ccp_alpha=0.0):
    clf_dt = DecisionTreeClassifier(criterion=criterion,
                                    splitter=splitter,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                    max_features=max_features,
                                    random_state=random_state,
                                    max_leaf_nodes=max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease,
                                    class_weight=class_weight,
                                    ccp_alpha=ccp_alpha
                                    )
    clf_dt = clf_dt.fit(X_train, y_train)
    '''
    返回已经建立好的模型
    '''
    return clf_dt


'''
查看初步建立好的树的模型的图
class_namse:分类结果
feature_name:特征的名字
'''


def saveTreeImage(clf_dt, class_namse=None, feature_name=None):
    plt.figure(figsize=(30, 20))

    plot_tree(clf_dt,
              filled=True,
              rounded=True,
              class_names=class_namse,
              feature_names=feature_name)

    plt.savefig("1.png")


'''
查看混淆矩阵
display_labels:展示出来的结果
'''


def confusionMatrix(clf_dt, X_test, y_test, display_labels=None,name="confusion"):
    plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=display_labels)
    resultName = name+".png"
    plt.savefig(resultName)


'''
防止过拟合，来修剪树
最小成本复杂度修剪第一部分——修剪参数 alpha
空值修剪的多或者少 
找到alpha的最佳方法就是绘制每个不同值得函数的准确性  acc
对训练集和测试集都会使用
'''


##得到不同ccp_alphas下的模型
def getClfdts(clf_dt, X_train, y_train):
    path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

    ccp_alphas = path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]

    clf_dts = []

    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf_dt.fit(X_train, y_train)
        clf_dts.append(clf_dt)

    return clf_dts


def getCcpAlphas(clf_dt, X_train, y_train):
    path = clf_dt.cost_complexity_pruning_path(X_train, y_train)

    ccp_alphas = path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]
    return ccp_alphas


## 不同ccp_alphas下的acc图像
def pruningAccImage(clf_dt, X_train, y_train, X_test, y_test):
    clf_dts = getClfdts(clf_dt, X_train, y_train)
    ccp_alphas = getCcpAlphas(clf_dt, X_train, y_train)
    train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
    test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

    fig, ax = plt.subplots()

    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig("3.png")


'''
返回测试集 准确率最高的test_scores

存疑
'''


def bestCcpAlpha(clf_dt, X_train, y_train, X_test, y_test):
    clf_dts = getClfdts(clf_dt, X_train, y_train)
    ccp_alphas = getCcpAlphas(clf_dt, X_train, y_train)
    train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
    test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]
    temp = test_scores.index(max(test_scores))
    return ccp_alphas[temp]


'''
验证选择的ccp
交叉验证
'''


def CrossValidationImage(ccp_alpha, X_train, y_train, cv):
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    ## 交叉验证
    ## cv代表是几折交叉验证，就是分成了几个不同的训练集和测试集
    scores = cross_val_score(clf_dt, X_train, y_train, cv=cv)
    df = pd.DataFrame(data={'tree': range(cv), 'accuracy': scores})
    df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

    plt.savefig("4.png")


'''
上述函数可以观察到交叉验证下 相同的ccp_alpha也是会有不同的acc

所以通过交叉验证来找到最优的ccp_alpha
'''


def bestCcpCrossValidation(clf_dt, X_train, y_train,cv):
    ccp_alphas = getCcpAlphas(clf_dt=clf_dt, X_train=X_train, y_train=y_train)

    alpha_loop_values = []

    for ccp_alpha in ccp_alphas:
        clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(clf_dt, X_train, y_train, cv=cv)
        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

    alpha_results = pd.DataFrame(alpha_loop_values,
                                 columns=['alpha', 'mean_accuracy', 'std'])

    alpha_results.plot(x='alpha',
                       y='mean_accuracy',
                       yerr='std',
                       marker='o',
                       linestyle='--')

    plt.savefig("5.png")

    mean_score = alpha_results['mean_accuracy'].tolist()
    temp = mean_score.index(max(mean_score))
    return (alpha_results['alpha'].tolist())[temp]

def last_Tree(clf_dt,name,
              filled = True,
              rounded = True,
              class_name = None,
              feature_names=None):
    plt.figure(figsize=(30, 15))
    plot_tree(clf_dt,
              filled=filled,
              rounded=rounded,
              class_names=class_name,
              feature_names=feature_names)

    plt.savefig(name)

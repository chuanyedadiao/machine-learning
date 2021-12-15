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


class DecisionTree:
    """
    clf_dts : 用于后续在剪枝中存储所生成的所有树模型
    df :读取的数据
    X : df去除目标列
    y : df中目标列
    X_train,y_train : 训练集的输入输出
    X_test,y_test : 测试集的输入输出
    clf_dt : 树模型
    Count一系列用于保存图片的标记
    ccp_alphas : 剪枝的重要参数
    bestCcp : 最优ccp_alphas
    """

    def __init__(self):
        self.clf_dts = []
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf_dt = None
        self.figCount = 0
        self.ConfigCount = 0
        self.ccpAlpha_ACC_Count = 0
        self.crossValidation_Count = 0

        self.ccp_alphas = None
        self.path = None
        self.bestCcp = 0

    """
    读取数据
    params:{
        filename:文件名/文件路径
    }
    """

    def readData(self, filename):
        self.df = pd.read_csv(filename, sep=';')
        return self.df

    '''
    one hot encoder
    编码
    '''

    def labelEncoder(self):

        for column in list(self.df):
            if not (np.issubdtype(self.df[column].dtype, np.number)):
                encoder = LabelEncoder()
                encoder.fit(self.df[column].values.tolist())
                temp = self.df[column].values.tolist()
                data = encoder.transform(temp)
                self.df.drop([column], axis=1)
                self.df[column] = data
        return self.df

    """
    分解X和y
    """

    def splitXy(self):
        self.X = self.df.drop("y", axis=1).copy()
        self.y = self.df["y"].copy()

    """
    分解训练集和测试集
    """

    def splitTrainTest(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                random_state=42)

    """
    建立树模型
    """

    def buildTree(self, criterion="gini", splitter="best",
                  max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0,
                  max_features=None, random_state=None,
                  max_leaf_nodes=None, min_impurity_decrease=0.0,
                  class_weight="balanced", ccp_alpha=0.0):
        self.clf_dt = DecisionTreeClassifier(criterion=criterion,
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
        self.clf_dt = self.clf_dt.fit(self.X_train, self.y_train)
        return self.clf_dt

    """
    保存树的图片
    """

    def saveTreeImage(self, class_names=None, feature_name=None, fig_name=None):
        plt.figure(figsize=(30, 20))

        plot_tree(self.clf_dt,
                  filled=True,
                  rounded=True,
                  class_names=class_names,
                  feature_names=feature_name)
        if fig_name is None:
            fig_name = "Tree" + str(self.figCount) + ".png"
            self.figCount += 1
        plt.savefig(fig_name)

    """
    得到混淆矩阵的图片
    """

    def confusionMatrix(self, display_labels=None, fig_name=None):
        plot_confusion_matrix(self.clf_dt, self.X_test, self.y_test,
                              display_labels=display_labels)
        if fig_name is None:
            fig_name = "Confusion" + str(self.ConfigCount) + ".png"
            self.ConfigCount += 1
        plt.savefig(fig_name)

    """
    得到不同ccp_alphas下的模型
    """

    def getClfdts(self):
        self.path = self.clf_dt.cost_complexity_pruning_path(self.X_train, self.y_train)

        self.ccp_alphas = self.path.ccp_alphas
        self.ccp_alphas = self.ccp_alphas[:-1]

        for ccp_alpha in self.ccp_alphas:
            temp = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            temp.fit(self.X_train, self.y_train)
            self.clf_dts.append(temp)
        return self.clf_dts

    """
    不同ccp_alphas下的acc图像
    """

    def pruningAccImage(self, fig_name=None):
        train_scores = [temp.score(self.X_train, self.y_train) for temp in self.clf_dts]
        test_scores = [temp.score(self.X_test, self.y_test) for temp in self.clf_dts]

        fig, ax = plt.subplots()

        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(self.ccp_alphas, train_scores, label="train", drawstyle="steps-post")
        ax.plot(self.ccp_alphas, test_scores, label="test", drawstyle="steps-post")
        ax.legend()
        if fig_name is None:
            fig_name = "ccpAlpha_ACC" + str(self.ccpAlpha_ACC_Count) + ".png"
            self.ccpAlpha_ACC_Count += 1

        plt.savefig(fig_name)

    '''
    返回测试集 准确率最高的test_scores

    存疑
    '''

    def bestCcpAlpha(self):
        # train_scores = [clf_dt.score(self.X_train, self.y_train) for clf_dt in self.clf_dts]
        test_scores = [temp.score(self.X_test, self.y_test) for temp in self.clf_dts]
        temp = test_scores.index(max(test_scores))
        self.bestCcp = self.ccp_alphas[temp]
        return self.bestCcp

    '''
    验证选择的ccp
    交叉验证
    '''

    def CrossValidationImage(self, cv=5, fig_name=None):
        temp_clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=self.bestCcp)
        ## 交叉验证
        ## cv代表是几折交叉验证，就是分成了几个不同的训练集和测试集
        scores = cross_val_score(temp_clf_dt, self.X_train, self.y_train, cv=cv)
        df = pd.DataFrame(data={'tree': range(cv), 'accuracy': scores})
        df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

        if fig_name is None:
            fig_name = "crossValidation" + str(self.crossValidation_Count) + ".png"
            self.crossValidation_Count += 1
        plt.savefig(fig_name)

    '''
    上述函数可以观察到交叉验证下 相同的ccp_alpha也是会有不同的acc

    所以通过交叉验证来找到最优的ccp_alpha
    '''

    def bestCcpCrossValidation(self, cv=5):
        alpha_loop_values = []

        for temp in self.ccp_alphas:
            temp_clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=temp)
            scores = cross_val_score(temp_clf_dt, self.X_train, self.y_train, cv=cv)
            alpha_loop_values.append([temp, np.mean(scores), np.std(scores)])

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
        self.bestCcp = (alpha_results['alpha'].tolist())[temp]

        return self.bestCcp


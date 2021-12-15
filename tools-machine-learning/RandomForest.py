import pandas as pd        #加载数据，以及针对One-Hot编码
import numpy as np         #计算平均值和标准差等等
import matplotlib.pyplot as plt #画画
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class RandomForest:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.rd_clf = None
        self.y_predict = 0

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

    def splitXy(self, y=None):
        self.X = self.df.drop(y, axis=1).copy()
        self.y = self.df[y].copy()

    '''
    移除低方差的特征

    param:threshold
    训练集方差低于此阈值的特征将被删除。
    默认是保留所有具有非零方差的特征，即移除所有样本中具有相同值的特征。
    '''

    def featureSelection(self, threshold=0.0):
        sel = VarianceThreshold(threshold=threshold)
        X_temp = sel.fit_transform(self.X)
        self.X = pd.DataFrame(X_temp)

    """
        分解训练集和测试集
        """

    def splitTrainTest(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                random_state=42)


    def buildRandomForest(self,n_estimators=100,
                          criterion='gini',
                          max_depth=None,
                          min_samples_split=2,
                          min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0,
                          max_features="auto",
                          max_leaf_nodes=None,
                          min_impurity_decrease=0.0,
                          bootstrap=True,
                          oob_score=False,
                          n_jobs=None,
                          random_state=None,
                          verbose=0,
                          warm_start=False,
                          class_weight=None,
                          ccp_alpha=0.0,
                          max_samples=None):
        self.rd_clf = RandomForestClassifier( n_estimators=n_estimators,
                                              criterion=criterion,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              min_weight_fraction_leaf=min_weight_fraction_leaf,
                                              max_features=max_features,
                                              max_leaf_nodes=max_leaf_nodes,
                                              min_impurity_decrease=min_impurity_decrease,
                                              bootstrap=bootstrap,
                                              oob_score=oob_score,
                                              n_jobs=n_jobs,
                                              random_state=random_state,
                                              verbose=verbose,
                                              warm_start=warm_start,
                                              class_weight=class_weight,
                                              ccp_alpha=ccp_alpha,
                                              max_samples=max_samples)
        self.rd_clf.fit(self.X_train,self.y_train)

    '''
    特征重要度
    '''
    def importanceImage(self,figname=None):
        importances_values = self.rd_clf.feature_importances_
        importances = pd.DataFrame(importances_values, columns=["importance"])
        feature_data = pd.DataFrame(self.X_train.columns, columns=["feature"])
        importance = pd.concat([feature_data, importances], axis=1)
        importance = importance.sort_values(["importance"], ascending=True)
        importance["importance"] = (importance["importance"] * 1000).astype(int)
        importance = importance.sort_values(["importance"])
        importance.set_index('feature', inplace=True)
        importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
        plt.savefig(figname)

    def confusionMatrix(self, display_labels=None, fig_name=None):
        plot_confusion_matrix(self.rd_clf, self.X_test, self.y_test,
                              display_labels=display_labels)
        # if fig_name is None:
        #     fig_name = "Confusion" + str(self.ConfigCount) + ".png"
        #     self.ConfigCount += 1
        plt.savefig(fig_name)

    def roc(self,fig_name=None):
        self.y_predict = self.rd_clf.predict(self.X_test)

        fpr, tpr, thersholds = roc_curve(self.y_test, self.y_predict, pos_label=self.rd_clf.classes_[1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.savefig(fig_name)

    '''
        准确率 accuracy
        '''

    def accuracy(self):
        return accuracy_score(self.y_test, self.y_predict)

    '''
    精确率
    '''

    def precision(self):
        return precision_score(self.y_test, self.y_predict)

    '''
    召回率
    '''

    def recall(self):
        return recall_score(self.y_test, self.y_predict)

    '''
    F1值
    '''

    def f1_score(self):
        return f1_score(self.y_test, self.y_predict)

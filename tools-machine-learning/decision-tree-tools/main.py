from sklearn.model_selection import train_test_split

import DecisionTree
import pprint

print("-----------------读取数据Start------------------------")
df = DecisionTree.readData("../bank-additional.csv")
pprint.pprint(df)
print("-----------------读取数据Done------------------------\n\n")

print("-----------------编码Start---------------------------")
df = DecisionTree.labelEncoder(df)
pprint.pprint(df)
print("-----------------编码Done---------------------------\n\n")

print("-----------------建立初步树Start----------------------")
X = df.drop("y", axis=1).copy()
y = df["y"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTree.buildTree(X_train, y_train, random_state=42)
print("-----------------建立初步树Done----------------------\n\n")

print("-----------------画图初步树Start----------------------")
class_namse = ["No Loan", "Yes Loan"]
feature_name = X.columns
DecisionTree.saveTreeImage(clf_dt, class_namse=class_namse, feature_name=feature_name)
print("-----------------画图初步树Done----------------------\n\n")

print("-----------------混淆矩阵Start----------------------")
DecisionTree.confusionMatrix(X_test=X_test, y_test=y_test,
                             clf_dt=clf_dt, display_labels=["No loan", "Got loan"])
print("-----------------混淆矩阵Done----------------------\n\n")

print("-----------------修剪Start----------------------")
## 修剪
## 不同ccp_alphas下的acc图像
DecisionTree.pruningAccImage(clf_dt, X_train, y_train, X_test, y_test)

ccp_alpha = DecisionTree.bestCcpAlpha(clf_dt, X_train, y_train, X_test, y_test)
print("first",ccp_alpha)
print("-----------------修剪Done----------------------")

print("-----------------交叉验证Start----------------------")
# 如果使用不同的训练和测试集的话，
# 同样的alpha其实会得到不同的准确率，接下来就要用交叉验证来找到ccp_alpha的最优值
cv = 5
DecisionTree.CrossValidationImage(ccp_alpha=ccp_alpha,
                                  X_train=X_train,
                                  y_train=y_train,
                                  cv=5)
ccp_alpha = DecisionTree.bestCcpCrossValidation(clf_dt, X_train, y_train, 5)
print("second",ccp_alpha)
print("-----------------交叉验证Done----------------------")


print("-----------------最终决策树Start----------------------")
clf_dt_pruned = DecisionTree.buildTree(X_train=X_train,
                                       y_train=y_train,
                                       random_state=42,
                                       ccp_alpha=ccp_alpha)

DecisionTree.confusionMatrix(clf_dt_pruned, X_test, y_test,
                             display_labels=["Does noe hava LOAN","Has LOAN"],
                             name="pruned_tree_confusion_matrix")
DecisionTree.last_Tree(clf_dt_pruned, "last_Tree.png",
                       class_name=["Does noe hava LOAN","Has LOAN"],
                       feature_names=X.columns
                       )
print("-----------------最终决策树Done----------------------")

# def func1(a=2, b=1, c=44, d=55):
#     print(a, b, c, d)
#
#
# def func2(a=2, **vardict):
#     print(vardict)
#     func1(a, vardict)
#
#
# func2(a=3, b=1, c=55, d=11)

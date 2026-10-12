from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树模型
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 预测
y_pred = dt_model.predict(X_test)

# 评估
acc = accuracy_score(y_test, y_pred)
print(f"模型准确率：{acc:.2f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=class_names))

# 打印决策树规则文本
tree_rules = export_text(dt_model, feature_names=feature_names)
print("\n决策树规则：")
print(tree_rules)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset and filter the two classes
iris = datasets.load_iris()
X = iris.data[(iris.target == 1) | (iris.target == 2)]
y = iris.target[(iris.target == 1) | (iris.target == 2)]

# Train and validation set split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

n_lambdas = 100
lambdas = np.logspace(-4, 2, n_lambdas)
coefs_ridge, coefs_lasso = [], []
train_loss_ridge, train_loss_lasso = [], []
val_loss_ridge, val_loss_lasso = [], []
train_acc_ridge, train_acc_lasso = [], []
val_acc_ridge, val_acc_lasso = [], []

for a in lambdas:
    # Ridge
    ridge = LogisticRegression(penalty='l2', C=1 / a, solver='saga', max_iter=5000)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_[0])
    train_loss_ridge.append(log_loss(y_train, ridge.predict_proba(X_train)))
    val_loss_ridge.append(log_loss(y_val, ridge.predict_proba(X_val)))
    train_acc_ridge.append(accuracy_score(y_train, ridge.predict(X_train)))
    val_acc_ridge.append(accuracy_score(y_val, ridge.predict(X_val)))

    # Lasso
    lasso = LogisticRegression(penalty='l1', C=1 / a, solver='saga', max_iter=5000)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_[0])
    train_loss_lasso.append(log_loss(y_train, lasso.predict_proba(X_train)))
    val_loss_lasso.append(log_loss(y_val, lasso.predict_proba(X_val)))
    train_acc_lasso.append(accuracy_score(y_train, lasso.predict(X_train)))
    val_acc_lasso.append(accuracy_score(y_val, lasso.predict(X_val)))

# Plot Ridge coefficients
ax = plt.gca()
ax.plot(lambdas, coefs_ridge)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("weights")
plt.title("Ridge Coefficients")
plt.axis("tight")
plt.legend(iris.feature_names)
plt.show()

# Plot Lasso coefficients
ax = plt.gca()
ax.plot(lambdas, coefs_lasso)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("weights")
plt.title("Lasso Coefficients")
plt.axis("tight")
plt.legend(iris.feature_names)
plt.show()

# Plot training and validation loss for Ridge
plt.plot(lambdas, train_loss_ridge, label='Train Loss')
plt.plot(lambdas, val_loss_ridge, label='Validation Loss')
plt.xscale("log")
plt.gca().set_xlim(plt.gca().get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("Loss")
plt.title("Ridge Loss")
plt.legend()
plt.show()

# Plot training and validation accuracy for Ridge
plt.plot(lambdas, train_acc_ridge, label='Train Accuracy')
plt.plot(lambdas, val_acc_ridge, label='Validation Accuracy')
plt.xscale("log")
plt.gca().set_xlim(plt.gca().get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("Ridge Accuracy")
plt.legend()
plt.show()

# Plot training and validation loss for Lasso
plt.plot(lambdas, train_loss_lasso, label='Train Loss')
plt.plot(lambdas, val_loss_lasso, label='Validation Loss')
plt.xscale("log")
plt.gca().set_xlim(plt.gca().get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("Loss")
plt.title("Lasso Loss")
plt.legend()
plt.show()

# Plot training and validation accuracy for Lasso
plt.plot(lambdas, train_acc_lasso, label='Train Accuracy')
plt.plot(lambdas, val_acc_lasso, label='Validation Accuracy')
plt.xscale("log")
plt.gca().set_xlim(plt.gca().get_xlim()[::-1])  # reverse axis
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("Lasso Accuracy")
plt.legend()
plt.show()


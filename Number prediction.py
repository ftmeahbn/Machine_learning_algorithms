import numpy as np
import cv2
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



digits = load_digits()

selected_digits = np.isin(digits.target , [4,7,8])
y = digits.target
X = digits.data
y_selected = digits.target[selected_digits]
X_selected = digits.data[selected_digits]



logistic_regression = LogisticRegression(C=5.0 , max_iter=10000)
logistic_regression.fit(X_selected, y_selected)

svm = SVC( C=5.0 , kernel="rbf" , gamma="scale")
svm.fit(X_selected, y_selected)

knn = KNeighborsClassifier(n_neighbors=20 , weights="uniform")
knn.fit(X_selected , y_selected)

image = "8.png"

image = cv2.imread(image , cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(image , (8,8))
input_data = np.reshape(resized_img , (1,64))


predict_lr = logistic_regression.predict(input_data)

predict_knn = knn.predict(input_data)

predict_svm = svm.predict(input_data)



print("predict svm :" ,predict_lr)
print("predict LR :" , predict_svm)
print("predict knn:" , predict_knn)
print()

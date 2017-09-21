from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics


#Loading the data from the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,color=True,funneled=False,slice_=(slice(0,250),slice(0,250)))



n_samples, h, w ,k= lfw_people.images.shape


X = lfw_people.data

print(X[0].shape)

n_features = X.shape[1]
print("Printing shape of X")
print(X.shape)

y = lfw_people.target

print("Value of y:")

target_names = lfw_people.target_names

print("Printing target names")
print(target_names)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# splitting into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training dataset:",X_train.shape)
print("Training classification:",y_train.shape)

print("Testing dataset:",X_test.shape)
print("Testing classification:",y_test.shape)

#Since we have 966 images, we will have 966 Eigenfaces(Eigenvectors)
#Since L will be of 966 X 966

#No of Eigenfaces taken into consideration
n_components = 200

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

print("Executing PCA:")
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w,k))
#Using PCA to transform the data

#By transformation we mean projecting the data onto the selected number of eigenvectors
print("Projecting the data onto the selected number of Eigenvectors")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Transformed training dataset:",X_train_pca.shape)
print("Transformed testing dataset :",X_test_pca.shape)



print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

print("Starting SVM:")

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)



print("Predicting people's names on the test set")

y_pred = clf.predict(X_test_pca)


#Calculating accuracy
accuracyscore=metrics.accuracy_score(y_test,y_pred)

print("Accuracy :",accuracyscore)


#Generating classification report

print("Classification report")
print(classification_report(y_test, y_pred, target_names=target_names))



def plot_gallery(images, titles, h, w, n_row=3, n_col=4):

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w,k)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())




def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w,k)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()






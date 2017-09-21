print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn import svm

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

print("Shape of X:",X[0].shape)
print("Shape pf y:",y.shape)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

w = clf.coef_[0]
print("Weights of features:",w)
#Role of w???? in RFE
#RFE ranks the pixels but how does it eliminates the features?

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
X_r=rfe.transform(X)

print("Shape of X:",X_r.shape)


ranking = rfe.ranking_.reshape(digits.images[0].shape)

print("Ranking:",ranking)
print("Ranking Shape:",ranking.shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

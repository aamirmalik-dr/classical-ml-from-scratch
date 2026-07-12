"""classml: classical machine learning algorithms from scratch in NumPy.

Every estimator follows one consistent, scikit-learn-style interface: construct
with hyperparameters, call ``fit(X, y)`` which returns ``self``, then ``predict``
(and ``predict_proba``, ``transform``, or ``score`` where they apply). scikit-learn
is used only as a test-time reference and a data source, never inside the
algorithms themselves.
"""

from .decision_tree import DecisionTreeClassifier
from .gmm import GaussianMixtureEM
from .kmeans import KMeans
from .knn import KNeighborsClassifier
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import GaussianNB
from .pca import PCA
from .svm import KernelSVM, rbf_kernel

__all__ = [
    "DecisionTreeClassifier",
    "GaussianMixtureEM",
    "GaussianNB",
    "KMeans",
    "KNeighborsClassifier",
    "KernelSVM",
    "LinearRegression",
    "LogisticRegression",
    "PCA",
    "rbf_kernel",
]

__version__ = "0.2.0"

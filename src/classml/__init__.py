"""classml: classical machine learning algorithms from scratch in NumPy."""

from .gmm import GaussianMixtureEM
from .kmeans import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .pca import PCA
from .svm import KernelSVM, rbf_kernel

__all__ = [
    "GaussianMixtureEM",
    "KMeans",
    "KernelSVM",
    "LinearRegression",
    "LogisticRegression",
    "PCA",
    "rbf_kernel",
]

__version__ = "0.1.0"

# classml API reference

Every estimator in `classml` follows one contract, borrowed from scikit-learn and
implemented independently on top of NumPy:

- construct with hyperparameters,
- call `fit(X, y)`, which always returns `self`,
- then `predict(X)`, and where they apply `predict_proba(X)`, `transform(X)`,
  `decision_function(X)`, and `score(...)`.

`X` is array-like of shape `(n_samples, n_features)`. A 1D input is reshaped to a
single column. Inputs are validated by `classml.base.check_array`, which rejects
non-finite values. Fitted attributes carry a trailing underscore, for example
`coef_`, `classes_`, `cluster_centers_`.

scikit-learn is used only for reference tests and toy data. No algorithm in this
package imports it.

## Base classes

`classml.base` defines the shared contract.

- `Estimator` (abstract): declares `fit` and provides `_check_fitted(attr)`.
- `Regressor(Estimator)`: adds `score(X, y)` returning the R^2 coefficient of
  determination.
- `Classifier(Estimator)`: adds `score(X, y)` returning mean accuracy.
- `check_array(X, name="X") -> np.ndarray`: validate and convert to 2D float64.

To add your own estimator, subclass `Classifier` or `Regressor`, implement `fit`
(return `self`) and `predict`, and you inherit `score` and the fitted-state guard.
See `examples/custom_estimator.py`.

## Supervised estimators

### `LinearRegression(solver="closed_form", learning_rate=0.01, n_iter=5000, tol=1e-10)`

Ordinary least squares. `solver="closed_form"` solves the normal equations by
least squares, `solver="gd"` runs batch gradient descent.

- `fit(X, y) -> self`
- `predict(X) -> (n_samples,)`
- `score(X, y) -> float` (R^2)
- Attributes: `coef_`, `intercept_`, `loss_history_` (gradient descent only).

### `LogisticRegression(learning_rate=0.1, n_iter=2000, l2=0.0, tol=1e-8)`

Binary or one-vs-rest multiclass logistic regression by batch gradient descent
with optional L2.

- `fit(X, y) -> self`
- `predict(X) -> (n_samples,)`
- `predict_proba(X) -> (n_samples, n_classes)`
- `score(X, y) -> float` (accuracy)
- Attributes: `classes_`, `coef_`, `intercept_`, `loss_history_`.

### `KNeighborsClassifier(n_neighbors=5, weights="uniform")`

Lazy k-nearest-neighbours vote by Euclidean distance. `weights` is `"uniform"` or
`"distance"`.

- `fit(X, y) -> self` (stores the training set)
- `predict(X) -> (n_samples,)`
- `predict_proba(X) -> (n_samples, n_classes)`
- `score(X, y) -> float` (accuracy)
- Attributes: `classes_`, `X_`, `y_`.

### `DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)`

CART classification tree grown by greedy Gini-impurity splitting.

- `fit(X, y) -> self`
- `predict(X) -> (n_samples,)`
- `score(X, y) -> float` (accuracy)
- Attributes: `classes_`, `n_classes_`, `root_`, `depth_`.

### `GaussianNB(var_smoothing=1e-9)`

Gaussian naive Bayes with a per-feature Gaussian per class, scored in log space.

- `fit(X, y) -> self`
- `predict(X) -> (n_samples,)`
- `predict_proba(X) -> (n_samples, n_classes)`
- `score(X, y) -> float` (accuracy)
- Attributes: `classes_`, `priors_`, `theta_`, `var_`.

### `KernelSVM(C=1.0, gamma="scale", tol=1e-3, max_passes=5, max_iter=200, random_state=None)`

Binary soft-margin RBF support vector machine solved by simplified SMO.

- `fit(X, y) -> self` (exactly two classes)
- `predict(X) -> (n_samples,)`
- `decision_function(X) -> (n_samples,)`
- `score(X, y) -> float` (accuracy)
- Attributes: `classes_`, `support_`, `alpha_`, `support_vectors_`,
  `support_labels_`, `b_`, `gamma_`.
- Helper: `rbf_kernel(A, B, gamma) -> (n, m)`.

## Unsupervised estimators

### `KMeans(n_clusters=8, n_init=10, max_iter=300, tol=1e-6, random_state=None)`

Lloyd's algorithm with k-means++ seeding and restarts.

- `fit(X) -> self`
- `predict(X) -> (n_samples,)` (nearest centroid)
- Attributes: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`.

### `PCA(n_components=None)`

Principal component analysis via the SVD of the centered data.

- `fit(X) -> self`
- `transform(X) -> (n_samples, n_components)`
- `fit_transform(X) -> (n_samples, n_components)`
- `inverse_transform(Z) -> (n_samples, n_features)`
- Attributes: `components_`, `explained_variance_`,
  `explained_variance_ratio_`, `singular_values_`, `mean_`.

### `GaussianMixtureEM(n_components=2, max_iter=200, tol=1e-6, reg_covar=1e-6, random_state=None)`

Full-covariance Gaussian mixture fitted by expectation-maximisation in log space,
initialised from an internal k-means run.

- `fit(X) -> self`
- `predict(X) -> (n_samples,)` (hard assignment)
- `predict_proba(X) -> (n_samples, n_components)` (responsibilities)
- `score(X) -> float` (mean log-likelihood)
- Attributes: `weights_`, `means_`, `covariances_`,
  `log_likelihood_history_`, `n_iter_`, `converged_`.

## Minimal example

```python
import numpy as np
from classml import KNeighborsClassifier

table = np.loadtxt("data/sample_moons.csv", delimiter=",", skiprows=1)
X, y = table[:, :2], table[:, 2].astype(int)

model = KNeighborsClassifier(n_neighbors=15).fit(X, y)
print(model.predict(X[:5]))
print(model.score(X, y))
```

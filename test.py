import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import numpy
import compare_auc_delong_xu


def test_variance():
    data = sklearn.datasets.load_iris()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        data.data, data.target == 1, test_size=0.8, random_state=42)
    predictions = sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train).predict_proba(x_test)[:, 1]
    auc, variance = compare_auc_delong_xu.delong_roc_variance(y_test, predictions)
    true_auc = sklearn.metrics.roc_auc_score(y_test, predictions)
    numpy.testing.assert_allclose(true_auc, auc)
    numpy.testing.assert_allclose(0.0014569635512, variance)

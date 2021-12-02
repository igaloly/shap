import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap031

def test_random_force_plot_mpl_with_data():
    """Test if force plot with matplotlib works"""
    import sklearn.ensemble
    import shap031
    # train model
    X, y = shap031.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap031.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap031.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)

def test_random_force_plot_mpl_text_rotation_with_data():
    """Test if force plot with matplotlib works when supplied with text_rotation"""
    import sklearn.ensemble
    import shap031
    # train model
    X, y = shap031.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap031.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap031.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, text_rotation=30, show=False)
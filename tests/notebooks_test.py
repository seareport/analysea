import glob

from pytest_notebook.nb_regression import NBRegressionFixture


def test_notebook(nb_regression: NBRegressionFixture):
    notebooks = glob.glob("./notebooks/*.ipynb")
    for notebook in notebooks:
        nb_regression.diff_ignore = (
            "/cells/*/outputs/*",
            "/cells/*/execution_count",
            "/metadata/language_info/version",
        )
        nb_regression.check(notebook)

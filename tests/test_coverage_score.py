"""CPU-only tests of the released coverage methods, without SHAP/model imports.

Execute each actual method body via AST; only notebook display/highlighting are
substituted. NumPy, pandas comparisons, and the displayed Styler data are real.
Run with: python -m pytest tests/test_coverage_score.py -q
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("des/base.py", "BaseDES"),
    ("des/desknn.py", "DESKNN"),
    ("des/knop.py", "KNOP"),
    ("des/knorae_w.py", "KNORAE_W"),
    ("des/knorau_w.py", "KNORAU_W"),
    ("dcs/lca.py", "LCA"),
    ("dcs/mla.py", "MLA"),
    ("dcs/ola.py", "OLA"),
    ("dcs/rank.py", "Rank"),
]


@pytest.fixture(params=METHODS, ids=lambda item: item[1])
def coverage_method(request):
    relative, name = request.param
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_coverage_score")
    displayed = []
    namespace = {"np": np, "pd": pd, "display": displayed.append,
                 "make_table_highlights": lambda style, title: style.set_caption(title)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_coverage_score"], displayed


CASES = [
    pytest.param([1, 1, 1, 1], [[0, 0, 0, 0]] * 2, [0, 1, 2, 3], False, False, id="binary-all-wrong"),
    pytest.param([0, 0, 0, 0], [[0, 0, 0, 0]] * 2, [0, 1, 2, 3], False, False, id="binary-all-correct-control"),
    pytest.param([2, 5, 9, 2], [[2, 5, 9, 2]] * 2, [0, 1, 2, 3], False, False, id="multiclass-all-correct"),
    pytest.param([2, 5, 9, 2], [[0, 0, 0, 0]] * 3, [0, 1, 2, 3], False, False, id="multiclass-all-wrong"),
    pytest.param([0, 1, 2, 1], [[9, 1, 2, 1], [9, 9, 2, 1], [9, 9, 9, 1]], [0, 1, 2, 3], False, False, id="all-contribution-levels"),
    pytest.param([0, 0, 0, 0], [[0, 0, 0, 0]] * 2, [11, 3, 20, 7], False, False, id="nonconsecutive-index"),
    pytest.param([0, 1, 2, 1], [[9, 1, 2, 1], [9, 9, 2, 1], [9, 9, 9, 1]], [2, 0, 3, 1], False, False, id="shuffled-index"),
    pytest.param([0, 0, 0, 0], [[0, 0, 0, 0]] * 2, [9, 9, 3, 3], False, False, id="duplicate-index"),
    pytest.param(["cat", "dog", "bird", "cat"], [["cat", "bird", "bird", "dog"], ["dog", "dog", "cat", "dog"]], [11, 3, 20, 7], True, False, id="string-labels-series-predictions"),
    pytest.param([-1, 1, -1, 1], [[-1, 1, 1, -1]], [2, 0, 3, 1], False, True, id="single-model-array-labels"),
]


@pytest.mark.parametrize("labels,predictions,index,series_predictions,array_labels", CASES)
def test_coverage_matches_independent_reference(coverage_method, capsys, labels, predictions, index, series_predictions, array_labels):
    method, displayed = coverage_method
    X = pd.DataFrame({"row": range(len(labels)), "other": range(len(labels))}, index=index)
    y = np.array(labels) if array_labels else pd.Series(labels, index=index)

    class Estimator:
        def __init__(self, values, columns):
            self.values = values
            self.columns = columns

        def predict(self, frame):
            assert list(frame.columns) == self.columns
            assert list(frame.index) == index
            # sklearn-style prediction order follows input rows. A Series must
            # not cause pandas to realign prediction values by its own index.
            values = [self.values[int(row)] for row in frame.iloc[:, 0]]
            return pd.Series(values, index=[40, 30, 20, 10]) if series_predictions else np.array(values)

    subsets = [["row"] if i % 2 == 0 else ["other"] for i in range(len(predictions))]
    obj = SimpleNamespace(X_dsel=X, y_dsel=y, feature_subsets=subsets,
                          pool_classifiers=[Estimator(values, columns) for values, columns in zip(predictions, subsets)])
    original_X, original_y = X.copy(deep=True), y.copy()
    assert method(obj) is None
    # Pure Python reference, independent of pandas aggregation/alignment.
    contributions = [sum(values[row] == label for values in predictions) for row, label in enumerate(labels)]
    counts = [contributions.count(level) for level in range(len(predictions) + 1)]
    score = round(100 * sum(level > 0 for level in contributions) / len(labels), 2)
    assert capsys.readouterr().out == f"Coverage score: {score}%\nNumber of models: {len(predictions)}\n"
    assert len(displayed) == 1
    styler = displayed[0]
    assert styler.caption == "Coverage Scores"
    table = styler.data
    assert list(table.columns) == ["Contribution", "Count", "Percentage"]
    assert table["Contribution"].tolist() == [f"Contributors {level}/{len(predictions)}" for level in range(len(predictions) + 1)]
    assert table["Count"].tolist() == counts
    assert sum(table["Count"]) == len(labels)
    assert table["Percentage"].tolist() == [round(count / len(labels) * 100, 3) for count in counts]
    assert sum(table["Percentage"]) == pytest.approx(100, abs=.001 * len(counts))
    assert sum(level * count for level, count in enumerate(table["Count"])) == sum(contributions)
    pd.testing.assert_frame_equal(X, original_X)
    if array_labels:
        np.testing.assert_array_equal(y, original_y)
    else:
        pd.testing.assert_series_equal(y, original_y)

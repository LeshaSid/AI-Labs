import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
    from sklearn.metrics import root_mean_squared_error, classification_report, roc_curve, auc
    import matplotlib.pyplot as plt

    df = pd.read_csv("Lab1/processed_automobile.csv")
    return (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        auc,
        classification_report,
        df,
        plot_tree,
        plt,
        roc_curve,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Классификация
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    auc,
    classification_report,
    df,
    plt,
    roc_curve,
    train_test_split,
):
    X_cl = df.drop(['origin_usa', 'origin_japan'], axis=1)
    y_cl = df['origin_usa'].astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cl, y_cl, test_size=0.3, random_state=42)

    model_dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_dt_clf.fit(X_train_c, y_train_c)

    y_probs = model_dt_clf.predict_proba(X_test_c)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test_c, y_probs)
    roc_auc = auc(fpr, tpr)


    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc}")
    print(classification_report(y_test_c, model_dt_clf.predict(X_test_c)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $Precision = \frac{TP}{TP + FP}$ - точность целевой классификации

    $Recall = \frac{TP}{TP + FN}$ - полнота

    $F_1 = 2×\frac{precision × recall}{(precision + recall)}$

    ### ROC-кривая
    $TPR = \frac{TP}{TP + FN}$

    $FPR = \frac{FP}{TN + FP}$

    ### ID3
    Энтропия
    $S = - \sum_{i=0}^{C} p_i log_2 p_i$

    где C — общее количество целевых классов, а pi — вероятность появления объекта i-го класса в выборке.

    ### Информационный выйгрыш
    $IG_k(j, a_k) = S_0 - (\frac{N_1}{N} S_1 + \frac{N_2}{N} S_2 )$, если j-й признак — вещественный

    $IG_k(j, X_j^k ) = S_0 - (\frac{N_1}{N} S_1 + \frac{N_2}{N} S_2 )$, если j-й признак — категориальный

    где S0 и N — энтропия выборки попавшей в k–ую вершину и количество объектов в этой выборке соответственно, S1 и S2 — энтропии подвыборок, а N1 и N2 — количество объектов в подвыборках.

    ### CART
    $S = 1 - \sum_{i=0}^C p_i^2$ критерий Джини (Gini Impurity)

    где C — общее количество целевых классов, а pi — вероятность появления объекта i-го класса в выборке.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Регрессия
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    df,
    plot_tree,
    plt,
    root_mean_squared_error,
    train_test_split,
):
    X_reg = df.drop(["model_year"], axis=1)
    y_reg = df["model_year"]

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    model_dt_reg = DecisionTreeRegressor(max_depth=3, random_state=42, min_samples_leaf=10)
    model_dt_reg.fit(X_train, y_train)

    model_dt_reg.fit(X_train, y_train)
    y_pred_reg = model_dt_reg.predict(X_test)
    y_pred_reg_train = model_dt_reg.predict(X_train)
    rmse = root_mean_squared_error(y_test, y_pred_reg)
    rmse_train = root_mean_squared_error(y_train, y_pred_reg_train)

    print(f"RMSE Дерева решений на обучающей выборке: {rmse_train}")
    print(f"RMSE Дерева решений на тестовой выборке: {rmse}")

    plt.figure(figsize=(12, 8))
    plot_tree(model_dt_reg, feature_names=X_reg.columns, filled=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $Avg(R) = \frac{1}{|R|} \sum_{y^i ∈ R} y^i$

    $H(R) = \sum_{y^i ∈ R}(Avg(R) - y^i)^2$

    $IG_k(j, a_k ) = H(R_k) - (\frac{|R_l|}{|R_k|} H(R_l) + \frac{|R_r|}{|R_k|} H(R_r)) → max$
    """)
    return


if __name__ == "__main__":
    app.run()

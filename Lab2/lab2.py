import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Регрессия

    ### MinMaxScaler
    $X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$
    """)
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error

    df = pd.read_csv("Lab1/processed_automobile.csv")
    X = df.drop(["model_year"], axis=1)
    y = df["model_year"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    def in_real(x):
        return x * (82 - 70) + 70

    y_pred_test = linear_model.predict(X_test)
    RMSE_real = root_mean_squared_error(in_real(y_test), in_real(y_pred_test))
    print(f"Реальный RMSE составил: {RMSE_real:.2f} года")
    return (
        X_test,
        X_train,
        df,
        in_real,
        linear_model,
        root_mean_squared_error,
        train_test_split,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Линейная регрессия

    $a(w, x) = w_0 + w_1⋅ x_1 + w_2⋅ x_2 + … + w_n⋅ x_n$

    $L_i (w) = |a(w, x^i) - y^i|$ — абсолютная ошибка(MAE)

    $L_i (w) = (a(w, x^i) - y^i)^2$— квадратичная ошибка(MSE)

    $L_i$ - функция потерь

    $Q(w) = \frac{1}{l} \sum_{i=0}^l L_i (w) → min$ - функция стоимости
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    in_real,
    linear_model,
    root_mean_squared_error,
    y_test,
    y_train,
):
    from sklearn.preprocessing import PolynomialFeatures
    n = 2
    poly_features = PolynomialFeatures(n)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)

    linear_model.fit(X_train_poly, y_train)
    y_pred_test_poly = linear_model.predict(X_test_poly)
    RMSE_poly_real = root_mean_squared_error(in_real(y_test), in_real(y_pred_test_poly))
    print(f"Реальный RMSE составил: {RMSE_poly_real:.2f} года")
    return PolynomialFeatures, X_test_poly, X_train_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Полиномиалная регрессия

    $a(w, x) = w_0 + w_1⋅ x + w_2⋅ x^2 + … + w_n⋅ x^n$
    """)
    return


@app.cell
def _(
    X_test_poly,
    X_train_poly,
    in_real,
    root_mean_squared_error,
    y_test,
    y_train,
):
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=0.1) 
    ridge_model.fit(X_train_poly, y_train)

    y_pred_ridge = ridge_model.predict(X_test_poly)
    RMSE_ridge_real = root_mean_squared_error(in_real(y_test), in_real(y_pred_ridge))
    print(f"Реальный RMSE составил: {RMSE_ridge_real:.2f} года")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $Q_{L1}(w) = \sum_{i=0}^l L_i (w) + λ ||w||_1 = \sum_{i=0}^l L_i (w) + λ \sum_{i=1}^n |w_i| → min$

    $Q_{L2}(w) = \sum_{i=0}^l L_i (w) + λ ||w||_2^2 = \sum_{i=0}^l L_i (w) + λ \sum_{i=1}^n w_i^2 → min$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Классификация

    $Q(w) =\sum_{i=0}^l [a(w, x^i) ≠ y^i] → min$ - функция стоимости

    $Q(w) =\sum_{i=0}^l [a(w, x^i)⋅ y^i < 0] → min$
    $M_i =〈w, x^i〉⋅ y^i$ - отступ
    $Q(w) =\sum_{i=0}^l [a(w, x^i)⋅ y^i < 0] = \sum_{i=0}^l [〈w, x^i〉⋅ y^i < 0] =\sum_{i=0}^l [M_i < 0] → min$

    ## Логистическая регрессия

    $L_{log}^i (w) = -(y_i⋅ log(p_i) +(1 - y_i)⋅ log(1 - p_i))$ - функция потерь

    $Q(w) = \sum_{i=0}^l L_{log}^i (w) → min$ - функция стоимости

    $Precision = \frac{TP}{TP + FP}$ - точность целевой классификации

    $Recall = \frac{TP}{TP + FN}$ - полнота

    $F_1 = 2×\frac{precision × recall}{(precision + recall)}$
    """)
    return


@app.cell
def _(df, train_test_split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt


    y_cl = df['origin_usa'].astype(int) 
    X_cl = df.drop(['origin_usa', 'origin_japan'], axis=1)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cl, y_cl, test_size=0.4, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_c, y_train_c)

    y_pred_cl = log_reg.predict(X_test_c)
    print("Отчет по классификации:")
    print(classification_report(y_test_c, y_pred_cl))

    cm = confusion_matrix(y_test_c, y_pred_cl)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return (
        LogisticRegression,
        X_test_c,
        X_train_c,
        classification_report,
        cm,
        confusion_matrix,
        plt,
        sns,
        y_test_c,
        y_train_c,
    )


@app.cell
def _(
    LogisticRegression,
    PolynomialFeatures,
    X_test_c,
    X_train_c,
    classification_report,
    cm,
    confusion_matrix,
    plt,
    sns,
    y_test_c,
    y_train_c,
):
    poly_cl = PolynomialFeatures(degree=2, include_bias=False)

    X_train_poly_cl = poly_cl.fit_transform(X_train_c)
    X_test_poly_cl = poly_cl.transform(X_test_c)

    log_reg_poly = LogisticRegression(max_iter=1000, C=1.0) 
    log_reg_poly.fit(X_train_poly_cl, y_train_c)

    y_pred_poly_cl = log_reg_poly.predict(X_test_poly_cl)

    print("Отчет по классификации:")
    print(classification_report(y_test_c, y_pred_poly_cl))
    cm2 = confusion_matrix(y_test_c, y_pred_poly_cl)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


if __name__ == "__main__":
    app.run()

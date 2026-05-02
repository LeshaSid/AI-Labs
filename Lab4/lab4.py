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
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_curve, auc
    import matplotlib.pyplot as plt

    df = pd.read_csv("Lab1/processed_automobile.csv")
    return (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        auc,
        classification_report,
        df,
        plt,
        roc_curve,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Бэггинг
    ## Случайный лес
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    auc,
    classification_report,
    df,
    plt,
    roc_curve,
    train_test_split,
):
    X = df.drop(['origin_usa', 'origin_japan'], axis=1)
    y = df['origin_usa'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_rf = RandomForestClassifier(oob_score=True)
    model_rf.fit(X_train, y_train)
    print("OOB Accuracy:", model_rf.oob_score_)
    print("OOBE: ", 1 - model_rf.oob_score_)

    y_probs_rf = model_rf.predict_proba(X_test)[:, 1]

    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_probs_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)


    plt.figure()
    plt.plot(fpr_rf, tpr_rf, color='darkorange', label=f'ROC curve (area = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc_rf}")
    print(classification_report(y_test, model_rf.predict(X_test)))
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Оценка Out-Of-Bag (OOB)

    Вероятность того, что объект **не** попадет в обучающую выборку при бутстрэпе (при $n \to \infty$):
    $$P_{not\_in} = \left(1 - \frac{1}{n}\right)^n \approx \frac{1}{e} \approx 0.368$$
    Следовательно, в обучение попадает примерно **63.2%** уникальных объектов.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Особенности Случайного леса (Random Forest)
    * **Случайные признаки:** В каждом узле выбирается лучший признак из $m$ случайных.
        * Для классификации: $m = \lfloor \sqrt{n} \rfloor$
        * Для регрессии: $m = \lfloor n/3 \rfloor$
    * **Переобучение:** Увеличение количества деревьев ($T$) не приводит к переобучению случайного леса.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Бустинг

    Для задач бинарной классификации:
    $$ Q(a, b) = \sum_{i=1}^l[yi ∙ \sum_{t=1}^Tα_t b_t(x_i) < 0] → min$$

    ## AdaBoost
    Экспоненциальная функция стоимости:
    $$Q (αt , bt(x)) = \sum_{i=1}^lexp(- y_i ∙ \sum_{t=1}^Tα_t b_t(x_i)) → min$$

    Оптимизируется функция потерь:
    $$N(bt(x)) = \sum_{i=1}^lω_{i, t-1} [b_t(x_i) ≠ y_i] → min$$


    Коэффициент веса модели $\alpha_t$ вычисляется на основе её ошибки $N(b_t(x))$:
    $$\alpha_t = \frac{1}{2} \ln \frac{1 - N(b_t(x))}{N(b_t(x))}$$
    Обновление весов объектов для следующей итерации:
    $$\omega_{i, t} = \omega_{i, t-1} \cdot \exp(-y_i \cdot \alpha_t \cdot b_t(x_i))$$
    """)
    return


@app.cell
def _(
    AdaBoostClassifier,
    X_test,
    X_train,
    auc,
    classification_report,
    plt,
    roc_curve,
    y_test,
    y_train,
):
    model_ada = AdaBoostClassifier()
    model_ada.fit(X_train, y_train)

    y_probs_ada = model_ada.predict_proba(X_test)[:, 1]

    fpr_ada, tpr_ada, thresholds_ada = roc_curve(y_test, y_probs_ada)
    roc_auc_ada = auc(fpr_ada, tpr_ada)


    plt.figure()
    plt.plot(fpr_ada, tpr_ada, color='darkorange', label=f'ROC curve (area = {roc_auc_ada:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc_ada}")
    print(classification_report(y_test, model_ada.predict(X_test)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Градиентный бустинг

    Сумма выходов модели:
    $$f_{T-1, i} = \sum_{t=1}^{T -1}α_t b_t(x_i) , i = 1,...,l$$

    Функция стоимости:
    $$Q (f_T) = \sum_{i=0}^l L_i (f_{T, i} , y_i ) → min$$

    Переходим к градиентному спуску:
    $$f_T = f_{T-1} - αg$$

    Каждая новая модель $b_t(x)$ обучается предсказывать антиградиент функции потерь (псевдо-остатки):
    $$g_i = \frac{\partial L(f_{t-1, i}, y_i)}{\partial f_{t-1, i}}$$
    Итоговая композиция:
    $$a(x) = \sum_{t=1}^{T} \alpha_t b_t(x)$$
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    X_test,
    X_train,
    auc,
    classification_report,
    plt,
    roc_curve,
    y_test,
    y_train,
):
    model_gb = GradientBoostingClassifier()
    model_gb.fit(X_train, y_train)

    y_probs_gb = model_gb.predict_proba(X_test)[:, 1]

    fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_probs_gb)
    roc_auc_gb = auc(fpr_gb, tpr_gb)


    plt.figure()
    plt.plot(fpr_gb, tpr_gb, color='darkorange', label=f'ROC curve (area = {roc_auc_gb:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc_gb}")
    print(classification_report(y_test, model_gb.predict(X_test)))
    return


if __name__ == "__main__":
    app.run()

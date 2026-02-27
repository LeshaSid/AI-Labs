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
    # Загрузка датасета
    Датасет взят с [Kaggle](https://www.kaggle.com/datasets/tawfikelmetwally/automobile-dataset)
    """)
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    df = pd.read_csv("Lab1/Automobile.csv")
    df.head()
    return MinMaxScaler, df, pd


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Пропущено 6 значений в столбце horsepower
    """)
    return


@app.cell
def _(df):
    horsepower_median = df["horsepower"].median()
    df["horsepower"] = df["horsepower"].fillna(horsepower_median)
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Заполнил пропущенные значения медианой
    """)
    return


@app.cell
def _(MinMaxScaler, df):
    scaler = MinMaxScaler()
    df["mpg"] = scaler.fit_transform(df[["mpg"]])
    df["cylinders"] = scaler.fit_transform(df[["cylinders"]])
    df["displacement"] = scaler.fit_transform(df[["displacement"]])
    df["horsepower"] = scaler.fit_transform(df[["horsepower"]])
    df["weight"] = scaler.fit_transform(df[["weight"]])
    df["acceleration"] = scaler.fit_transform(df[["acceleration"]])
    df["model_year"] = scaler.fit_transform(df[["model_year"]])
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Нормализация данных
    """)
    return


@app.cell
def _(df, pd):
    df_new = pd.get_dummies(df, columns=["origin"], drop_first=True)
    df_new.to_csv("Lab1/processed_automobile.csv", index=False)
    df_new.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Преобразование категориальных данных и сохранение в файл
    """)
    return


if __name__ == "__main__":
    app.run()

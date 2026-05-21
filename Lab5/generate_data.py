import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return (np,)


@app.cell
def _(Y, np):
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(1000, 12))  # 100 примеров, 12 бинарных признаков
    #Y = np.random.randint(0, 2, size=(1000, 2))  # 100 примеров, 2 класса (one-hot encoding)

    # Сохранение данных в файлы
    np.savetxt('Lab5/dataIn.txt', X, fmt='%d')
    np.savetxt('Lab5/dataOut.txt', Y, fmt='%d')
    return


if __name__ == "__main__":
    app.run()

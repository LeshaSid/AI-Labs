import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return (np,)


@app.cell
def _(np):
    np.random.seed(42)
    N = 1000

    X = np.random.randint(0, 2, size=(N, 12))

    key_questions_sum = X[:, 0] + X[:, 4] + X[:, 8]

    labels = (key_questions_sum >= 2).astype(int)

    noise_mask = np.random.rand(N) < 0.05
    labels[noise_mask] = 1 - labels[noise_mask]

    Y = np.zeros((N, 2), dtype=int)
    Y[labels == 1] = [1, 0]
    Y[labels == 0] = [0, 1]

    np.savetxt('Lab5/dataIn.txt', X.T, fmt='%d')
    np.savetxt('Lab5/dataOut.txt', Y.T, fmt='%d')
    return


if __name__ == "__main__":
    app.run()

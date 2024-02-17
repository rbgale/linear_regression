import pandas as pd
import matplotlib.pyplot as plt
import math

data: pd.core.frame.DataFrame = pd.read_csv('data.csv')


def gradient_descent(
        m_old: float, b_old: float,
        points: pd.core.frame.DataFrame,
        learning_rate: float
) -> tuple[float, float]:
    """
    Given some m, b values for a data set, calculate the
    gradient of error and adjust m, b by an amount equal
    to the gradient multiplied by the learning rate.

    Args:
        m_old (float): original slope
        b_old (float): origin y intercept
        points (pd.core.frame.DataFrame): set of points

    Returns:
        m, b (tuple(float, float)): new m, b values
    """
    m_gradient: float = 0.0
    b_gradient: float = 0.0

    n: int = len(points)

    for i in range(n):
        x: float = points.iloc[i].horizontal
        y: float = points.iloc[i].vertical

        m_gradient += (-(2/n) * x * (y - (m_old * x + b_old)))
        b_gradient += (-(2/n) * (y - (m_old * x + b_old)))

    m = m_old - m_gradient * learning_rate
    b = b_old - b_gradient * learning_rate
    return m,b


# set initial m, b values
# and desired learning rate & iterations
m: float = 0
b: float = 0
L: float = 0.0001
epochs: int = 101

for i in range(epochs):
    # perform gradient descent desired number of times
    if i % 100 == 0:
        print(f"Epoch: {i}")
        print(f"m: {m}\nb: {b}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)

# create plot
plt.scatter(data.horizontal, data.vertical, s = 1, color = "black")
plt.plot(list(range(0, 100)), [m * x + b for x in range (0, 100)], color = "red")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

petrol = np.array(
    [
        63122,
        60953,
        59551,
        58785,
        59795,
        60083,
        61819,
        63107,
        64978,
        66090,
        66541,
        67186,
        67396,
        67619,
        69006,
        70258,
        71880,
        73597,
        74274,
        75975,
        76928,
        77732,
        78457,
        80089,
        83063,
        84558,
        85566,
        86724,
        86046,
        84972,
        88157,
        89105,
        90340,
        91195,
    ]
)
# petrol consumption per year via straight-line fit
n = len(petrol)
A = np.matrix([np.ones(n), np.linspace(1, n, n)]).transpose()

x = np.linalg.lstsq(A, petrol, rcond=None)[0]

fig, ax = plt.subplots()
ax.scatter(range(1980, 2014), petrol, c='r')
ax.plot(range(1980, 2014), np.matmul(A, x).transpose(), 'b-')
plt.show()
plt.close(fig)
fig, ax = plt.subplots()
de_trended = petrol.flatten() - np.matmul(A, x).transpose().A1
ax.plot(range(1980, 2014), de_trended, 'ro-')
plt.show()
plt.close(fig)

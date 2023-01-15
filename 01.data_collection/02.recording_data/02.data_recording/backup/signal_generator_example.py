from scipy.signal import sweep_poly
import matplotlib.pyplot as plt
import numpy as np

p = np.poly1d([0.025, -0.36, 1.25, 2.0])

t = np.linspace(0, 10, 5001)

w = sweep_poly(t, p)


plt.subplot(2, 1, 1)

plt.plot(t, w)

plt.title("Sweep Poly\nwith frequency " +

          "$f(t) = 0.025t^3 - 0.36t^2 + 1.25t + 2$")

plt.subplot(2, 1, 2)

plt.plot(t, p(t), 'r', label='f(t)')

plt.legend()

plt.xlabel('t')

plt.tight_layout()

plt.show()

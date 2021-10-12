import prosail
import numpy as np
import matplotlib.pyplot as plt
from prosail import run_prosail

rr = run_prosail(1.5, 40., 8., 0.0, 0.01, 0.009, 3., -0.35, 0.01,
                         30., 10., 0., typelidf=1, lidfb=-0.15,
                         rsoil=1., psoil=1., factor="SDR")
plt.plot(np.arange(400, 2501), rr, 'r-')
plt.show()


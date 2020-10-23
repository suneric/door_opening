import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x = np.array([0,-1,-1,2,2,0.9])
    y = np.array([0,0,-3,-3,0,0])
    plt.figure(1)
    plt.xlim(-1.5,2.5)
    plt.ylim(-3,1)
    plt.axis('equal')
    plt.plot(x,y,linewidth=5.0,color='k')
    plt.grid(True)

    plt.show()

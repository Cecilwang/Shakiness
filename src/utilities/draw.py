# Author: Cecil Wang (cecilwang@126.com)

import matplotlib.pyplot as plt

def draw(values):
    for x in values:
        plt.plot(x)
    plt.show()

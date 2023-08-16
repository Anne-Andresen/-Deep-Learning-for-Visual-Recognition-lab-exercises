import numpy as np
import matplotlib.pyplot as plt

# initializing the constant
α = 1.0

def ELU(x):
    if x > 0:
        return x
    return α*(np.exp(x) - 1)

x = np.linspace(-5.0, 5.0)
result = []
for i in x:
    result.append(ELU(i))

plt.plot(x, result)
plt.title("ELU activation function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.savefig('elu_plot.png')
plt.show()
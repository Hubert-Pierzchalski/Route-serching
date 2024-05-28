import numpy as np
import matplotlib.pyplot as plt
y = np.zeros(50)
x = np.zeros(50)
for i in range(50):
    y= np.random.randint(-1,1)
    x = np.random.randint(-10,1)
    print(f"{x},y: {y}")
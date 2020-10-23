import numpy as np
import matplotlib.pyplot as plt


y1 = np.load("./glosses.npy")
y2 = np.load("./accs.npy")
print(y1)
x = np.arange(1, y1.size + 1, 1)
plt.figure()
plt.xlim((0, 1000))
plt.ylim((0.2, 0.7))

l1, = plt.plot(x, y1, label='generator losses')
l2, = plt.plot(x, y1, label='accuracy', color='red', linewidth=1.0, linestyle='--')
plt.legend(loc='upper right')
plt.show()
import numpy as np

Nx = 50
Ny = 50

x = np.linspace(-Nx//2, Nx//2, num=51)
y = np.linspace(-Ny//2, Ny//2, num=51)

X,Y = np.meshgrid(x,y)
print(X)
print(Y)
def alpha(t):
    return 50-2*t
def beta(t):
    return 0
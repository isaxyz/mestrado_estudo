import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 400)

# Activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# --- Linear function plot ---
plt.figure(figsize=(6, 4))
plt.plot(x, linear(x), label='Linear', color='orange')
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Sigmoid function plot ---
plt.figure(figsize=(6, 4))
plt.plot(x, sigmoid(x), label='Sigmoid', color='blue')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Tanh function plot ---
plt.figure(figsize=(6, 4))
plt.plot(x, tanh(x), label='Hyperbolic Tangent (tanh)', color='green')
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

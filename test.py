import numpy as np
import time

# Tworzenie dużych tablic losowych
arr_float32 = np.random.rand(10000, 10000).astype(np.float32)
arr_float64 = np.random.rand(10000, 10000).astype(np.float64)

# Pomiar czasu dla float32
start = time.time()
np.dot(arr_float32, arr_float32)
end = time.time()
print(f"Czas dla float32: {end - start:.5f} sekund")

# Pomiar czasu dla float64
start = time.time()
np.dot(arr_float64, arr_float64)
end = time.time()
print(f"Czas dla float64: {end - start:.5f} sekund")
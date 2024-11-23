import numpy as np
import time

# Tworzenie dużych tablic losowych
arr_float32 = np.random.rand(10000, 10000).astype(np.float32)
arr_float64 = np.random.rand(10000, 10000).astype(np.float64)
arr_default = np.random.rand(10000, 10000)

# Pomiar czasu dla float32
start = time.time()
res32 = np.dot(arr_float32, arr_float32)
end = time.time()
print(f"Czas dla float32: {end - start:.5f} sekund, typ: {res32.dtype}")

# Pomiar czasu dla float64
start = time.time()
res64 = np.dot(arr_float64, arr_float64)
end = time.time()
print(f"Czas dla float64: {end - start:.5f} sekund, typ: {res64.dtype}")

# Pomiar czasu dla domyślnego typu
start = time.time()
res_default = np.dot(arr_default, arr_default)
end = time.time()
print(f"Czas dla domyślnego typu: {end - start:.5f} sekund, typ: {res_default.dtype}")

# Pomiar czasu dla mieszanych typów
start = time.time()
res_mixed = np.dot(arr_default, arr_float32)
end = time.time()
print(f"Czas dla mieszanych typów: {end - start:.5f} sekund, typ: {res_mixed.dtype}")
import matplotlib.pyplot as plt
import numpy as np

# Path to your .npy file
files = ["1nteriors-1560485669103686348.jpg_1_original.npz",  "1nteriors-1560485669103686348.jpg_2_cropped.npz", "1nteriors-1560485669103686348.jpg_3_resized.npz","1nteriors-1560485669103686348.jpg_4_flipped.npz",  "1nteriors-1560485669103686348.jpg_5_normalized.npz"] 

for fileName in files:
    npy_file_path = f"/run/media/m1h1r/04E884E1E884D1FA/debugTensor/debug/interior/{fileName}"
    dirs = npy_file_path.split("/")
    print(f"{dirs[-1]}\n")

    # Load the numpy array from the file
    loaded_array = np.load(npy_file_path)['arr_0']

    # Print basic information about the array
    print("Array shape:", loaded_array.shape)
    print("Data type:", loaded_array.dtype)
    print("Min value:", loaded_array.min())
    print("Max value:", loaded_array.max())
    print("Mean value:", loaded_array.mean())

    # Print a small sample of the array (first few values)
    N = 5
    print(f"\nSample of the array (first {N}x{N} values of first channel):")
    print(loaded_array[:N, :N, 0])
    print("\n\n")

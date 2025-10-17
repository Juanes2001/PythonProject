import numpy as np
import matplotlib.pyplot as plt

# --- List your text files here ---
files = [
    "C:/Users/jero0/Desktop/LabView_Osita/Linterna_muestra1.txt",
    "C:/Users/jero0/Desktop/LabView_Osita/Linterna_muestra2.txt",
    "C:/Users/jero0/Desktop/LabView_Osita/Halogena_Muestra1.txt",
    "C:/Users/jero0/Desktop/LabView_Osita/Halogena_Muestra2.txt"
]

# --- Loop through files and plot each one ---
for i, file in enumerate(files, start=1):
    # Load the data (assuming comma-separated values, 2 columns)
    data = np.loadtxt(file, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]

    # Plot
    plt.figure(i)
    plt.plot(x, y, lw=1.5)
    plt.title(f"Spectrum {i}")
    plt.ylim(0,2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()

# Show all figures

plt.show()


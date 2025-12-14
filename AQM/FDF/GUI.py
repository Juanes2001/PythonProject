# ===============================
# Optical Reflectance & Transmittance GUI (TEMPLATE)
# ===============================
# This code creates a full Python user interface with:
# - 4 input entries: epsilon_av, delta, lambda_1, lambda_2
# - A button: "Calculate Reflectance and Transmittance"
# - Two plots:
# * Reflectance (%) vs Wavelength (nm)
# * Transmittance (%) vs Wavelength (nm)
#
# You only need to FILL IN the PHYSICS/MODEL FUNCTIONS
# where indicated.
#
# Requirements:
# pip install numpy matplotlib
# (tkinter comes with standard Python distributions)


import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import funciones as fun
import math



# -------------------------------------------------
# GUI APPLICATION
# -------------------------------------------------


class OpticalGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.polarization = tk.StringVar(value="LCP")  # default
        self.title("Reflectance & Transmittance Calculator")
        self.geometry("1200x650")

        self._build_inputs()
        self._build_plots()

    # -----------------------------
    # INPUT PANEL
    # -----------------------------
    def _build_inputs(self):
        input_frame = ttk.LabelFrame(self, text="Input Parameters")

        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.entries = {}
        parameters = [
            (r"ε_av", "epsilon_av"),
            (r"n (host)", "n_host"),
            (r"h", "width"),
            (r"p", "pitch"),
            (r"δ", "delta"),
            (r"λ₁ (nm)", "lambda_1"),
            (r"λ₂ (nm)", "lambda_2"),
        ]

        for i, (label, key) in enumerate(parameters):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(input_frame, width=15)
            entry.grid(row=i, column=1, pady=5)
            self.entries[key] = entry

        calc_button = ttk.Button(
            input_frame,
            text="Calculate Reflectance and Transmittance",
            command=self.calculate
        )
        calc_button.grid(row=len(parameters), column=0, columnspan=2, pady=15)

        pol_frame = ttk.LabelFrame(input_frame, text="Polarization")
        pol_frame.grid(row=len(parameters) + 1, column=0, columnspan=2, pady=10)

        ttk.Button(
            pol_frame,
            text="LCP",
            command=lambda: self.polarization.set("LCP")
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            pol_frame,
            text="RCP",
            command=lambda: self.polarization.set("RCP")
        ).grid(row=0, column=1, padx=5)

    # -----------------------------
    # PLOT AREA
    # -----------------------------
    def _build_plots(self):
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


        self.fig, (self.ax_R, self.ax_T) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.tight_layout(pad=4)


        self.ax_R.set_title("Reflectance")
        self.ax_R.set_xlabel("Wavelength (nm)")
        self.ax_R.set_ylabel("Reflectance (%)")
        self.ax_R.set_ylim(0, 100)


        self.ax_T.set_title("Transmittance")
        self.ax_T.set_xlabel("Wavelength (nm)")
        self.ax_T.set_ylabel("Transmittance (%)")
        self.ax_T.set_ylim(0, 100)


        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    # -----------------------------
    # CALCULATION CALLBACK
    # -----------------------------
    def calculate(self):
        try:
            epsilon_av = float(self.entries["epsilon_av"].get())**2
            n_host = float(self.entries["n_host"].get())  # ← ADD THIS
            width = float(self.entries["width"].get())
            pitch = float(self.entries["pitch"].get())# ← ADD THIS
            delta = float(self.entries["delta"].get())*math.sqrt(epsilon_av)/2
            lambda_1 = float(self.entries["lambda_1"].get())
            lambda_2 = float(self.entries["lambda_2"].get())
            polarization = self.polarization.get()
        except ValueError:
            messagebox.showerror("Input Error", "All parameters must be numeric")
            return


        wavelength_nm = np.linspace(lambda_1*1e-9, lambda_2*1e-9, 5000)


        R = 100*fun.find_amplitudes_LCP_RCP(wavelength_nm,n_host,width,pitch,epsilon_av,delta,polarization)
        T =100*(1-fun.find_amplitudes_LCP_RCP(wavelength_nm,n_host,width,pitch,epsilon_av,delta,polarization))


        self.ax_R.clear()
        self.ax_T.clear()


        self.ax_R.plot(wavelength_nm, R)
        self.ax_R.set_title("Reflectance")
        self.ax_R.set_xlabel("Wavelength (nm)")
        self.ax_R.set_ylabel("Reflectance (%)")
        self.ax_R.grid(True)
        self.ax_R.set_ylim(0, 100)


        self.ax_T.plot(wavelength_nm, T)
        self.ax_T.set_title("Transmittance")
        self.ax_T.set_xlabel("Wavelength (nm)")
        self.ax_T.set_ylabel("Transmittance (%)")
        self.ax_T.grid(True)
        self.ax_T.set_ylim(0, 100)


        self.canvas.draw()

# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------


if __name__ == "__main__":
    app = OpticalGUI()
    app.mainloop()
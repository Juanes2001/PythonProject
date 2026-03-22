import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import serial
import serial.tools.list_ports
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SerialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Arduino Nano Data Sweeper & Monitor")
        self.root.geometry("1200x700")

        # Serial & Data Variables
        self.ser = None
        self.running = False
        self.curves = []
        self.current_curve_data = []
        self.current_base_current = ""

        self.setup_ui()

    def setup_ui(self):
        # --- TOP: Control Panel ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Port:").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(control_frame, postcommand=self.refresh_ports)
        self.port_combo.pack(side=tk.LEFT, padx=5)
        self.refresh_ports()

        ttk.Button(control_frame, text="↻ Refresh", width=10, command=self.refresh_ports).pack(side=tk.LEFT, padx=5)
        self.btn_connect = ttk.Button(control_frame, text="Connect", command=self.toggle_connection)
        self.btn_connect.pack(side=tk.LEFT, padx=5)

        self.btn_sweep = ttk.Button(control_frame, text="Do Sweep", command=self.send_sweep_command, state=tk.DISABLED)
        self.btn_sweep.pack(side=tk.LEFT, padx=5)

        # --- MAIN CONTENT: PanedWindow (The Draggable Slider) ---
        # This allows the user to slide the divider between the graph and the text box
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Side: Graph Frame
        graph_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(graph_frame, weight=3)  # Weight 3 means it starts larger

        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_title("Sweep Visualization")
        self.ax.set_xlabel("X Value")
        self.ax.set_ylabel("Y Value")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right Side: Command/Menu Cluster Frame
        right_frame = ttk.LabelFrame(self.paned_window, text="Arduino Messages / Menu")
        self.paned_window.add(right_frame, weight=1)  # Weight 1 means it starts smaller

        self.console = scrolledtext.ScrolledText(right_frame, width=30, font=("Consolas", 10), bg="#f8f9fa")
        self.console.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(right_frame, text="Clear Console", command=lambda: self.console.delete('1.0', tk.END)).pack(pady=5)

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports and not self.port_combo.get():
            self.port_combo.current(0)

    def log_to_console(self, text):
        """Thread-safe logging to the text cluster."""
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)

    def toggle_connection(self):
        if not self.ser or not self.ser.is_open:
            try:
                port = self.port_combo.get()
                if not port: return
                self.ser = serial.Serial(port, 115200, timeout=0.1)
                self.running = True
                self.btn_connect.config(text="Disconnect")
                self.btn_sweep.config(state=tk.NORMAL)

                self.thread = threading.Thread(target=self.read_serial, daemon=True)
                self.thread.start()
                self.log_to_console(f">> Connected to {port}")
            except Exception as e:
                messagebox.showerror("Error", f"Connection failed: {e}")
        else:
            self.running = False
            if self.ser: self.ser.close()
            self.btn_connect.config(text="Connect")
            self.btn_sweep.config(state=tk.DISABLED)
            self.log_to_console(">> Disconnected.")

    def send_sweep_command(self):
        if self.ser and self.ser.is_open:
            self.curves = []
            self.current_curve_data = []
            self.ax.clear()
            self.ax.set_title("Waiting for data stream...")
            self.canvas.draw()
            self.ser.write(b'Do_sweep@')
            self.log_to_console(">> Sent: DO_SWEEP")

    def read_serial(self):
        """Background thread logic."""
        while self.running:
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
                    self.log_to_console(line)
                    if not line: continue

                    # Check for end character to finish a curve
                    if line == "||":
                        if self.current_curve_data:
                            self.curves.append((self.current_base_current, self.current_curve_data))
                        self.current_curve_data = []
                        self.root.after(0, self.update_plot)
                        continue

                    # Try to parse as Data (3 items: Label X Y)
                    parts = line.split(' ')
                    if len(parts) == 3:
                        try:
                            lbl, x, y = parts[0], float(parts[1]), float(parts[2])
                            self.current_base_current = lbl + "uA"
                            self.current_curve_data.append((x, y))
                        except ValueError:
                            # Not numbers? Log as message instead
                            self.root.after(0, self.log_to_console, line)
                    else:
                        # Standard text/menu message
                        self.root.after(0, self.log_to_console, line)

                except Exception as e:
                    print(f"Serial Thread Error: {e}")

    def update_plot(self):
        """Redraw the graph using stored curve arrays."""
        self.ax.clear()
        self.ax.set_title("Sweep Results")
        self.ax.set_xlabel("Vce [V]")
        self.ax.set_ylabel("Ic [mA]")
        self.ax.grid(True)

        for label, data in self.curves:
            if data:
                x_coords, y_coords = zip(*data)
                self.ax.plot(x_coords, y_coords, label=f"Base: {label}", marker='o', markersize=2)

        if self.curves:
            self.ax.legend(loc='best')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SerialGUI(root)
    root.mainloop()
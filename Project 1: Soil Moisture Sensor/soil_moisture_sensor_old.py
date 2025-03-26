# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# import serial
# import threading
# import csv
# from datetime import datetime
# from serial.tools import list_ports
# import matplotlib.pyplot as plt
# import time

# class ADCReaderApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("ADC Monitor")

#         # Serial configuration
#         self.ser = None
#         self.running = False
#         self.serial_thread = None
#         self.log_file = None
#         self.log_file_avg = None

#         # Data collection variables
#         self.current_state = "idle"  # Possible states: idle, stabilizing, capturing
#         self.stabilization_duration = 60  # Default stabilization time (seconds)
#         self.capture_duration = 5      # Default capture time (seconds)
#         self.phase_start_time = 0
#         self.capture_samples = []
#         self.current_input = 0

#         # ADC configuration
#         self.config_ranges = []
#         self.config_file = "adc_config.csv"

#         # Create UI elements
#         self.create_widgets()
#         self.load_config()

#     def create_widgets(self):
#         # Configure grid columns to expand properly
#         self.root.columnconfigure(1, weight=1)

#         # COM Port selection
#         ttk.Label(self.root, text="COM Port:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
#         self.com_port = ttk.Combobox(self.root, values=[port.device for port in list_ports.comports()])
#         self.com_port.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

#         # Baud Rate selection
#         ttk.Label(self.root, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
#         self.baud_rate = ttk.Entry(self.root)
#         self.baud_rate.insert(0, "230400")
#         self.baud_rate.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

#         # Timing configuration
#         ttk.Label(self.root, text="Delay (s):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
#         self.stabilization_entry = ttk.Entry(self.root)
#         self.stabilization_entry.insert(0, "60")
#         self.stabilization_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

#         ttk.Label(self.root, text="Capture (s):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
#         self.capture_entry = ttk.Entry(self.root)
#         self.capture_entry.insert(0, "5")
#         self.capture_entry.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

#         # Start/Stop button
#         self.start_btn = ttk.Button(self.root, text="Start", command=self.toggle_serial)
#         self.start_btn.grid(row=4, column=0, columnspan=2, pady=5)

#         # Display area
#         self.display = tk.Text(self.root, height=15, width=50, state=tk.DISABLED)
#         self.display.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

#         # Input control
#         input_frame = ttk.Frame(self.root)
#         input_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
#         ttk.Label(input_frame, text="Input Step (ml):").grid(row=0, column=0, padx=5, sticky='w')
#         self.input_step_entry = ttk.Entry(input_frame, width=5)
#         self.input_step_entry.insert(0, "1")
#         self.input_step_entry.grid(row=0, column=1, padx=5, sticky='w')
#         self.current_input_label = ttk.Label(input_frame, text="Current Input: 0 ml")
#         self.current_input_label.grid(row=0, column=2, padx=10, sticky='w')
#         self.increment_btn = ttk.Button(input_frame, text="Increment", command=self.increment_input, state=tk.DISABLED)
#         self.increment_btn.grid(row=0, column=3, padx=5, sticky='e')

#         # File selection
#         ttk.Label(self.root, text="Select CSV:").grid(row=7, column=0, padx=5, pady=5)
#         self.file_path_entry = ttk.Entry(self.root, state="readonly", width=40)
#         self.file_path_entry.grid(row=7, column=1, padx=5, pady=5)
#         self.select_file_btn = ttk.Button(self.root, text="Browse", command=self.select_file)
#         self.select_file_btn.grid(row=7, column=2, padx=5, pady=5)

#         # Plot controls
#         plot_frame = ttk.Frame(self.root)
#         plot_frame.grid(row=8, column=0, columnspan=3, pady=5)
#         self.plot_type = tk.StringVar(value="raw")
#         ttk.Radiobutton(plot_frame, text="Raw Data", variable=self.plot_type, value="raw").grid(row=0, column=0, padx=5)
#         ttk.Radiobutton(plot_frame, text="Averaged Data", variable=self.plot_type, value="avg").grid(row=0, column=1, padx=5)
#         self.plot_btn = ttk.Button(plot_frame, text="Plot Data", command=self.plot_data)
#         self.plot_btn.grid(row=0, column=2, padx=5)

#     def load_config(self):
#         try:
#             with open(self.config_file, 'r') as file:
#                 reader = csv.reader(file)
#                 next(reader)
#                 for row in reader:
#                     if len(row) >= 2:
#                         low, high = map(int, row[0].split('-'))
#                         self.config_ranges.append({'low': low, 'high': high, 'condition': row[1]})
#                 self.config_ranges.sort(key=lambda x: x['low'])
#             self.append_display("Configuration loaded")
#         except Exception as e:
#             self.append_display(f"Config error: {str(e)}")

#     def select_file(self):
#         file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#         if file_path:
#             self.file_path_entry.config(state="normal")
#             self.file_path_entry.delete(0, tk.END)
#             self.file_path_entry.insert(0, file_path)
#             self.file_path_entry.config(state="readonly")
#             self.selected_file = file_path

#     def plot_data(self):
#         if not hasattr(self, 'selected_file') or not self.selected_file:
#             messagebox.showwarning("Warning", "Select a CSV file first.")
#             return
#         try:
#             plot_type = self.plot_type.get()
#             if plot_type == "raw":
#                 times, adc_values = [], []
#                 with open(self.selected_file, 'r') as file:
#                     reader = csv.reader(file)
#                     header = next(reader)
#                     time_idx = header.index('time')
#                     adc_idx = header.index('adc_value')
#                     for row in reader:
#                         times.append(row[time_idx])
#                         adc_values.append(int(row[adc_idx]))
#                 plt.figure()
#                 plt.plot(times, adc_values, 'b-', label="ADC Value")
#                 plt.xticks(rotation=45)
#                 plt.xlabel("Time")
#                 plt.ylabel("Value")
#                 plt.title("Raw ADC Data")
#             elif plot_type == "avg":
#                 inputs, averages = [], []
#                 with open(self.selected_file, 'r') as file:
#                     reader = csv.reader(file)
#                     header = next(reader)
#                     input_idx = header.index('input')
#                     adc_idx = header.index('adc_value')
#                     for row in reader:
#                         inputs.append(int(row[input_idx]))
#                         averages.append(float(row[adc_idx]))
#                 plt.figure()
#                 plt.plot(inputs, averages, 'ro-', label="Averaged ADC")
#                 plt.xlabel("Input (ml)")
#                 plt.ylabel("Value")
#                 plt.title("Averaged ADC per Input")
#             plt.grid()
#             plt.legend()
#             plt.show()
#         except Exception as e:
#             messagebox.showerror("Error", f"Plotting failed: {str(e)}")

#     def toggle_serial(self):
#         if self.running:
#             self.running = False
#             self.start_btn.config(text="Start")
#             self.increment_btn.config(state=tk.DISABLED)
#             if self.ser:
#                 self.ser.close()
#             if self.log_file:
#                 self.log_file.close()
#             if self.log_file_avg:
#                 self.log_file_avg.close()
#             if hasattr(self, 'stabilization_timer'):
#                 self.stabilization_timer.cancel()
#             if hasattr(self, 'capture_timer'):
#                 self.capture_timer.cancel()
#             self.current_state = 'idle'
#             self.append_display("Stopped")
#         else:
#             try:
#                 time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 self.log_file = open(f"adc_log_{time_str}.csv", 'w', newline='')
#                 csv.writer(self.log_file).writerow(['time', 'adc_value', 'condition'])
#                 self.log_file_avg = open(f"adc_log_avg_{time_str}.csv", 'w', newline='')
#                 csv.writer(self.log_file_avg).writerow(['input', 'adc_value'])
                
#                 self.ser = serial.Serial(
#                     port=self.com_port.get(),
#                     baudrate=int(self.baud_rate.get()),
#                     timeout=1
#                 )
#                 self.running = True
#                 self.current_input = 0
#                 self.update_current_input_label()
#                 self.start_btn.config(text="Stop")
#                 self.current_state = 'waiting'
#                 self.increment_btn.config(state=tk.DISABLED)
                
#                 stabilization_time = int(self.stabilization_entry.get())
#                 self.stabilization_timer = threading.Timer(stabilization_time, self.start_capture)
#                 self.stabilization_timer.start()
#                 self.append_display(f"Stabilizing for {stabilization_time} seconds...")
                
#                 self.serial_thread = threading.Thread(target=self.read_serial)
#                 self.serial_thread.daemon = True
#                 self.serial_thread.start()
#             except Exception as e:
#                 self.append_display(f"Error: {str(e)}")
#                 if self.log_file:
#                     self.log_file.close()
#                 if self.log_file_avg:
#                     self.log_file_avg.close()

#     def start_capture(self):
#         self.root.after(0, self.append_display, "Capturing data for 5 seconds...")
#         self.current_state = 'capturing'
#         with self.lock:
#             self.adc_samples = []
#         self.capture_timer = threading.Timer(self.capture_duration, self.finish_capture)
#         self.capture_timer.start()

#     def finish_capture(self):
#         with self.lock:
#             samples = self.adc_samples.copy()
#         if samples:
#             avg = sum(samples) / len(samples)
#             self.root.after(0, self._finish_capture_gui, avg)
#         else:
#             self.root.after(0, self.append_display, "No data captured during the 5-second window.")
#             self.root.after(0, self._finish_capture_gui, None)

#     def _finish_capture_gui(self, avg):
#         if avg is not None:
#             csv.writer(self.log_file_avg).writerow([self.current_input, avg])
#             self.log_file_avg.flush()
#             self.append_display(f"Average ADC for {self.current_input}ml: {avg:.2f}")
#         self.current_state = 'ready'
#         self.increment_btn.config(state=tk.NORMAL)
#         self.append_display("Ready for next increment.")

#     def read_serial(self):
#         while self.running:
#             try:
#                 data = self.ser.readline().decode().strip()
#                 if "Raw ADC Value:" in data:
#                     adc_value = int(data.split(":")[1])
#                     condition = self.get_condition(adc_value)
#                     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     self.root.after(0, self.append_data, adc_value, condition)
                    
#                     if self.current_state == 'capturing':
#                         with self.lock:
#                             self.adc_samples.append(adc_value)
#                         self.root.after(0, lambda: csv.writer(self.log_file).writerow(
#                             [timestamp, adc_value, condition]))
#                         self.log_file.flush()
#             except Exception as e:
#                 if self.running:
#                     self.append_display(f"Serial error: {str(e)}")
#                 self.running = False
#                 self.start_btn.config(text="Start")

#     def increment_input(self):
#         if self.current_state != 'ready':
#             return
#         try:
#             step = int(self.input_step_entry.get())
#         except ValueError:
#             messagebox.showerror("Error", "Invalid step value.")
#             return
        
#         self.current_input += step
#         self.update_current_input_label()
#         self.current_state = 'waiting'
#         self.increment_btn.config(state=tk.DISABLED)
        
#         stabilization_time = int(self.stabilization_entry.get())
#         self.stabilization_timer = threading.Timer(stabilization_time, self.start_capture)
#         self.stabilization_timer.start()
#         self.append_display(f"Stabilizing for {stabilization_time} seconds...")

#     def update_current_input_label(self):
#         self.current_input_label.config(text=f"Current Input: {self.current_input} ml")

#     def get_condition(self, value):
#         for r in self.config_ranges:
#             if r['low'] <= value <= r['high']:
#                 return r['condition']
#         return "Unknown"

#     def append_display(self, message):
#         self.display.config(state=tk.NORMAL)
#         self.display.insert(tk.END, message + "\n")
#         self.display.see(tk.END)
#         self.display.config(state=tk.DISABLED)

#     def append_data(self, value, condition):
#         self.display.config(state=tk.NORMAL)
#         self.display.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ADC: {value}, Condition: {condition}\n")
#         self.display.see(tk.END)
#         self.display.config(state=tk.DISABLED)

#     def on_closing(self):
#         self.running = False
#         if self.ser:
#             self.ser.close()
#         if self.log_file:
#             self.log_file.close()
#         if self.log_file_avg:
#             self.log_file_avg.close()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ADCReaderApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import threading
import csv
from datetime import datetime
from serial.tools import list_ports
import matplotlib.pyplot as plt
import winsound


class soil_moisture_sensor:
    def __init__(self, root):
        self.root = root
        self.root.title("ADC Monitor")

        # Serial configuration
        self.ser = None
        self.running = False
        self.serial_thread = None
        self.log_file = None
        self.log_file_avg = None

        # Data collection parameters
        self.stabilization_time = 60  # Default stabilization time in seconds
        self.capture_duration = 5     # Fixed 5-second capture window
        self.current_state = 'idle'   # Possible states: idle, waiting, capturing, ready

        # Historical data and input tracking
        self.history = []
        self.current_input = 0
        self.adc_samples = []
        self.lock = threading.Lock()

        # ADC configuration
        self.config_ranges = []
        self.config_file = "adc_config.csv"

        # Create UI elements first
        self.create_widgets()

        # Load configuration after UI elements are created
        self.load_config()

    def create_widgets(self):
        # Configure grid columns to expand properly
        self.root.columnconfigure(1, weight=1)

        # COM Port selection
        ttk.Label(self.root, text="COM Port:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.com_port = ttk.Combobox(self.root, values=[port.device for port in list_ports.comports()])
        self.com_port.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Baud Rate selection
        ttk.Label(self.root, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.baud_rate = ttk.Entry(self.root)
        self.baud_rate.insert(0, "230400")
        self.baud_rate.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Stabilization Time
        ttk.Label(self.root, text="Stabilization (s):").grid(row=2, column=0, padx=5, sticky='w')
        self.stabilization_entry = ttk.Entry(self.root, width=8)
        self.stabilization_entry.insert(0, "60")
        self.stabilization_entry.grid(row=2, column=1, padx=5, sticky='w')

        # Capture Duration (now in new row)
        ttk.Label(self.root, text="Capture (s):").grid(row=3, column=0, padx=5, sticky='w')
        self.capture_entry = ttk.Entry(self.root, width=8)
        self.capture_entry.insert(0, "5")
        self.capture_entry.grid(row=3, column=1, padx=5, sticky='w')

        # Start/Stop button (now row 4, centered)
        self.start_btn = ttk.Button(self.root, text="Start", command=self.toggle_serial)
        self.start_btn.grid(row=4, column=0, columnspan=2, pady=5)

        # Display area (shifted to row 5)
        self.display = tk.Text(self.root, height=15, width=50, state=tk.DISABLED)
        self.display.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

        # File selection (row 6)
        ttk.Label(self.root, text="Select CSV:").grid(row=6, column=0, padx=5, pady=5)
        self.file_path_entry = ttk.Entry(self.root, state="readonly", width=40)
        self.file_path_entry.grid(row=6, column=1, padx=5, pady=5)
        self.select_file_btn = ttk.Button(self.root, text="Browse", command=self.select_file)
        self.select_file_btn.grid(row=6, column=2, padx=5, pady=5)

        # Input control (row 7)
        input_frame = ttk.Frame(self.root)
        input_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        ttk.Label(input_frame, text="Input Step (ml):").grid(row=0, column=0, padx=5, sticky='w')
        self.input_step_entry = ttk.Entry(input_frame, width=5)
        self.input_step_entry.insert(0, "1")
        self.input_step_entry.grid(row=0, column=1, padx=5, sticky='w')
        self.current_input_label = ttk.Label(input_frame, text="Current Input: 0 ml")
        self.current_input_label.grid(row=0, column=2, padx=10, sticky='w')
        self.increment_btn = ttk.Button(input_frame, text="Increment", command=self.increment_input, state=tk.DISABLED)
        self.increment_btn.grid(row=0, column=3, padx=5, sticky='e')

        # Plot controls (row 8)
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=8, column=0, columnspan=3, pady=5)
        self.plot_type = tk.StringVar(value="raw")
        ttk.Radiobutton(plot_frame, text="Raw Data", variable=self.plot_type, value="raw").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(plot_frame, text="Averaged Data", variable=self.plot_type, value="avg").grid(row=0, column=1, padx=5)
        self.plot_btn = ttk.Button(plot_frame, text="Plot Data", command=self.plot_data)
        self.plot_btn.grid(row=0, column=2, padx=5)


    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        low, high = map(int, row[0].split('-'))
                        self.config_ranges.append({'low': low, 'high': high, 'condition': row[1]})
                self.config_ranges.sort(key=lambda x: x['low'])
            self.append_display("Configuration loaded")
        except Exception as e:
            self.append_display(f"Config error: {str(e)}")

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path_entry.config(state="normal")
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            self.file_path_entry.config(state="readonly")
            self.selected_file = file_path

    def plot_data(self):
        if not hasattr(self, 'selected_file') or not self.selected_file:
            messagebox.showwarning("Warning", "Select a CSV file first.")
            return
        try:
            plot_type = self.plot_type.get()
            if plot_type == "raw":
                times, adc_values = [], []
                with open(self.selected_file, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    time_idx = header.index('time')
                    adc_idx = header.index('adc_value')
                    for row in reader:
                        times.append(row[time_idx])
                        adc_values.append(int(row[adc_idx]))
                plt.figure()
                plt.plot(times, adc_values, 'b-', label="ADC Value")
                plt.xticks(rotation=45)
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Raw ADC Data")
            elif plot_type == "avg":
                inputs, averages = [], []
                with open(self.selected_file, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    input_idx = header.index('input')
                    adc_idx = header.index('adc_value')
                    for row in reader:
                        inputs.append(int(row[input_idx]))
                        averages.append(float(row[adc_idx]))
                plt.figure()
                plt.plot(inputs, averages, 'ro-', label="Averaged ADC")
                plt.xlabel("Input (ml)")
                plt.ylabel("Value")
                plt.title("Averaged ADC per Input")
            plt.grid()
            plt.legend()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")

    def toggle_serial(self):
        if self.running:
            self.running = False
            self.start_btn.config(text="Start")
            self.increment_btn.config(state=tk.DISABLED)
            if self.ser:
                self.ser.close()
            if self.log_file:
                self.log_file.close()
            if self.log_file_avg:
                self.log_file_avg.close()
            if hasattr(self, 'stabilization_timer'):
                self.stabilization_timer.cancel()
            if hasattr(self, 'capture_timer'):
                self.capture_timer.cancel()
            self.current_state = 'idle'
            self.append_display("Stopped")
        else:
            try:
                # Get timing parameters from UI
                self.stabilization_time = int(self.stabilization_entry.get())
                self.capture_duration = int(self.capture_entry.get())
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = open(f"adc_log_{time_str}.csv", 'w', newline='')
                csv.writer(self.log_file).writerow(['time', 'adc_value', 'condition'])
                self.log_file_avg = open(f"adc_log_avg_{time_str}.csv", 'w', newline='')
                csv.writer(self.log_file_avg).writerow(['input', 'adc_value'])
                
                self.ser = serial.Serial(
                    port=self.com_port.get(),
                    baudrate=int(self.baud_rate.get()),
                    timeout=1
                )
                self.running = True
                self.current_input = 0
                self.update_current_input_label()
                self.start_btn.config(text="Stop")
                self.current_state = 'waiting'
                self.increment_btn.config(state=tk.DISABLED)
                
                stabilization_time = int(self.stabilization_entry.get())
                self.stabilization_timer = threading.Timer(stabilization_time, self.start_capture)
                self.stabilization_timer.start()
                self.append_display(f"Stabilizing for {stabilization_time} seconds...")
                
                self.serial_thread = threading.Thread(target=self.read_serial)
                self.serial_thread.daemon = True
                self.serial_thread.start()
            except Exception as e:
                self.append_display(f"Error: {str(e)}")
                if self.log_file:
                    self.log_file.close()
                if self.log_file_avg:
                    self.log_file_avg.close()

    def start_capture(self):
        self.root.after(0, self.append_display, 
                      f"Capturing data for {self.capture_duration} seconds...")
        self.current_state = 'capturing'
        with self.lock:
            self.adc_samples = []
        self.capture_timer = threading.Timer(self.capture_duration, self.finish_capture)
        self.capture_timer.start()

    def finish_capture(self):
        with self.lock:
            samples = self.adc_samples.copy()
        if samples:
            avg = sum(samples) / len(samples)
            self.root.after(0, self._finish_capture_gui, avg)
        else:
            self.root.after(0, self.append_display, "No data captured during the 5-second window.")
            self.root.after(0, self._finish_capture_gui, None)

    def _finish_capture_gui(self, avg):
        if avg is not None:
            csv.writer(self.log_file_avg).writerow([self.current_input, avg])
            self.log_file_avg.flush()
            self.append_display(f"Average ADC for {self.current_input}ml: {avg:.2f}")
        self.current_state = 'ready'
        # Frequency (Hz) and Duration (ms)
        winsound.Beep(1000, 1000)  
        self.increment_btn.config(state=tk.NORMAL)
        self.append_display("Ready for next increment.")

    def read_serial(self):
        while self.running:
            try:
                data = self.ser.readline().decode().strip()
                if "Raw ADC Value:" in data:
                    adc_value = int(data.split(":")[1])
                    condition = self.get_condition(adc_value)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.root.after(0, self.append_data, adc_value, condition)
                    
                    if self.current_state == 'capturing':
                        with self.lock:
                            self.adc_samples.append(adc_value)
                        self.root.after(0, lambda: csv.writer(self.log_file).writerow(
                            [timestamp, adc_value, condition]))
                        self.log_file.flush()
            except Exception as e:
                if self.running:
                    self.append_display(f"Serial error: {str(e)}")
                self.running = False
                self.start_btn.config(text="Start")

    def increment_input(self):
        if self.current_state != 'ready':
            return
        try:
            step = int(self.input_step_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid step value.")
            return
        
        self.current_input += step
        self.update_current_input_label()
        self.current_state = 'waiting'
        self.increment_btn.config(state=tk.DISABLED)
        
        stabilization_time = int(self.stabilization_entry.get())
        self.stabilization_timer = threading.Timer(stabilization_time, self.start_capture)
        self.stabilization_timer.start()
        self.append_display(f"Stabilizing for {stabilization_time} seconds...")

    def update_current_input_label(self):
        self.current_input_label.config(text=f"Current Input: {self.current_input} ml")

    def get_condition(self, value):
        for r in self.config_ranges:
            if r['low'] <= value <= r['high']:
                return r['condition']
        return "Unknown"

    def append_display(self, message):
        self.display.config(state=tk.NORMAL)
        self.display.insert(tk.END, message + "\n")
        self.display.see(tk.END)
        self.display.config(state=tk.DISABLED)

    def append_data(self, value, condition):
        self.display.config(state=tk.NORMAL)
        self.display.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ADC: {value}, Condition: {condition}\n")
        self.display.see(tk.END)
        self.display.config(state=tk.DISABLED)

    def on_closing(self):
        self.running = False
        if self.ser:
            self.ser.close()
        if self.log_file:
            self.log_file.close()
        if self.log_file_avg:
            self.log_file_avg.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = soil_moisture_sensor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import re
import csv
from datetime import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import medfilt
import numpy as np
import time
import math
import pandas as pd

class SerialADCCollector:
    def __init__(self):
        # Initialize root window first
        self.root = tk.Tk()
        self.root.title("ADC Data Collector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Now create Tkinter variables after root exists
        self.high_ratio_ch1_var = tk.DoubleVar(self.root, value=0.6)
        self.low_ratio_ch1_var = tk.DoubleVar(self.root, value=0.4)
        self.high_ratio_ch2_var = tk.DoubleVar(self.root, value=0.6)
        self.low_ratio_ch2_var = tk.DoubleVar(self.root, value=0.4)
        self.duration_var = tk.StringVar(self.root)
        self.filter_type_var = tk.StringVar(self.root, value="Median")
        self.filter_size_ch1_var = tk.IntVar(self.root, value=5)
        self.filter_size_ch2_var = tk.IntVar(self.root, value=5)

        self.ser = None
        self.running_event = threading.Event()
        self.data_lock = threading.Lock()
        self.com_port = None
        
        # Data storage
        self.baseline_data = []
        self.main_data = []
        self.filtered_ch1 = []
        self.filtered_ch2 = []
        self.quantized_ch1 = []
        self.quantized_ch2 = []

        # Data storage
        self.quantized_filename = None  # Add to track latest file

        # GUI variables
        self.breath_label = None  # Will be created in create_widgets
        
        # GUI variables
        self.high_ratio_ch1_var = tk.DoubleVar(value=0.6)
        self.low_ratio_ch1_var = tk.DoubleVar(value=0.4)
        self.high_ratio_ch2_var = tk.DoubleVar(value=0.6)
        self.low_ratio_ch2_var = tk.DoubleVar(value=0.4)
        self.selected_duration = 0
        self.state = "idle"

        # Auto-detect COM port
        self.find_com_port()
        if not self.com_port:
            self.show_error("No compatible device found!")
            return

        self.create_widgets()
        self.root.after(100, self.update_sample_count)
        self.root.mainloop()

    def find_com_port(self):
        ports = serial.tools.list_ports.comports()
        pattern = re.compile(r"Channel 1: (\d+), Channel 2: (\d+)")
        
        for port in ports:
            try:
                with serial.Serial(port.device, baudrate=230400, timeout=2) as ser:
                    print(f"Testing {port.device}...")
                    start_time = time.time()
                    while time.time() - start_time < 2:
                        line = ser.readline().decode(errors='ignore').strip()
                        if line and pattern.match(line):
                            print(f"Found device on {port.device}")
                            self.com_port = port.device
                            return
            except Exception as e:
                print(f"Error testing {port.device}: {e}")

    def create_widgets(self):
        # Control buttons
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(
            self.btn_frame, text="Start", command=self.start_collecting)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.duration_var = tk.StringVar()
        self.duration_combobox = ttk.Combobox(
            self.btn_frame, textvariable=self.duration_var,
            values=["52 seconds resting", "32 seconds light exercise"],
            state="readonly", width=20
        )
        self.duration_combobox.pack(side=tk.LEFT, padx=5)
        self.duration_combobox.current(0)

        self.plot_btn = ttk.Button(
            self.btn_frame, text="Plot", command=self.plot_data, state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=5)

        # Add Calculate BPM button next to Plot button
        self.calculate_btn = ttk.Button(
            self.btn_frame, text="Calculate BPM", 
            command=self.calculate_breathing_rate, state=tk.DISABLED
        )
        self.calculate_btn.pack(side=tk.LEFT, padx=5)

        # Add label for displaying breathing rate
        self.breath_label = ttk.Label(
            self.root, text="Breathing Rate: --", 
            font=("Arial", 14, "bold")
        )
        self.breath_label.pack(pady=10)

        # Status and counters
        self.status_label = ttk.Label(self.root, text="Status: Idle")
        self.status_label.pack(pady=5)
        self.count_label = ttk.Label(self.root, text="Samples collected: 0")
        self.count_label.pack(pady=5)

        # Filter controls
        filter_frame = ttk.LabelFrame(self.root, text="Denoising Filter")
        filter_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Label(filter_frame, text="Filter Type:").pack(side=tk.LEFT, padx=5)
        self.filter_type_var = tk.StringVar(value="Median")
        ttk.Combobox(filter_frame, textvariable=self.filter_type_var,
                    values=["None", "Median"], state="readonly", width=15).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Ch1 Kernel Size:").pack(side=tk.LEFT, padx=5)
        self.filter_size_ch1_var = tk.IntVar(value=5)
        ttk.Spinbox(filter_frame, from_=3, to=51, increment=2,
                   textvariable=self.filter_size_ch1_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_frame, text="Ch2 Kernel Size:").pack(side=tk.LEFT, padx=5)
        self.filter_size_ch2_var = tk.IntVar(value=5)
        ttk.Spinbox(filter_frame, from_=3, to=51, increment=2,
                   textvariable=self.filter_size_ch2_var, width=5).pack(side=tk.LEFT, padx=5)

        # Hysteresis thresholds
        hyst_frame = ttk.LabelFrame(self.root, text="Hysteresis Quantization Thresholds")
        hyst_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Label(hyst_frame, text="Ch1 High:").grid(row=0, column=0, padx=5)
        ttk.Spinbox(hyst_frame, from_=0.0, to=1.0, increment=0.01,
                   textvariable=self.high_ratio_ch1_var, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(hyst_frame, text="Ch1 Low:").grid(row=1, column=0, padx=5)
        ttk.Spinbox(hyst_frame, from_=0.0, to=1.0, increment=0.01,
                   textvariable=self.low_ratio_ch1_var, width=6).grid(row=1, column=1, padx=5)

        ttk.Label(hyst_frame, text="Ch2 High:").grid(row=0, column=2, padx=5)
        ttk.Spinbox(hyst_frame, from_=0.0, to=1.0, increment=0.01,
                   textvariable=self.high_ratio_ch2_var, width=6).grid(row=0, column=3, padx=5)
        ttk.Label(hyst_frame, text="Ch2 Low:").grid(row=1, column=2, padx=5)
        ttk.Spinbox(hyst_frame, from_=0.0, to=1.0, increment=0.01,
                   textvariable=self.low_ratio_ch2_var, width=6).grid(row=1, column=3, padx=5)

    def start_collecting(self):
        # Get selected duration
        duration_str = self.duration_var.get()
        self.selected_duration = 52 if "52" in duration_str else 32
        self.selected_duration -= 2

        try:
            self.ser = serial.Serial(self.com_port, baudrate=230400, timeout=0.1)
        except Exception as e:
            self.show_error(f"Failed to open port: {e}")
            return

        self.running_event.set()
        self.start_btn.config(state=tk.DISABLED)
        self.duration_combobox.config(state=tk.DISABLED)
        self.plot_btn.config(state=tk.DISABLED)
        self.update_status("Collecting baseline...")

        self.baseline_data = []
        self.main_data = []
        self.state = "baseline"

        # Start baseline collection timer
        self.baseline_timer = threading.Timer(2.0, self.end_baseline)
        self.baseline_timer.start()

        self.thread = threading.Thread(target=self.read_serial)
        self.thread.start()

    def end_baseline(self):
        if not self.baseline_data:
            self.show_error("Baseline collection failed!")
            self.stop_collecting()
            return

        self.baseline_mean = sum(self.baseline_data) / len(self.baseline_data)
        self.threshold_high = self.baseline_mean * 1.02
        self.threshold_low = self.baseline_mean * 0.98
        self.state = "ready"
        self.update_status("Ready. Waiting for trigger...")

    def read_serial(self):
        pattern = re.compile(r"Channel 1: (\d+), Channel 2: (\d+)")
        while self.running_event.is_set():
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    match = pattern.match(line)
                    if match:
                        ch1 = int(match.group(1))
                        ch2 = int(match.group(2))
                        
                        if self.state == "baseline":
                            with self.data_lock:
                                self.baseline_data.append(ch1)
                        elif self.state == "ready":
                            if ch1 > self.threshold_high or ch1 < self.threshold_low:
                                self.state = "collecting"
                                self.collection_start_time = time.time()
                                self.root.after(0, self.update_countdown)
                                # self.update_status("Collecting...")
                        elif self.state == "collecting":
                            elapsed = time.time() - self.collection_start_time
                            if elapsed >= self.selected_duration:
                                self.stop_collecting()
                            else:
                                with self.data_lock:
                                    self.main_data.append((ch1, ch2))
            except Exception as e:
                print(f"Serial read error: {e}")

    def update_countdown(self):
        if self.state != "collecting":
            return
        elapsed = time.time() - self.collection_start_time
        remaining = self.selected_duration - elapsed
        if remaining <= 0:
            self.stop_collecting()
        else:
            self.status_label.config(text=f"Status: Collecting... Time remaining: {int(remaining)}s")
            self.root.after(1000, self.update_countdown)

    def stop_collecting(self):
        self.running_event.clear()
        if self.thread.is_alive():
            self.thread.join()
        if self.ser:
            self.ser.close()
        self.start_btn.config(state=tk.NORMAL)
        self.duration_combobox.config(state=tk.NORMAL)
        self.plot_btn.config(state=tk.NORMAL)
        # Enable calculate button after collection
        self.calculate_btn.config(state=tk.NORMAL)
        self.save_to_csv()
        self.save_filtered_to_csv()
        self.update_status("Finished")

    def save_to_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adc_data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Channel 1", "Channel 2"])
            with self.data_lock:
                writer.writerows(self.main_data)
        print(f"Data saved to {filename}")

    def apply_median_filter(self, data, kernel_size):
        if len(data) < kernel_size:
            return data
        return medfilt(data, kernel_size=kernel_size if kernel_size%2 else kernel_size+1)

    def save_filtered_to_csv(self):
        with self.data_lock:
            if not self.main_data:
                return
            ch1 = [x[0] for x in self.main_data]
            ch2 = [x[1] for x in self.main_data]

        filter_type = self.filter_type_var.get()
        kernel_size_ch1 = self.filter_size_ch1_var.get()
        kernel_size_ch2 = self.filter_size_ch2_var.get()

        if filter_type == "Median":
            self.filtered_ch1 = self.apply_median_filter(ch1, kernel_size_ch1)
            self.filtered_ch2 = self.apply_median_filter(ch2, kernel_size_ch2)
        else:
            self.filtered_ch1 = ch1
            self.filtered_ch2 = ch2

        # Apply quantization
        self.quantized_ch1 = 1 - self.hysteresis_quantization(
            self.filtered_ch1, 
            self.high_ratio_ch1_var.get(),
            self.low_ratio_ch1_var.get()
        )
        self.quantized_ch2 = self.hysteresis_quantization(
            self.filtered_ch2,
            self.high_ratio_ch2_var.get(),
            self.low_ratio_ch2_var.get()
        )

        self.quantized_ch1[0] = 0
        self.quantized_ch2[0] = 0

        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"adc_filtered_data_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filtered Ch1", "Filtered Ch2"])  # Optional header
            writer.writerows(zip(self.filtered_ch1, self.filtered_ch2))

        self.quantized_filename = f"adc_quantized_breath_{timestamp}.csv"
        with open(self.quantized_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Quantized Ch1", "Quantized Ch2"])  # Corrected headers
            writer.writerows(zip(self.quantized_ch1, self.quantized_ch2))

    def hysteresis_quantization(self, signal, high_ratio, low_ratio):
        signal = np.array(signal)
        min_val, max_val = np.min(signal), np.max(signal)
        # min_val, max_val = self.baseline_mean, np.max(signal)
        
        high_thresh = min_val + high_ratio * (max_val - min_val)
        low_thresh = min_val + low_ratio * (max_val - min_val)
        
        quantized = np.zeros_like(signal)
        state = False
        for i in range(len(signal)):
            if not state and signal[i] > high_thresh:
                state = True
            elif state and signal[i] < low_thresh:
                state = False
            quantized[i] = int(state)
        return quantized
    
    def calculate_breathing_rate(self):
        if not self.quantized_filename:
            self.show_error("No data available. Collect data first.")
            return
        try:
            df = pd.read_csv(self.quantized_filename)
            ch1 = df['Quantized Ch1'].values
            ch2 = df['Quantized Ch2'].values

            sample_rate = 100  # Assuming 100Hz sampling rate

            # def calculate_rate(signal):
            #     signal = np.array(signal)
            #     binary = (signal > 0.5).astype(int)
            #     transitions = np.sum(binary[1:] != binary[:-1])
            #     breaths = transitions / 2
            #     duration_sec = ((len(signal)/sample_rate)/10)*10
            #     print(f"Duration in seconds: {duration_sec}")
            #     return (breaths / duration_sec) * 60 if duration_sec > 0 else 0
            
            def calculate_rate(signal):
                try:
                    signal = np.array(signal)
                    binary = (signal > 0.5).astype(int)
                    
                    # Count ANY transition (both 0->1 and 1->0)
                    transitions = np.sum((binary[1:] == 1) & (binary[:-1] == 0))
                    
                    duration_sec = round((len(signal) / sample_rate),-1)
                    print(f"Duration in seconds: {duration_sec}")
                    return (transitions / duration_sec) * 60 if duration_sec > 0 else 0
                except Exception as e:
                    print(f"Error in calculate_rate: {e}")
                    return 0

            rate1 = calculate_rate(ch1)
            rate2 = calculate_rate(ch2)
            fused_rate = (round(rate1 * 0.7 + rate2 * 0.3))

            # Update GUI
            self.breath_label.config(
                text=f"Breathing Rate: {fused_rate:.1f} BPM"
            )

            # Console output
            print(f"Breathing Rate (Channel 1): {rate1:.1f} BPM")
            print(f"Breathing Rate (Channel 2): {rate2:.1f} BPM")
            print(f"Fused Breathing Rate: {int(fused_rate)} BPM")

        except Exception as e:
            self.show_error(f"Calculation error: {str(e)}")

    def plot_data(self):
        with self.data_lock:
            if not self.main_data:
                return
            ch1 = [x[0] for x in self.main_data]
            ch2 = [x[1] for x in self.main_data]

        plt.figure(figsize=(10, 8))
        
        # Raw vs Filtered
        plt.subplot(3, 1, 1)
        plt.plot(ch1, 'b-', alpha=0.5, label='Raw Ch1')
        plt.plot(self.filtered_ch1, 'g-', label='Filtered Ch1')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(ch2, 'r-', alpha=0.5, label='Raw Ch2')
        plt.plot(self.filtered_ch2, 'm-', label='Filtered Ch2')
        plt.legend()
        plt.grid(True)

        # Quantized
        plt.subplot(3, 1, 3)
        plt.step(range(len(self.quantized_ch1)), self.quantized_ch1, 'y-', label='Quantized Ch1')
        plt.step(range(len(self.quantized_ch2)), self.quantized_ch2, 'c-', label='Quantized Ch2')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def update_sample_count(self):
        with self.data_lock:
            count = len(self.main_data)
        self.count_label.config(text=f"Samples collected: {count}")
        self.root.after(100, self.update_sample_count)

    def show_error(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        ttk.Label(error_window, text=message).pack(padx=20, pady=20)
        ttk.Button(error_window, text="OK", command=error_window.destroy).pack(pady=10)

    def on_close(self):
        if self.running_event.is_set():
            self.stop_collecting()
        self.root.destroy()

if __name__ == "__main__":
    app = SerialADCCollector()
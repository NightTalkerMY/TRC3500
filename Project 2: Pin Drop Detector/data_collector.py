import serial
import serial.tools.list_ports
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from collections import deque
import time

# Matplotlib setup for Tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SerialADCReader:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.ser = None
        self.running = False
        self.thread = None
        self.buffer = bytearray()
        self.connect()

    def connect(self):
        ports = serial.tools.list_ports.comports()
        if not ports:
            raise Exception("No COM ports found!")
        
        try:
            self.ser = serial.Serial(
                port=ports[0].device,
                baudrate=230400,
                timeout=0.001,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            raise Exception(f"Failed to open serial port: {str(e)}")

    def _read_thread(self):
        while self.running:
            try:
                data = self.ser.read(self.ser.in_waiting or 1)
                if data:
                    self.buffer.extend(data)
                    
                    while b'\n' in self.buffer:
                        line, self.buffer = self.buffer.split(b'\n', 1)
                        try:
                            value = int(line.decode('utf-8').strip())
                            self.data_queue.put(value)
                        except (UnicodeDecodeError, ValueError):
                            continue
                
            except serial.SerialException:
                break
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                break

    def start(self):
        if not self.running and self.ser:
            self.running = True
            self.thread = threading.Thread(target=self._read_thread, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.ser and self.ser.is_open:
            self.ser.close()

class ADCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("High-Speed ADC Recorder")
        self.geometry("400x300")
        
        self.data_queue = queue.Queue()
        self.reader = None
        self.data_buffer = deque(maxlen=10_000_000)
        self.is_recording = False
        self.waiting_for_trigger = False
        self.trigger_level = 0
        self.last_update = time.monotonic()

        # GUI components
        self.filename_label = ttk.Label(self, text="Filename:")
        self.filename_label.pack(pady=5)
        
        self.filename_entry = ttk.Entry(self, width=30)
        self.filename_entry.pack(pady=5)

        self.trigger_label = ttk.Label(self, text="Trigger Level:")
        self.trigger_label.pack(pady=5)
        
        self.trigger_entry = ttk.Entry(self, width=30)
        self.trigger_entry.pack(pady=5)

        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            self.button_frame, 
            text="Start", 
            command=self.start_recording
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            self.button_frame, 
            text="Stop", 
            command=self.stop_recording, 
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(self, text="Ready")
        self.status_label.pack(pady=5)
        self.sample_count_label = ttk.Label(self, text="Samples: 0")
        self.sample_count_label.pack(pady=5)

        self.after(100, self.update_ui)

    def start_recording(self):
        filename = self.filename_entry.get().strip()
        trigger_level_str = self.trigger_entry.get().strip()
        
        if not filename:
            messagebox.showerror("Error", "Please enter a filename")
            return
        if not trigger_level_str:
            messagebox.showerror("Error", "Please enter a trigger level")
            return
        
        try:
            trigger_level = int(trigger_level_str)
        except ValueError:
            messagebox.showerror("Error", "Trigger level must be an integer")
            return

        try:
            self.reader = SerialADCReader(self.data_queue)
            self.waiting_for_trigger = True
            self.is_recording = False
            self.trigger_level = trigger_level
            self.data_buffer.clear()
            
            self.reader.start()
            
            self.filename_entry.config(state='disabled')
            self.trigger_entry.config(state='disabled')
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Waiting for trigger...")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            if self.reader:
                self.reader.stop()
            self.reader = None

    def stop_recording(self):
        self.is_recording = False
        self.waiting_for_trigger = False
        if self.reader:
            self.reader.stop()
        
        filename = self.filename_entry.get().strip()
        if filename and self.data_buffer:
            try:
                df = pd.DataFrame(list(self.data_buffer), columns=['ADC_Value'])
                df.to_csv(f"./dataset/{filename}.csv", index=False)
                self.status_label.config(text=f"Saved {len(self.data_buffer)} samples to {filename}.csv")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
        
        # Reset UI
        self.filename_entry.config(state='normal')
        self.trigger_entry.config(state='normal')
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.sample_count_label.config(text=f"Samples: {len(self.data_buffer)}")

        # Plot data
        if self.data_buffer:
            self.plot_data()

    def plot_data(self):
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(list(self.data_buffer), color='blue', linewidth=0.5)
        ax.set_title("ADC Values Over Time")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("ADC Value")
        
        plot_window = tk.Toplevel(self)
        plot_window.title("ADC Data Plot")
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_ui(self):
        count = 0
        while True:
            try:
                value = self.data_queue.get_nowait()
                if self.waiting_for_trigger:
                    if value >= self.trigger_level:
                        self.waiting_for_trigger = False
                        self.is_recording = True
                        self.trigger_time = time.monotonic()
                        self.status_label.config(text="Recording...")
                        self.after(3000, self.stop_recording)
                        self.data_buffer.append(value)
                        count += 1
                elif self.is_recording:
                    self.data_buffer.append(value)
                    count += 1
            except queue.Empty:
                break
        
        if count > 0:
            self.sample_count_label.config(text=f"Samples: {len(self.data_buffer)}")
        
        self.after(50, self.update_ui)

    def on_closing(self):
        if self.is_recording or self.waiting_for_trigger:
            self.stop_recording()
        self.destroy()

if __name__ == "__main__":
    app = ADCApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
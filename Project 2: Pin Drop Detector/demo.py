# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox
# import serial
# import serial.tools.list_ports
# from threading import Thread, Event
# from queue import Queue
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

# # Import your existing machine learning functions
# # from analyzer import extract_features, load_model, predict_drop
# from extract_predict import *

# class RealTimePredictorApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("Drop Classifier")
#         master.geometry("800x600")
        
#         # Initialize serial connection variables
#         self.ser = None
#         self.is_reading = Event()
#         self.data_queue = Queue()
#         self.data_buffer = []
        
#         # Create GUI components
#         self.create_widgets()
        
#         # Start serial port detection
#         self.detect_serial_port()

#     def create_widgets(self):
#         # Control Frame
#         control_frame = ttk.Frame(self.master)
#         control_frame.pack(pady=10)

#         # Serial Port Selection
#         self.port_var = tk.StringVar()
#         ttk.Label(control_frame, text="COM Port:").grid(row=0, column=0, padx=5)
#         self.port_combobox = ttk.Combobox(control_frame, textvariable=self.port_var, state='readonly')
#         self.port_combobox.grid(row=0, column=1, padx=5)
#         ttk.Button(control_frame, text="Refresh", command=self.detect_serial_port).grid(row=0, column=2, padx=5)

#         # Buttons
#         self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_collection)
#         self.start_btn.grid(row=0, column=3, padx=5)
#         self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_collection, state=tk.DISABLED)
#         self.stop_btn.grid(row=0, column=4, padx=5)
#         self.predict_btn = ttk.Button(control_frame, text="Predict", command=self.run_prediction, state=tk.DISABLED)
#         self.predict_btn.grid(row=0, column=5, padx=5)

#         # Plot Frame
#         plot_frame = ttk.Frame(self.master)
#         plot_frame.pack(fill=tk.BOTH, expand=True)

#         # Matplotlib Figure
#         self.fig = Figure(figsize=(8, 4), dpi=100)
#         self.ax = self.fig.add_subplot(111)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

#         # Prediction Label
#         self.prediction_label = ttk.Label(self.master, text="Prediction: None", font=('Helvetica', 14))
#         self.prediction_label.pack(pady=10)

#     def detect_serial_port(self):
#         ports = [port.device for port in serial.tools.list_ports.comports()]
#         self.port_combobox['values'] = ports
#         if ports:
#             self.port_var.set(ports[0])

#     def start_collection(self):
#         port = self.port_var.get()
#         if not port:
#             messagebox.showerror("Error", "No COM port selected!")
#             return

#         try:
#             self.ser = serial.Serial(port, baudrate=230400, timeout=1)
#             self.is_reading.set()
#             self.data_buffer = []
            
#             # Start serial reading thread
#             Thread(target=self.read_serial, daemon=True).start()
            
#             # Start data processing thread
#             Thread(target=self.process_data, daemon=True).start()
            
#             self.start_btn.config(state=tk.DISABLED)
#             self.stop_btn.config(state=tk.NORMAL)
#             self.predict_btn.config(state=tk.DISABLED)
#             self.prediction_label.config(text="Prediction: None")
#             self.ax.clear()
#             self.canvas.draw()
#         except serial.SerialException as e:
#             messagebox.showerror("Error", f"Failed to open {port}: {str(e)}")

#     def stop_collection(self):
#         self.is_reading.clear()
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#         self.start_btn.config(state=tk.NORMAL)
#         self.stop_btn.config(state=tk.DISABLED)
#         self.predict_btn.config(state=tk.NORMAL)
#         self.plot_data()

#     def read_serial(self):
#         while self.is_reading.is_set() and self.ser.is_open:
#             try:
#                 line = self.ser.readline().decode().strip()
#                 if line:
#                     self.data_queue.put(float(line))
#             except (UnicodeDecodeError, ValueError) as e:
#                 print(f"Data error: {str(e)}")
#             except Exception as e:
#                 print(f"Serial error: {str(e)}")
#                 break

#     def process_data(self):
#         while self.is_reading.is_set():
#             try:
#                 data = self.data_queue.get(timeout=0.1)
#                 self.data_buffer.append(data)
#             except:
#                 continue

#     def plot_data(self):
#         self.ax.clear()
#         if self.data_buffer:
#             time = np.arange(len(self.data_buffer)) / 10000  # Assuming 1000 Hz sample rate
#             self.ax.plot(time, self.data_buffer)
#             self.ax.set_title("Recorded Signal")
#             self.ax.set_xlabel("Time (s)")
#             self.ax.set_ylabel("Amplitude")
#             self.canvas.draw()

#     def run_prediction(self):
#         if not self.data_buffer:
#             messagebox.showwarning("Warning", "No data collected!")
#             return
        
#         signal_data = np.array(self.data_buffer)
#         try:
#             runner("model_direct", signal_data)
#         except Exception as e:
#             messagebox.showerror("Error", f"Prediction failed: {str(e)}")

#     def on_closing(self):
#         self.is_reading.clear()
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#         self.master.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = RealTimePredictorApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()

# import tkinter as tk
# from tkinter import ttk
# from tkinter import messagebox
# import serial
# import serial.tools.list_ports
# from threading import Thread, Event
# from queue import Queue
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

# # Import your existing machine learning functions
# # from analyzer import extract_features, load_model, predict_drop
# from extract_predict import *

# class RealTimePredictorApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("Drop Classifier")
#         master.geometry("800x600")
        
#         # Initialize serial connection variables
#         self.ser = None
#         self.is_reading = Event()
#         self.data_queue = Queue()
#         self.data_buffer = []
        
#         # Create GUI components
#         self.create_widgets()
        
#         # Start serial port detection
#         self.detect_serial_port()

#     def create_widgets(self):
#         # Control Frame
#         control_frame = ttk.Frame(self.master)
#         control_frame.pack(pady=10)

#         # Serial Port Selection
#         self.port_var = tk.StringVar()
#         ttk.Label(control_frame, text="COM Port:").grid(row=0, column=0, padx=5)
#         self.port_combobox = ttk.Combobox(control_frame, textvariable=self.port_var, state='readonly')
#         self.port_combobox.grid(row=0, column=1, padx=5)
#         ttk.Button(control_frame, text="Refresh", command=self.detect_serial_port).grid(row=0, column=2, padx=5)

#         # Buttons
#         self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_collection)
#         self.start_btn.grid(row=0, column=3, padx=5)
#         self.predict_btn = ttk.Button(control_frame, text="Predict", command=self.run_prediction, state=tk.DISABLED)
#         self.predict_btn.grid(row=0, column=4, padx=5)

#         # Plot Frame
#         plot_frame = ttk.Frame(self.master)
#         plot_frame.pack(fill=tk.BOTH, expand=True)

#         # Matplotlib Figure
#         self.fig = Figure(figsize=(8, 4), dpi=100)
#         self.ax = self.fig.add_subplot(111)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
#         self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

#         # Prediction Label
#         self.prediction_label = ttk.Label(self.master, text="Prediction: None", font=('Helvetica', 14))
#         self.prediction_label.pack(pady=10)

#     def detect_serial_port(self):
#         ports = [port.device for port in serial.tools.list_ports.comports()]
#         self.port_combobox['values'] = ports
#         if ports:
#             self.port_var.set(ports[0])

#     def start_collection(self):
#         port = self.port_var.get()
#         if not port:
#             messagebox.showerror("Error", "No COM port selected!")
#             return

#         try:
#             self.ser = serial.Serial(port, baudrate=230400, timeout=1)
#             self.is_reading.set()
#             self.data_buffer = []
            
#             # Disable UI elements during collection
#             self.start_btn.config(state=tk.DISABLED)
#             self.port_combobox.config(state='disabled')
#             self.predict_btn.config(state=tk.DISABLED)
#             self.prediction_label.config(text="Collecting 5 seconds of data...")
            
#             # Start threads
#             Thread(target=self.read_serial, daemon=True).start()
#             Thread(target=self.process_data, daemon=True).start()
            
#             # Schedule automatic stop after 5 seconds
#             self.master.after(5000, self.stop_collection)
            
#         except serial.SerialException as e:
#             messagebox.showerror("Error", f"Failed to open {port}: {str(e)}")

#     def stop_collection(self):
#         self.is_reading.clear()
#         if self.ser and self.ser.is_open:
#             self.ser.close()
        
#         # Process any remaining data in the queue
#         while not self.data_queue.empty():
#             self.data_buffer.append(self.data_queue.get())
        
#         # Enable UI elements
#         self.start_btn.config(state=tk.NORMAL)
#         self.port_combobox.config(state='readonly')
#         self.predict_btn.config(state=tk.NORMAL)
#         self.prediction_label.config(text="Data collection complete!")
        
#         self.plot_data()

#     def read_serial(self):
#         while self.is_reading.is_set() and self.ser.is_open:
#             try:
#                 line = self.ser.readline().decode().strip()
#                 if line:
#                     self.data_queue.put(float(line))
#             except (UnicodeDecodeError, ValueError) as e:
#                 print(f"Data error: {str(e)}")
#             except Exception as e:
#                 print(f"Serial error: {str(e)}")
#                 break

#     def process_data(self):
#         while self.is_reading.is_set():
#             try:
#                 data = self.data_queue.get(timeout=0.1)
#                 self.data_buffer.append(data)
#             except:
#                 continue

#     def plot_data(self):
#         self.ax.clear()
#         if self.data_buffer:
#             time = np.arange(len(self.data_buffer)) / 10000  # Assuming 10000 Hz sample rate
#             self.ax.plot(time, self.data_buffer)
#             self.ax.set_title("Recorded Signal (5 seconds)")
#             self.ax.set_xlabel("Time (s)")
#             self.ax.set_ylabel("Amplitude")
#             self.canvas.draw()

#     def run_prediction(self):
#         if not self.data_buffer:
#             messagebox.showwarning("Warning", "No data collected!")
#             return
        
#         signal_data = np.array(self.data_buffer)
#         try:
#             # Assuming your runner function returns a prediction
#             prediction = runner("model_direct", signal_data)
#             self.prediction_label.config(text=f"Prediction: {prediction}")
#         except Exception as e:
#             messagebox.showerror("Error", f"Prediction failed: {str(e)}")

#     def on_closing(self):
#         self.is_reading.clear()
#         if self.ser and self.ser.is_open:
#             self.ser.close()
#         self.master.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = RealTimePredictorApp(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import serial
import serial.tools.list_ports
from threading import Thread, Event
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

# Import your existing machine learning functions
# from analyzer import extract_features, load_model, predict_drop
from extract_predict import *

class RealTimePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Drop Classifier")
        master.geometry("800x600")
        
        # Initialize serial connection variables
        self.ser = None
        self.is_reading = Event()
        self.data_queue = Queue()
        self.data_buffer = []
        self.waiting_for_trigger = False
        self.recording = False
        self.trigger_level = 0
        
        # Create GUI components
        self.create_widgets()
        
        # Start serial port detection
        self.detect_serial_port()

    def create_widgets(self):
        # Control Frame
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=10)

        # Serial Port Selection
        self.port_var = tk.StringVar()
        ttk.Label(control_frame, text="COM Port:").grid(row=0, column=0, padx=5)
        self.port_combobox = ttk.Combobox(control_frame, textvariable=self.port_var, state='readonly')
        self.port_combobox.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.detect_serial_port).grid(row=0, column=2, padx=5)

        # Trigger Level Input
        ttk.Label(control_frame, text="Trigger Level:").grid(row=0, column=3, padx=5)
        self.trigger_entry = ttk.Entry(control_frame, width=10)
        self.trigger_entry.grid(row=0, column=4, padx=5)

        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_collection)
        self.start_btn.grid(row=0, column=5, padx=5)
        self.predict_btn = ttk.Button(control_frame, text="Predict", command=self.run_prediction, state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=6, padx=5)

        # Plot Frame
        plot_frame = ttk.Frame(self.master)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Prediction Label
        self.prediction_label = ttk.Label(self.master, text="Prediction: None", font=('Helvetica', 14))
        self.prediction_label.pack(pady=10)

    def detect_serial_port(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combobox['values'] = ports
        if ports:
            self.port_var.set(ports[0])

    def start_collection(self):
        port = self.port_var.get()
        trigger_level = self.trigger_entry.get()
        
        if not port:
            messagebox.showerror("Error", "No COM port selected!")
            return
        if not trigger_level:
            messagebox.showerror("Error", "Please enter a trigger level!")
            return
            
        try:
            self.trigger_level = float(trigger_level)
        except ValueError:
            messagebox.showerror("Error", "Trigger level must be a number!")
            return

        try:
            self.ser = serial.Serial(port, baudrate=230400, timeout=1)
            self.data_buffer = []
            self.waiting_for_trigger = True
            self.recording = False
            
            # Disable UI elements during collection
            self.start_btn.config(state=tk.DISABLED)
            self.port_combobox.config(state='disabled')
            self.predict_btn.config(state=tk.DISABLED)
            self.prediction_label.config(text="Waiting for trigger...")
            
            # Start threads
            self.is_reading.set()
            Thread(target=self.read_serial, daemon=True).start()
            Thread(target=self.process_data, daemon=True).start()
            
        except serial.SerialException as e:
            messagebox.showerror("Error", f"Failed to open {port}: {str(e)}")

    def stop_collection(self):
        self.is_reading.clear()
        self.waiting_for_trigger = False
        self.recording = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        # Enable UI elements
        self.start_btn.config(state=tk.NORMAL)
        self.port_combobox.config(state='readonly')
        self.predict_btn.config(state=tk.NORMAL)
        self.prediction_label.config(text="Data collection complete!")
        
        self.plot_data()

    def read_serial(self):
        while self.is_reading.is_set() and self.ser.is_open:
            try:
                line = self.ser.readline().decode().strip()
                if line:
                    self.data_queue.put(float(line))
            except (UnicodeDecodeError, ValueError) as e:
                print(f"Data error: {str(e)}")
            except Exception as e:
                print(f"Serial error: {str(e)}")
                break

    def process_data(self):
        while self.is_reading.is_set():
            try:
                data = self.data_queue.get(timeout=0.1)
                
                if self.waiting_for_trigger:
                    if data >= self.trigger_level:
                        self.waiting_for_trigger = False
                        self.recording = True
                        self.prediction_label.config(text="Recording...")
                        self.master.after(5000, self.stop_collection)
                        self.data_buffer.append(data)
                elif self.recording:
                    self.data_buffer.append(data)
                    
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

    def plot_data(self):
        self.ax.clear()
        if self.data_buffer:
            time = np.arange(len(self.data_buffer)) / 10000  # Assuming 10000 Hz sample rate
            self.ax.plot(time, self.data_buffer)
            self.ax.set_title("Recorded Signal (5 seconds)")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.canvas.draw()

    def run_prediction(self):
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data collected!")
            return
        
        signal_data = np.array(self.data_buffer)
        try:
            # if (800 <= max(signal_data) <=1000):
            #     prediction = "Material: soft, Height: 10, Distance: 10"
            # else:
            #     # Assuming your runner function returns a prediction
            #     prediction = runner("model_direct", signal_data)
            
            prediction = runner("model_direct", signal_data)
            self.prediction_label.config(text=f"Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def on_closing(self):
        self.is_reading.clear()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePredictorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
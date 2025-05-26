import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

# # Replace with your actual file names
df1 = pd.read_csv('adc_data_20250526_134454.csv')
df2 = pd.read_csv('adc_filtered_data_20250526_134454.csv')

def debounce_binary_signal(binary_signal, min_duration_samples):
    """
    Removes short-duration pulses from a binary signal (debouncing).
    
    Parameters:
    - binary_signal (array-like): The quantized signal (0s and 1s).
    - min_duration_samples (int): Minimum number of samples a state must persist to be valid.
    
    Returns:
    - cleaned_signal (np.ndarray): Debounced binary signal.
    """
    cleaned = binary_signal.copy()
    start = 0
    current_state = binary_signal[0]

    for i in range(1, len(binary_signal)):
        if binary_signal[i] != current_state:
            duration = i - start
            if duration < min_duration_samples:
                cleaned[start:i] = [1 - current_state] * duration  # remove the pulse
            start = i
            current_state = binary_signal[i]
    return np.array(cleaned)


# df1 = pd.read_csv('adc_data_20250519_202202.csv')
# df2 = pd.read_csv('adc_filtered_data_20250519_202202.csv')
def mean_filter(data, window_size=8):
    """
    Apply a simple moving average filter to smooth the input data.
    
    Parameters:
    - data (array-like): Input data to be smoothed.
    - window_size (int): Number of samples over which to average.
    
    Returns:
    - filtered (np.ndarray): Smoothed signal.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def stateful_binary_quantize(data, inhale_threshold, exhale_threshold, window=5):
        if len(data) < window:
            return [0] * len(data)

        quantized = [0] * window
        state = 0  # 0 = rest/exhale, 1 = inhale/hold

        for i in range(window, len(data)):
            diff = data[i] - data[i - window]

            if state == 0:
                if diff > inhale_threshold:
                    state = 1  # inhale started
            else:  # state == 1
                if diff < -exhale_threshold:
                    state = 0  # exhale started

            quantized.append(state)

        return quantized

# def hysteresis_quantization(signal, high_ratio=0.8, low_ratio=0.6):
def hysteresis_quantization(signal, high_ratio=0.6, low_ratio=0.4):
    """
    Quantize a signal to 0 or 1 using hysteresis thresholding based on signal amplitude.
    
    Parameters:
    - signal (array-like): The input signal (e.g. filtered ADC values).
    - high_ratio (float): Ratio of (max - min) above min to set high threshold (default 0.6).
    - low_ratio (float): Ratio of (max - min) above min to set low threshold (default 0.4).
    
    Returns:
    - quantized (np.ndarray): Binary quantized output (0 or 1).
    """
    signal = np.asarray(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)

    high_thresh = min_val + high_ratio * (max_val - min_val)
    low_thresh = min_val + low_ratio * (max_val - min_val)

    quantized = np.zeros_like(signal, dtype=int)
    high = False

    for i in range(len(signal)):
        if not high and signal[i] > high_thresh:
            high = True
        elif high and signal[i] < low_thresh:
            high = False
        
        quantized[i] = 1 if high else 0

    return quantized
# quantized_ch1 = stateful_binary_quantize(df2.iloc[:, 0], 100, 100,50)
# quantized_ch2 = stateful_binary_quantize(df2.iloc[:, 1], 50, 50,50)
# Apply mean filter to each channel
# filtered_ch1 = mean_filter(df2.iloc[:, 0])
# filtered_ch2 = mean_filter(df2.iloc[:, 1])

# Then apply hysteresis quantization
quantized_ch1 = 1-hysteresis_quantization(df2.iloc[:, 0],0.5,0.4)
quantized_ch2 = hysteresis_quantization(df2.iloc[:, 1])
# Apply debouncing with a minimum duration of e.g. 15 samples
debounced_ch1 = debounce_binary_signal(quantized_ch1, min_duration_samples=50)
# debounced_ch2 = debounce_binary_signal(quantized_ch2, min_duration_samples=100)


fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Channel 1 subplot
axs[0].plot(df1.iloc[:, 0], label='File1 Channel 1')
axs[0].plot(df2.iloc[:, 0], label='File2 Channel 1')
# axs[0].plot(filtered_ch1, label='mean filtered')
axs[0].set_ylabel('Channel 1 Values')
axs[0].set_title('Channel 1')
axs[0].legend()

# Channel 2 subplot
axs[1].plot(df1.iloc[:, 1], label='File1 Channel 2')
axs[1].plot(df2.iloc[:, 1], label='File2 Channel 2')
# axs[1].plot(filtered_ch2, label='mean filtered')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Channel 2 Values')
axs[1].set_title('Channel 2')
axs[1].legend()


axs[2].plot(quantized_ch1, 'y-', label='Quantized Ch1')   
axs[2].plot(quantized_ch2, 'c-', label='Quantized Ch2')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Discrete Signals')
axs[2].set_title('Quantized Channels')
axs[2].legend()

# New subplot for debounced signals
axs[3].plot(debounced_ch1, 'orange', label='Debounced Ch1')
axs[3].plot(quantized_ch2, 'lime', label='Debounced Ch2')
axs[3].set_xlabel('Index')
axs[3].set_ylabel('Discrete Signals')
axs[3].set_title('Debounced Binary Signals')
axs[3].legend()

plt.tight_layout()
plt.show()


with open("quantised_4.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Quantized Ch1", "Quantized Ch2"])
    writer.writerows(zip(quantized_ch1, quantized_ch2))
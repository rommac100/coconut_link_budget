
import numpy as np
import matplotlib.pyplot as plt



# Parameters
B = 500e3  # (500 kHz) bandwidth
T = 1 / B  # sampling period
SF = 12  # spreading factor {7,8,9,10,11,12}
T_s = (2**SF) * T  # symbol period
num_simulations = 100000  # number of simulations

# Desired transmission power (dBm to W)
P_dBm = 14  # Example: 14 dBm
P_W = 10**((P_dBm - 30) / 10)

# Function to generate symbols
def generate_symbol(w):
    symbol = 0
    for h in range(SF):
        symbol += w[h] * (2**h)
    return symbol

# Function to adjust signal power
def adjust_power(signal, desired_power):
    current_power = np.mean(np.abs(signal)**2)
    adjustment_factor = np.sqrt(desired_power / current_power)
    return signal * adjustment_factor


# SNR values in dB
SNR_dB_values = np.arange(-30, -20, 1)
#SNR_dB_values = [-6.8]
BER_values = []

for SNR_dB in SNR_dB_values:
    SNR = 10**(SNR_dB / 10)
    
    # Noise power
    noise_power = P_W / SNR
    
    bit_errors = 0
    total_bits = 0
    
    for _ in range(num_simulations):
        w = np.random.randint(0, 2, SF)
        symbol = generate_symbol(w)
        
        t = np.linspace(0, T_s, int(T_s / T))
        k = np.arange(0, len(t))
        
        # Chirp generation
        chirp = np.exp(1j * 2 * np.pi * ((symbol + k) % (2**SF)) / (2**SF) * k)
        
        # Adjust chirp power
        chirp = adjust_power(chirp, P_W)
        
        # Generate Gaussian noise
        noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, len(k)) + 1j * np.random.normal(0, 1, len(k)))
        
        # Received signal
        received_signal = chirp + noise
        
        # Down-chirp
        down_chirp = np.exp(-1j * 2 * np.pi * (k**2) / (2**SF))
        down_chirp = adjust_power(down_chirp, P_W)
        
        # Received signal multiplied by down-chirp
        mult_signal = received_signal * down_chirp
        
        # Discrete Fourier Transform
        dft_signal = np.fft.fft(mult_signal)
        
        # Estimate of the transmitted symbol
        symbol_estimate = np.argmax(np.abs(dft_signal))
        
        # Convert estimate back to bit vector
        w_estimate = [(symbol_estimate >> h) & 1 for h in range(SF)]
        
        # Calculate bit errors
        bit_errors +=np.sum(np.array(w) != np.array(w_estimate))
        total_bits += SF
    
    # Calculate bit error rate for current SNR
    BER = bit_errors / total_bits
    BER_values.append(BER)
    print(f"SNR (dB): {SNR_dB}, BER: {BER}")


plt.figure(figsize=(10, 6))
plt.semilogy(SNR_dB_values, BER_values, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. SNR')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
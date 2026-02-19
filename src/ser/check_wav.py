import soundfile as sf
import torch
import numpy as np
import os

filename = "current_input.wav"

if not os.path.exists(filename):
    print(f"Error: {filename} does not exist.")
else:
    print(f"File found: {filename}")
    print(f"Size: {os.path.getsize(filename)} bytes")
    
    try:
        data, samplerate = sf.read(filename)
        print(f"Soundfile read shape: {data.shape}")
        print(f"Sample rate: {samplerate}")
        print(f"Data type: {data.dtype}")
        print(f"Max value: {np.max(data)}")
        print(f"Min value: {np.min(data)}")
        
        # Test the custom load logic manually
        # specific to how soundfile handles mono (1D array)
        if data.ndim == 1:
            # Add channel dimension: (time,) -> (1, time)
            tensor = torch.from_numpy(data).unsqueeze(0)
            print("Detected Mono. Unsqueezing to (1, time).")
        else:
            # Transpose: (time, channels) -> (channels, time)
            tensor = torch.from_numpy(data.transpose())
            print(f"Detected {data.shape[1]} channels. Transposing.")
            
        print(f"Final Tensor shape: {tensor.shape}")
        
        if tensor.shape[1] < 100:
             print("WARNING: Audio is extremely short!")
             
    except Exception as e:
        print(f"Error reading file with soundfile: {e}")

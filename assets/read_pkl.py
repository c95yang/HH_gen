import pickle
import numpy as np

# Try different approaches to read the pickle file
methods = [
    lambda f: pickle.load(f, encoding='latin1'),
    lambda f: pickle.load(f, encoding='bytes'),
    lambda f: pickle.load(f, fix_imports=True, encoding='latin1'),
    lambda f: pickle.load(f, fix_imports=True, encoding='bytes'),
]

data = None
for i, method in enumerate(methods):
    try:
        with open('/home/cc/Downloads/tridi-main/assets/smpl_segmentation.pkl', 'rb') as f:
            data = method(f)
        print(f"Successfully loaded with method {i+1}")
        break
    except Exception as e:
        print(f"Method {i+1} failed: {e}")

if data is not None:
    # Examine the structure
    print("Type:", type(data))
    print("Keys:", data.keys() if isinstance(data, dict) else "Not a dictionary")

    # Handle potential bytes keys
    if isinstance(data, dict):
        # Convert bytes keys to strings if necessary
        if any(isinstance(k, bytes) for k in data.keys()):
            data = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in data.items()}
            print("Converted bytes keys to strings")
            print("New keys:", data.keys())

    # Look at the 'segm' array
    segm_key = 'segm'
    if segm_key in data:
        segm = data[segm_key]
        print(f"Segmentation array shape: {segm.shape}")
        print(f"Data type: {segm.dtype}")
        print(f"Min value: {segm.min()}")
        print(f"Max value: {segm.max()}")
        print(f"Unique values: {np.unique(segm)}")
        print(f"First 20 values: {segm[:20]}")
    elif b'segm' in data:
        segm = data[b'segm']
        print("Found segm with bytes key")
        print(f"Segmentation array shape: {segm.shape}")
        print(f"Data type: {segm.dtype}")
        print(f"Min value: {segm.min()}")
        print(f"Max value: {segm.max()}")
        print(f"Unique values: {np.unique(segm)}")
        print(f"First 20 values: {segm[:20]}")
    else:
        print("Could not find 'segm' key")
        print("Available keys/items:", list(data.keys()) if isinstance(data, dict) else data)
else:
    print("Failed to load the pickle file with any method")
from pathlib import Path
import time
import numpy as np

from model_inference import get_model_inference
from podsai_onnx_inference import get_podsai_onnx_inference

wav_dir=Path('output/testing-wav')
wavs=sorted([p for p in wav_dir.rglob('*.wav')])[:10]
print('Using wavs:', [p.name for p in wavs])

print('Loading HF podsai model...')
hf = get_model_inference(model_type='podsai', model_path='davethaler/whale-call-detector', model_revision=None)
print('Loading ONNX podsai model...')
onnx = get_podsai_onnx_inference('podsai.onnx', feature_extractor='davethaler/whale-call-detector')

hf_times=[]
onnx_times=[]

print('\nRunning HF (PyTorch) predictions:')
for p in wavs:
    t0=time.perf_counter()
    _ = hf.predict(str(p))
    t1=time.perf_counter()
    hf_times.append(t1-t0)
    print('HF', p.name, f"{hf_times[-1]:.3f}s")

print('\nRunning ONNX predictions:')
for p in wavs:
    t0=time.perf_counter()
    _ = onnx.predict(str(p))
    t1=time.perf_counter()
    onnx_times.append(t1-t0)
    print('ONNX', p.name, f"{onnx_times[-1]:.3f}s")

print() 
print('HF mean', np.mean(hf_times), 'ONNX mean', np.mean(onnx_times))
for i,p in enumerate(wavs):
    print(p.name, 'HF', f"{hf_times[i]:.3f}s", 'ONNX', f"{onnx_times[i]:.3f}s", 'diff', f"{(hf_times[i]-onnx_times[i]):.3f}s")

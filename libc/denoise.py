import ctypes

cfunc = ctypes.CDLL("libc/librnnoise.so")
cfunc.rnnRealWrap.restype = ctypes.POINTER(ctypes.c_float)

# 录音文件路径降噪
def denoise_file(src_path):
    path = src_path.encode('utf-8')
    cfunc.rnnDeNoise(ctypes.c_char_p(path), ctypes.c_char_p(path))

# 实时录音降噪(采样率48k)
def real_denoise(audio, sr=32000):
    frameSize = len(audio)
    wav = cfunc.rnnRealWrap(ctypes.byref((ctypes.c_float * frameSize)(*(f for f in audio))), ctypes.c_uint64(frameSize), ctypes.c_uint32(sr), ctypes.c_uint32(1))
    wavs = [i for i in wav[0:frameSize]]
    return wavs


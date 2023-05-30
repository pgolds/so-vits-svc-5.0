import torch

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='/root/.cache/torch/hub/snakers4_silero-vad_master', model='silero_vad', force_reload=False, source='local', onnx=False)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

#wav = read_audio(f'/data2/song_dir/阿拉斯加海湾/v.wav', sampling_rate=16000)

vad_iterator = VADIterator(model)


def vad_slice1(wav_path):
    wav = read_audio(wav_path, sampling_rate=16000)
    vad_arr = []
    index = 0
    window_size_samples = 8192
    end = 0
    for i in range(0, len(wav), window_size_samples):
        speech_dict = vad_iterator(wav[i:i+window_size_samples], return_seconds=True)
        if speech_dict:
            if "end" in speech_dict:
#                vad_arr[index]["end"] = speech_dict["end"]
#                index = index + 1
                end = speech_dict["end"]
            else:
#                vad_arr.append(speech_dict)
                start = speech_dict["start"]
                if start - end > 3:
                    if end != 0:
                        end += 0.2
                    vad_arr.append({"start": end, "end": start - 0.8})
    if len(vad_arr) > 0 and end != vad_arr[-1]["start"]:
        vad_arr.append({"start": end, "end": len(wav)/16000})
    vad_iterator.reset_states()
    return vad_arr

def vad_slice(wav):
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    return speech_timestamps

#vad_arr = vad_slice(wav)

#for arr in vad_arr:
#    start_index = int(arr["start"] * 16000)
#    end_index = int(arr["end"] * 16000)
#    print(f"{start_index}-----{end_index}")
#    wav[start_index:end_index] = wav[start_index:end_index] * 1.5
#    print(arr)

#save_audio('vad_test.wav', wav, sampling_rate=16000)

#speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

#import time
#t0 = time.perf_counter()
#info = vad_slice1(wav)
#print(info)
#print(time.perf_counter() - t0)

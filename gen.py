import os
import torch
import argparse
import numpy as np

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer
from pitch import load_csv_pitch

from vad import vad_slice1

def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def main(args):
    modelPath = os.path.join(args.modelPath, args.spk, "best.pth")
    spkPath = os.path.join(args.modelPath, args.spk, "spk.npy")
    songV = os.path.join(args.songPath, "v.wav")
    songI = os.path.join(args.songPath, "i.wav")
    ppgPath = os.path.join(args.songPath, "v.ppg.npy")
    pitPath = os.path.join(args.songPath, "v.pit.csv")
    output = args.output

    if not os.path.exists(ppgPath):
        print(
            f"Auto run : python whisper/inference.py -w {songV} -p {ppgPath}")
        os.system(f"python whisper/inference.py -w {songV} -p {ppgPath}")

    if not os.path.exists(pitPath):
        print(
            f"Auto run : python pitch/inference.py -w {songV} -p {pitPath}")
        os.system(f"python pitch/inference.py -w {songV} -p {pitPath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(modelPath, model)
    model.eval()
    model.to(device)

    spk = np.load(spkPath)
    spk = torch.FloatTensor(spk)

    ppg = np.load(ppgPath)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = load_csv_pitch(pitPath)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift

    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]

    vad_timestamp = vad_slice1(songV)

    with torch.no_grad():

        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        # pitwav = model.source2wav(source)
        # write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []
        has_audio = False

        while (out_index + out_chunk < all_frame):
            has_audio = True
            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = out_index + out_chunk
                cut_e_out = 0
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = ppg[cut_s:cut_e, :].unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        if (out_index < all_frame):
            if (has_audio):
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size
            else:
                cut_s = 0
                cut_s_out = 0
            sub_ppg = ppg[cut_s:, :].unsqueeze(0).to(device)
            sub_pit = pit[cut_s:].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([all_frame - cut_s]).to(device)
            sub_har = source[:, :, cut_s * hop_size:].to(device)
            sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:]
            out_audio.extend(sub_out)
        out_audio = np.asarray(out_audio)

    for time_point in vad_timestamp:
        start_index = int(time_point["start"] * hp.data.sampling_rate)
        end_index = int(time_point["end"] * hp.data.sampling_rate)
        out_audio[start_index:end_index] = 0

    tmpOut = f"{output}.wav"
    write(tmpOut, hp.data.sampling_rate, out_audio)

    if os.path.exists(songI):
        mergeCmd = f"ffmpeg -i {tmpOut} -i {songI} -filter_complex amix=inputs=2:duration=first -ac 2 -f mp3 {output}"
        os.system(mergeCmd)
        os.remove(tmpOut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="configs/finetune.yaml",
                        help="yaml file for config.")
    parser.add_argument('-m', '--modelPath', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('-w', '--songPath', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path of output audio")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)

import os
import argparse
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("--dataRoot", type=str, help="datasets path")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.dataRoot, "filelists"), exist_ok=True)

    rootPath = os.path.join(args.dataRoot, "waves")
    all_items = []
    for spks in os.listdir(f"{rootPath}"):
        if os.path.isdir(f"{rootPath}/{spks}"):
            for file in os.listdir(f"{rootPath}/{spks}"):
                if file.endswith(".wav"):
                    file = file[:-4]
                    path_spk = f"{args.dataRoot}/speaker/{spks}/{file}.spk.npy"
                    path_wave = f"{args.dataRoot}/waves/{spks}/{file}.wav"
                    path_spec = f"{args.dataRoot}/specs/{spks}/{file}.pt"
                    path_pitch = f"{args.dataRoot}/pitch/{spks}/{file}.pit.npy"
                    path_whisper = f"{args.dataRoot}/whisper/{spks}/{file}.ppg.npy"
                    assert os.path.isfile(path_spk), path_spk
                    assert os.path.isfile(path_wave), path_wave
                    assert os.path.isfile(path_spec), path_spec
                    assert os.path.isfile(path_pitch), path_pitch
                    assert os.path.isfile(path_whisper), path_whisper
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{path_spk}")
        else:
            file = spks
            if file.endswith(".wav"):
                file = file[:-4]
                path_spk = f"{args.dataRoot}/speaker/{file}.spk.npy"
                path_wave = f"{args.dataRoot}/waves/{file}.wav"
                path_spec = f"{args.dataRoot}/specs/{file}.pt"
                path_pitch = f"{args.dataRoot}/pitch/{file}.pit.npy"
                path_whisper = f"{args.dataRoot}/whisper/{file}.ppg.npy"
                assert os.path.isfile(path_spk), path_spk
                assert os.path.isfile(path_wave), path_wave
                assert os.path.isfile(path_spec), path_spec
                assert os.path.isfile(path_pitch), path_pitch
                assert os.path.isfile(path_whisper), path_whisper
                all_items.append(
                    f"{path_wave}|{path_spec}|{path_pitch}|{path_whisper}|{path_spk}")

    random.shuffle(all_items)
    valids = all_items[:50]
    valids.sort()
    trains = all_items[50:]
    # trains.sort()
    fw = open(os.path.join(args.dataRoot, "filelists", "valid.txt"), "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open(os.path.join(args.dataRoot, "filelists", "train.txt"), "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()

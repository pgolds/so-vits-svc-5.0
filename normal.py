import os
import argparse
from libc.denoise import denoise_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="./data_svc/waves-slice", help="Directory path where output audio files will be saved.")
    parser.add_argument("--out_dir", type=str, default="./data_svc/waves", help="Directory path where output audio files will be saved.")
    parser.add_argument("--sr", type=int, default=48000, help="audio sampleRate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for f in os.scandir(args.in_dir):
        wav_path = os.path.join(args.in_dir, f.name)
        out_path = os.path.join(args.out_dir, f.name)
        os.system(f"ffmpeg -i {wav_path} -ar {args.sr} -af loudnorm=I=-16:TP=-1:LRA=7:print_format=json {out_path}")


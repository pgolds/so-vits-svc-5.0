import os
import argparse
from libc.denoise import denoise_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./data_svc/waves-32k", help="Directory path where output audio files will be saved.")
    args = parser.parse_args()

    for f in os.scandir(args.out_dir):
        wav_path = os.path.join(args.out_dir, f.name)
        denoise_file(wav_path)


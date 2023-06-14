import os
import time
import argparse
import gradio as gr

def inference_svc(speaker, songName):
    songPath = os.path.join(args.song_path, songName)
    outputPath = os.path.join(args.outPath, speaker, time.strftime("%Y-%m-%d", time.localtime()))
    os.makedirs(outputPath, exist_ok=True)
    nowTimestamp = str(round(time.time() * 1000))
    output = f"{outputPath}/{nowTimestamp}.mp3"
    cmd = f"CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python gen.py --modelPath {args.model_path} --spk {speaker} --songPath {songPath} --output {output}"
    print(cmd)
    os.system(cmd)
    return output

def refresh_spk():
    speakers = []
    for f in os.scandir(args.model_path):
        if f.is_dir():
            speakers.append(f.name)
    return gr.update(choices=speakers)

def refresh_song():
    songs = []
    for f in os.scandir(args.song_path):
        if f.is_dir():
            songs.append(f.name)
    return gr.update(choices=songs)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True,
                    help="yaml file for config.")
parser.add_argument("--share", action="store_true",
                    default=False, help="share gradio app")
parser.add_argument("--song_path", type=str,
                    default="/data2/song_dir", help="song dir")
parser.add_argument("--model_path", type=str,
                    default="/data2/singing/models", help="model dir")
parser.add_argument("--outPath", type=str,
                    default="infer_out", help="inference output dir")

args = parser.parse_args()

speakers = []
songs = []

if __name__ == "__main__":

    for f in os.scandir(args.model_path):
        if f.is_dir():
            speakers.append(f.name)

    for f in os.scandir(args.song_path):
        if f.is_dir():
            songs.append(f.name)

    app = gr.Blocks()
    with app:
        gr.Markdown("## speech singing transfer")

        with gr.Tabs():
            with gr.TabItem("SVC"):
                with gr.Row():
                    with gr.Column():
                        speaker = gr.Radio(choices=speakers, value=speakers[0], label='发声人')
                        song = gr.Radio(choices=songs, value=songs[0], label='歌曲')
                    with gr.Column():
                        audio_output = gr.Audio(
                            label="合成歌曲", elem_id="generate-audio")
                        btn = gr.Button("生成歌曲")
                        btn.click(inference_svc,
                                  inputs=[speaker, song],
                                  outputs=[audio_output])

                    speaker.change(refresh_spk, [], [speaker])
                    song.change(refresh_song, [], [song])

    app.queue(concurrency_count=3).launch(show_api=False, share=args.share, server_name="0.0.0.0", server_port=12000)
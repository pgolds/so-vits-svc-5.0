import os
import sys
import argparse
import subprocess
import shutil

python_bin = "/root/miniconda3/envs/vits/bin/python"

def run_cmd(cmd, cwd_path, message, check_result_path=None):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd_path)
    res = p.wait()
    print(cmd)
    print("运行结果：", res)

    # 检查是否有对应文件生成
    if check_result_path:
        file_num = [filename for filename in os.listdir(check_result_path)]
        if file_num == 0:
            res = 1

    if res == 0:
        # 运行成功
        return True
    else:
        # 运行失败
        print(message)
        print(f"你可以新建终端，打开命令行，进入：{cwd_path}")
        print(f"执行：{cmd}")
        return False

def run(args):
    # 原始音频地址
    rawAudioPath = os.path.join(args.dataRoot, "raw")
    # 分割音频地址
    audioSlicePath = os.path.join(args.dataRoot, "waves-slice")
    os.makedirs(audioSlicePath, exist_ok=True)
    # 训练录音
    audioPath = os.path.join(args.dataRoot, "waves")
    os.makedirs(audioPath, exist_ok=True)
    # 音色编码地址
    speakerPath = os.path.join(args.dataRoot, "speaker")
    # ppg抗扰文件地址
    whisperPath = os.path.join(args.dataRoot, "whisper")
    # pit文件地址
    pitPath = os.path.join(args.dataRoot, "pitch")
    # specs文件地址
    specsPath = os.path.join(args.dataRoot, "specs")
    # 模型保存地址
    modelPath = os.path.join(args.modelPath, args.name)
    os.makedirs(modelPath, exist_ok=True)
    # 合成音色保存地址
    timbrePath = os.path.join(modelPath, "spk.npy")
    sr = args.sr

    filelistsPath = os.path.join(args.dataRoot, "filelists")

    # slice audio
    slice_cmd = f"{python_bin} slicer.py {rawAudioPath} --out {audioPath}"
    result = run_cmd(slice_cmd, None, message=f"error: step 1 => 分割音频处理失败", check_result_path=audioSlicePath)
    if not result:
        sys.exit(1)

    # normal audio
    normal_cmd = f"{python_bin} normal.py --in_dir {audioSlicePath} --out_dir {audioPath} --sr {sr}"
    result = run_cmd(normal_cmd, None, message=f"error step 1 => 标准化音频处理失败", check_result_path=audioPath)
    if not result:
        sys.exit(1)

    # denoise audio
    denoise_cmd = f"{python_bin} denoise.py --out_dir {audioPath}"
    result = run_cmd(denoise_cmd, None, message=f"error: step 2 => 降噪音频处理失败", check_result_path=audioPath)
    if not result:
        sys.exit(1)

    # generate speaker
    speaker_cmd = f"{python_bin} prepare/preprocess_speaker.py {audioPath} {speakerPath}"
    result = run_cmd(speaker_cmd, None, message=f"error: step 3 => 提取音频音色失败", check_result_path=speakerPath)
    if not result:
        sys.exit(1)

    # generate ppg
    ppg_cmd = f"{python_bin} prepare/preprocess_ppg.py -w {audioPath} -p {whisperPath}"
    result = run_cmd(ppg_cmd, None, message=f"error: step 4 => 提取音频PPG失败", check_result_path=whisperPath)
    if not result:
        sys.exit(1)

    # generate pit
    pit_cmd = f"{python_bin} prepare/preprocess_f0.py -w {audioPath} -p {pitPath}"
    result = run_cmd(pit_cmd, None, message=f"error: step 5 => 提取音频F0失败", check_result_path=pitPath)
    if not result:
        sys.exit(1)

    # generate specs
    specs_cmd = f"{python_bin} prepare/preprocess_spec.py -w {audioPath} -s {specsPath}"
    result = run_cmd(specs_cmd, None, message=f"error: step 6 => 提取音频线性谱失败", check_result_path=specsPath)
    if not result:
        sys.exit(1)

    # generate speaker ave
    timbre_cmd = f"{python_bin} prepare/preprocess_speaker_ave.py {speakerPath} {args.dataRoot}"
    result = run_cmd(timbre_cmd, None, message=f"error: step 7 => 提取平均音色合成文件失败")
    if not result:
        sys.exit(1)

    # generate filelists
    filelists_cmd = f"{python_bin} prepare/preprocess_train.py --dataRoot {args.dataRoot}"
    result = run_cmd(filelists_cmd, None, message=f"error: step 8 => 生成训练标注文件失败", check_result_path=filelistsPath)
    if not result:
        sys.exit(1)

    # 复制合成音色文件至模型保存目录
    shutil.copy(os.path.join(args.dataRoot, "spk.npy"), timbrePath)

    # run train
    train_cmd = f"{python_bin} svc_trainer.py --config {args.config} --name {args.name} --chkpt_dir {args.modelPath} --dataRoot {args.dataRoot} --save_interval {args.save_interval} --max_step {args.max_step}"
    result = run_cmd(train_cmd, None, message=f"error: step 9 => 训练异常", check_result_path=modelPath)
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/finetune.yaml",
                        help="yaml file for config.")
    parser.add_argument('--dataRoot', type=str,
                        help="Path of datasets root")
    parser.add_argument('--modelPath', type=str, required=True,
                        help="model file for write path")
    parser.add_argument('-s', '--save_interval', type=int, default=1500,
                        help="saving step checkpoint")
    parser.add_argument('-m', '--max_step', type=int, default=1510,
                        help="saving step checkpoint")
    parser.add_argument('--sr', type=int, default=48000, help="audio sample rate")
    parser.add_argument('-n', '--name', type=str, help="train run name")
    args = parser.parse_args()
    run(args)


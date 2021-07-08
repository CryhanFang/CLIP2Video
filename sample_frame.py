import os
from concurrent import futures
import argparse

def extract_frames(video_name, out_folder, fps=5):
    if os.path.exists(out_folder):
        os.system('rm -rf ' + out_folder + '/*')
        os.system('rm -rf ' + out_folder)
    os.makedirs(out_folder)
    cmd = 'ffmpeg -v 0 -i %s -r %d -q 0 %s/%s.jpg' % (video_name, fps, out_folder, '%08d')
    os.system(cmd)

def process(line):
    print(line)
    mp4_name, folder_frame = line
    extract_frames(mp4_name, folder_frame)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get frames from video')
    parser.add_argument('--input_path', type=str, default='input', help='input directory of videos')
    parser.add_argument('--output_path', type=str, default='output', help='output directory of sampled frames')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)


    mp4_file = os.listdir(args.input_path)
    lines = [(os.path.join(args.input_path, mp4), os.path.join(args.output_path, mp4.split(".")[0])) for mp4 in mp4_file]

    # multi thread
    with futures.ProcessPoolExecutor(max_workers=10) as executer:
        fs = [executer.submit(process, line) for line in lines]
    print("done")

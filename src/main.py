import argparse

from src.id_detect import csv_reader, detect_from_id
from src.video_detect import detect_only_vid

parser = argparse.ArgumentParser(description='Detection setting')
parser.add_argument('--vid_path', dest="vid_path", type=str, default="../source/parasite-test-result.avi",
                    help='source video path')
parser.add_argument('--csv_path', dest="csv_path", type=str, default="../output/detection/parasite-test.csv",
                    help='source csv path')
parser.add_argument("--3D", dest="dim", action="store_const",
                    const="3D", default="2D", help='2D or 3D when FAN')

args = parser.parse_args()

if __name__ == "__main__":
    if args.vid_path == None:
        print("Please add video path")
    elif args.csv_path == None:
        detect_only_vid(args.vid_path, args.dim)
    else:
        detect_from_id(args.vid_path, csv_reader(args.csv_path))
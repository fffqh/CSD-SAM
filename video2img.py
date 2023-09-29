import cv2
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='data/csd.mp4')
    parser.add_argument('--out_path', type=str, default='data/ori_imgs/csd')
    args = parser.parse_args()
    
    video_path = args.video_path
    out_path = args.out_path
    
    assert os.path.exists(video_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    
    cap = cv2.VideoCapture(video_path)
    
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 获得视频文件的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)              # 获得视频文件的帧率
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # 获得视频文件的帧宽
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获得视频文件的帧高
    
    # 
    #video_length = frames / int(fps)
    for i in range(int(frames)):
        ret, frame = cap.read()
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.imwrite(os.path.join(out_path, "{:03d}.jpg".format(int(pos))), frame)
    cap.release()
    print(int(frames))
    exit(0)

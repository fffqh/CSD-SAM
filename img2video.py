import os
import cv2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='data/output/csd')
    parser.add_argument('--out_path', type=str, default='data/output/output.mp4')
    args = parser.parse_args()
    
    img_path = args.img_path
    out_path = args.out_path
    
    assert os.path.exists(img_path)
    assert not os.path.exists(out_path)

    img_file_list = sorted(list(filter(lambda x: x.endswith('final.jpg'),
                                       os.listdir(img_path))))
    assert len(img_file_list) > 0
    
    img = cv2.imread(os.path.join(img_path, img_file_list[0]))
    height,width = img.shape[0], img.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 10, (int(width), int(height)))

    for file_name in img_file_list:
        frame = cv2.imread(os.path.join(img_path, file_name))
        out.write(frame)
    out.release()
    print("Done.")
    
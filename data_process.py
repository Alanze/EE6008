import os
import shutil

import cv2


def convert_gt_to_yolo(mot_root, output_root):

    for seq in os.listdir(mot_root):
        seq_path = os.path.join(mot_root, seq)
        if not os.path.isdir(seq_path):
            continue

        print(f"Processing {seq}...")

        img_dir = os.path.join(seq_path, "img1")
        gt_file = os.path.join(seq_path, "gt", "gt.txt")

        out_img_dir = os.path.join(output_root, "images", "train", seq)
        out_lbl_dir = os.path.join(output_root, "labels", "train", seq)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        sample_img = cv2.imread(os.path.join(img_dir, sorted(os.listdir(img_dir))[0]))
        imHeight, imWidth = sample_img.shape[:2]

        # gt.txt
        with open(gt_file) as f:
            lines = f.readlines()

        annotations = {}
        for line in lines:
            frame, _obj_id, x, y, w, h, conf, _cls, _vis = line.strip().split(",")
            frame, x, y, w, h = int(frame), float(x), float(y), float(w), float(h)

            if float(conf) < 0.5:
                continue

            # 转换为 YOLO 格式
            x_center = (x + w / 2) / imWidth
            y_center = (y + h / 2) / imHeight
            w_norm = w / imWidth
            h_norm = h / imHeight

            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        for img_name in sorted(os.listdir(img_dir)):
            frame_id = int(os.path.splitext(img_name)[0])
            img_in = os.path.join(img_dir, img_name)
            img_out = os.path.join(out_img_dir, img_name)

            shutil.copy(img_in, img_out)

            lbl_file = os.path.join(out_lbl_dir, f"{frame_id:06d}.txt")
            with open(lbl_file, "w") as f:
                if frame_id in annotations:
                    f.write("\n".join(annotations[frame_id]))

    print("Conversion finished!")


mot_root = r"E:\\yolov8\\OpenDataLab___MOT20\\raw\\MOT20\\train"
output_root = r"E:\\yolov8\\datasets\\MOT20-YOLO"
convert_gt_to_yolo(mot_root, output_root)

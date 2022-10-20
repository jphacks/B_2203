import cv2, dlib, sys, glob, pprint, os
import numpy as np
from pathlib import Path  # 要インストール
#from natsort import natsorted  # 要インストール

# 入力ディレクトリの指定
indir = "C:\\Users\\Masaharu\\Desktop\\main\\test_images_origin"
# 切り抜かれた顔画像の保存先
clipped_face_dir =  "C:\\Users\\Masaharu\\Desktop\\main\\test_images_clipped"
# Cycleganの出力先ディレクトリ
cyclegan_result_dir = "C:\\Users\\Masaharu\\Desktop\\main\\results\\mask2nonmask_results_succes\\test_latest\\images\\fake_image"
# 出力ディレクトリの指定
outdir = "C:\\Users\\Masaharu\\Desktop\\main\\test_images_insert"

# 保存先ディレクトリの作成
if not os.path.exists(clipped_face_dir):
    os.makedirs(clipped_face_dir)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# 暫定的な画像のID
fid = 0000

# 入力画像をリサイズするか？
flag_resize = False

# dlibをはじめる
detector = dlib.get_frontal_face_detector()

# 顔画像を取得して保存する
def get_face(fname, clipped_face_dir, detector):
    global fid
    img = cv2.imread(fname)
    face_list = []
    cornar_list = [[],[],[],[]]

    # デジタルカメラなどの画像であれば
    # サイズが大きいのでリサイズ
    if flag_resize:
        img = cv2.resize(img, None,
            fx = 0.2, fy = 0.2)
    # 顔検出
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        pprint.pprint(d)
        x1 = int(d.left())
        y1 = int(d.top())
        x2 = int(d.right())
        y2 = int(d.bottom())
        im = img[y1:y2, x1:x2]
        
        # 50x50にリサイズ
        """try:
            im = cv2.resize(im, (50, 50))
        except:
            continue
        # 保存
        out = outdir + "/" + str(fid) + ".jpg"
        cv2.imwrite(out, im)
        fid += 1"""
        # リサイズなし
        out = clipped_face_dir + "/" + str(fid) + ".jpg"
        cv2.imwrite(out, im)
        fid += 1

        face_list.append(im)
        cornar_list[0].append(x1)
        cornar_list[1].append(x2)
        cornar_list[2].append(y1)
        cornar_list[3].append(y2)

    return img, face_list, cornar_list


def insert_translated_face(img, t_img_list, cornar_list, outdir):
    global fid
    for i in range(len(t_img_list)):
        # 変換された画像を切り抜かれた時の大きさにリサイズ
        t_img_list[i] = cv2.resize(t_img_list[i], dsize=(cornar_list[1][i]-cornar_list[0][i], cornar_list[3][i]-cornar_list[2][i]))
        # 顔が切り抜かれた部分に，マスク付きの画像を挿入
        img[cornar_list[2][i]:cornar_list[3][i], cornar_list[0][i]:cornar_list[1][i]] = t_img_list[i]
        """cv2.imshow("img_test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    out = outdir + "/" + str(fid) + ".jpg"
    cv2.imwrite(out, img)
    fid += 1

def read_img(indir, cyclegan_result_dir, clipped_face_dir, outdir):
    # 画像の読み取り
    image_paths = [str(p) for p in Path(indir).glob("*")]  # natsortedで番号順に並び替え
    for f in image_paths:
        print(f)
        img, face_list, cornar_list = get_face(f, clipped_face_dir)  # 顔がクリップされる前の画像，クリップされた顔の画像，クリップされた場所
        
        """face_listをtranslated_img_list(マスク付き画像)に変換"""
        translated_img_list = []
        translated_img_dir = [str(p) for p in Path(cyclegan_result_dir).glob("*")]
        for i in range(len(translated_img_dir)):
            print(translated_img_dir[i])
            translated_img = cv2.imread(translated_img_dir[i])
            translated_img_list.append(translated_img)

        insert_translated_face(img, translated_img_list, cornar_list, outdir)
    print("ok")
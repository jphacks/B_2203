import cv2, pprint
from pathlib import Path  # 要インストール

# 顔画像を取得して保存する
def get_face(fname, detector):
    #global fid
    img = cv2.imread(fname)
    face_list = []
    cornar_list = [[],[],[],[]]

    # デジタルカメラなどの画像であれば
    # サイズが大きいのでリサイズ
    flag_resize = False
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

        face_list.append(im)
        cornar_list[0].append(x1)
        cornar_list[1].append(x2)
        cornar_list[2].append(y1)
        cornar_list[3].append(y2)

    print("-------------------------")
    print(face_list)
    print("-------------------------")

    return img, face_list, cornar_list


def insert_translated_face(img, cornar_list, ganout_dir, outdir):
    translated_img_list = []
    translated_img_dir = [str(p) for p in Path(ganout_dir).glob("*")]
    for i in range(len(translated_img_dir)):
        print(translated_img_dir[i])
        translated_img = cv2.imread(translated_img_dir[i])
        translated_img_list.append(translated_img)

    for i in range(len(translated_img_list)):
        # 変換された画像を切り抜かれた時の大きさにリサイズ
        translated_img_list[i] = cv2.resize(translated_img_list[i], dsize=(cornar_list[1][i]-cornar_list[0][i], cornar_list[3][i]-cornar_list[2][i]))
        # 顔が切り抜かれた部分に，マスク付きの画像を挿入
        img[cornar_list[2][i]:cornar_list[3][i], cornar_list[0][i]:cornar_list[1][i]] = translated_img_list[i]
        """cv2.imshow("img_test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    out = outdir + "/output.jpg"
    cv2.imwrite(out, img)

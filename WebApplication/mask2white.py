import cv2
import numpy as np
import os
from pathlib import Path  # 要インストール
#from natsort import natsorted  # 要インストール

'''
READ_DIR = "C:\\Users\\Masaharu\\Desktop\\main\\test_images_clipped"  # マスクの輪郭抽出を行う画像のディレクトリpath
SAVE_DIR = "C:\\Users\\Masaharu\\Desktop\\main\\test_images_white"  # 保存先のpath
'''

def synthesize_img(READ_DIR, SAVE_DIR):
    # 画像の読み取り
    image_paths = [str(p) for p in Path(READ_DIR).glob("*")]  # natsortedで番号順に並び替え

    for i in range(len(image_paths)):
        # 画像を読み込む。
        fg_img = cv2.imread(image_paths[i])

        # HSV に変換する。
        hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

        # 2値化する。
        #bin_img = cv2.inRange(hsv, (0, 0, 100), (255, 35, 255))  # 範囲内が白，それ以外が黒くなる
        bin_img = cv2.inRange(hsv, (0, 0, 100), (255, 60, 255))
        """(h(色相),s(明度),v(彩度)) = (0~255, 0~255, 0~255)で表現
            元々のHSVは(0~360°, 0~100%, 0~100%)であるためスケールを合わせる必要あり
            https://pystyle.info/opencv-inrange/
            """
        """cv2.imshow("img_test", bin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        # 輪郭抽出する。
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:  # contoursがemptyじゃない場合
            # 面積が最大の輪郭を取得する
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            # マスク画像を作成する。
            mask = np.zeros_like(bin_img)
            mask = cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
            """cv2.imshow("img_test", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            # 合成する画像の幅、高さ
            w = fg_img.shape[1]
            h = fg_img.shape[0]

            # 合成する領域
            fg_roi = fg_img[:h, :w]

            # 合成する。
            dst = np.where(mask[:h, :w, np.newaxis] == 255, 255, fg_roi)

            # 画像の保存
            frame = os.path.join(SAVE_DIR, "%04d.png" % i)
            # frame = os.path.join(SAVE_DIR, "%04d.jpg" % i)
            cv2.imwrite(frame, dst)
            print("saved as ", "%03d.png" % i)
            # print("saved as ", "%03d.jpg" % i)

        else:
            # 合成する画像の幅、高さ
            w = fg_img.shape[1]
            h = fg_img.shape[0]

            # 合成する領域
            fg_roi = fg_img[:h, :w]

            # 合成する。
            dst = np.where(bin_img[:h, :w, np.newaxis] == 255, 255, fg_roi)

            # 画像の保存
            frame = os.path.join(SAVE_DIR, "%04d.png" % i)
            # frame = os.path.join(SAVE_DIR, "%04d.jpg" % i)
            cv2.imwrite(frame, dst)
            print("saved as ", "%03d.png" % i)
            # print("saved as ", "%03d.jpg" % i)

        #break

        

'''
if __name__ == '__main__':

    # 保存先のディレクトリの作成
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    synthesize_img(READ_DIR, SAVE_DIR)
'''

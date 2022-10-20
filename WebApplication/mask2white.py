import cv2
import numpy as np
import os


def mask2white(face_list, SAVE_DIR):

    for i in range(len(face_list)):

        # HSV に変換する。
        hsv = cv2.cvtColor(face_list[i], cv2.COLOR_BGR2HSV)

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
            w = face_list[i].shape[1]
            h = face_list[i].shape[0]

            # 合成する領域
            fg_roi = face_list[i][:h, :w]

            # 合成する。
            dst = np.where(mask[:h, :w, np.newaxis] == 255, 255, fg_roi)

            # 画像の保存
            frame = os.path.join(SAVE_DIR, "%04d.png" % i)
            cv2.imwrite(frame, dst)

        else:
            # 合成する画像の幅、高さ
            w = face_list[i].shape[1]
            h = face_list[i].shape[0]

            # 合成する領域
            fg_roi = face_list[i][:h, :w]

            # 合成する。
            dst = np.where(bin_img[:h, :w, np.newaxis] == 255, 255, fg_roi)

            # 画像の保存
            frame = os.path.join(SAVE_DIR, "%04d.png" % i)
            cv2.imwrite(frame, dst)

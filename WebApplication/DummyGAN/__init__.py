import cv2
import os
import shutil

# デバック時にCUDAが乗らない人向けのダミー関数(エッジ検出)
def run_dummy(image, path):
    # 出力画像の保存先を削除と生成
    if not os.path.isdir(path):
        os.mkdir(path)
    else: # 前の画像を削除
        shutil.rmtree(path)
        os.mkdir(path)
    # 画像をエッジ検出
    img = cv2.Canny(image, 100, 200)
    # 画像を保存
    # 出力画像を指定フォルダへ保存
    cv2.imwrite(path + '/' + 'output.png', img)
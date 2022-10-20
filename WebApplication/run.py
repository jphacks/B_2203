import os
import cv2
import string
import random
import shutil
import numpy as np
import urllib.parse
from datetime import datetime

from scipy.misc import face
from DummyGAN import run_dummy
from mask2white import synthesize_img
import face_clipper
import dlib
from pathlib import Path  # 要インストール
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# 出力画像の保存先
OUTPUT_DIR = "./templates/images"

INPUT_DIR = "./input_image"

CLIPPED_FACE_DIR = "./clipped_face"

# CycleGAN側で指定している入力画像のフォルダ
GANIN_DIR = "./CycleGAN/test_img"

# CycleGAN側で指定している出力画像のフォルダ
GANOUT_DIR = "./results/CycleGAN/mask2nonmask_results_succes/test_latest/images/fake_image"

# デバック時にCUDAが使えない人向けのダミー用出力フォルダ
DMGANOUT_DIR = "./DummyGAN/result"

# デバックモード切り替え
DEBUG_MODE = False

# 暫定的な画像のID
fid = 0000

# dlibをはじめる
detector = dlib.get_frontal_face_detector()

# Flask 定義
app = Flask( __name__, static_folder="./templates" )

# versition管理
def __version__():
    print(" * Versition:1.01")
    return "1.01"

# 外部プログラムを実行するためのメソッド
def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

# フォルダ内のファイル名listの先頭のみを取得
def file_name(path):
    files = os.listdir(path)
    return files[0]

# ランダムに文字列を与える(静的なシステムへの対応)
def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

# 画像を入力する画面
@app.route('/')
def index():
    # 出力画像の保存先を削除と生成
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        # 出力画像の保存先を削除
        shutil.rmtree(OUTPUT_DIR)
        # 出力画像の保存先を生成
        os.mkdir(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    else:
        # 入力画像の保存先を削除
        shutil.rmtree(INPUT_DIR)
        # 入力画像の保存先を生成
        os.mkdir(INPUT_DIR)
    
    if not os.path.exists(CLIPPED_FACE_DIR):
        os.mkdir(CLIPPED_FACE_DIR)
    else:
        shutil.rmtree(CLIPPED_FACE_DIR)
        # 入力画像の保存先を生成
        os.mkdir(CLIPPED_FACE_DIR)
    
    if not os.path.exists(GANIN_DIR):
        os.mkdir(GANIN_DIR)
    else: # 前の画像を削除
        # 入力画像の保存先を削除
        shutil.rmtree(GANIN_DIR)
        # 入力画像の保存先を生成
        os.mkdir(GANIN_DIR)
    
    if os.path.exists(GANOUT_DIR):
        shutil.rmtree(GANOUT_DIR)

    return render_template('index.html')

# 変換中のページ(ロード画面を作成しても良さそう)
@app.route('/upload', methods=['POST'])
def upload():
    # 警告フォーム化する
    if 'image' not in request.files:
        # ファイルが選択されていない場合
        print('ファイルが選択されていません')
        return redirect('/')

    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 入力された画像をGAN側の指定先フォルダへ保存
        cv2.imwrite(f'{INPUT_DIR}/input.png', img)

        # 実行モードを判定
        if not DEBUG_MODE:
            # 切り出した背景画像， 顔画像, 切り抜いた画像の座標
            back_img, face_list, cornar_list = face_clipper.get_face(INPUT_DIR + '/input.png', CLIPPED_FACE_DIR, detector)
            
            # 
            synthesize_img(CLIPPED_FACE_DIR, GANIN_DIR)

            # 外部からGANを実行(CycleGANプログラムの設計者の思想に合わせた)
            run('python ./CycleGAN/mask_delete.py --dataroot ./CycleGAN/test_img --name ./CycleGAN/mask2nonmask_results_succes --model test --model_suffix _A --num_test 500 --no_dropout')
            translated_img_list = []
            translated_img_dir = [str(p) for p in Path(GANOUT_DIR).glob("*")]
            for i in range(len(translated_img_dir)):
                print(translated_img_dir[i])
                translated_img = cv2.imread(translated_img_dir[i])
                translated_img_list.append(translated_img)
            print(translated_img_list)
            face_clipper.insert_translated_face(back_img, translated_img_list, cornar_list, OUTPUT_DIR)
            
            '''
            # GAN側で指定している出力画像を読み込み
            filename = file_name(GANOUT_DIR) # 出力画像のfile名を取得
            img = cv2.imread(GANOUT_DIR + '/' + filename)
            '''
        else:
            # デバック時にCUDAが使えない人向けのダミー関数
            run_dummy(img, DMGANOUT_DIR)

            # GAN側で指定している出力画像を読み込み
            filename = file_name(DMGANOUT_DIR) # 出力画像のfile名を取得
            img = cv2.imread(DMGANOUT_DIR + '/' + filename)

        # 出力画像を指定フォルダへ保存
        dt_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + random_str(5)
        save_path = os.path.join(OUTPUT_DIR, dt_now + ".png")
        cv2.imwrite(save_path, img)

        print("save", save_path)
    return render_template('result.html', Path = file_name('./templates/images'))

# 変換結果 ページ
@app.route('/result', methods=['POST'])
def result():
    return redirect('result.html')

# 使い方 ページ
@app.route('/manual')
def manual():
    return render_template('manual.html')

'''実装迷い段階　画像ダウンロードをflask側で処理するためのコード
@app.route('/')
def download_image():
    return send_file('output.png', minetype = 'png')
'''

# メイン関数
if __name__ == '__main__':
    # Serverのバージョンを表示
    __version__()
    # モード指定
    app.debug = True
    # 実行
    # app.run()
    app.run(host='0.0.0.0', port=8080)
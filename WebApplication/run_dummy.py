import os
import cv2
import string
import random
import shutil
import numpy as np
from datetime import datetime
from DummyGAN import run_dummy
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# 出力画像の保存先
OUTPUT_DIR = "./templates/images"

# 入力画像の保存先
INPUT_DIR = "./input_image"

# CycleGAN側で指定している入力画像のフォルダ
GANIN_DIR = "./CycleGAN/test_img"

# CycleGAN側で指定している出力画像のフォルダ
GANOUT_DIR = "./results/CycleGAN/mask2nonmask_results_succes/test_latest/images/fake_image"

# デバック時にCUDAが使えない人向けのダミー用出力フォルダ
DMGANOUT_DIR = "./DummyGAN/result"

# Flask 定義
app = Flask( __name__, static_folder="./templates" )

# versition管理
def __version__():
    print(" * Versition:1.01")
    return "1.01a"

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

        # 入力された画像を入力の指定先フォルダへ保存
        cv2.imwrite(f'{INPUT_DIR}/input.png', img)
            
        # デバック時にCUDAが使えない人向けのダミー関数
        run_dummy(img, DMGANOUT_DIR)

        # GAN側で指定している出力画像を読み込み
        filename = file_name(DMGANOUT_DIR) # 出力画像のfile名を取得
        img = cv2.imread(DMGANOUT_DIR + '/' + filename)

        # 出力画像を指定フォルダへ保存
        dt_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + random_str(5)
        save_path = os.path.join(OUTPUT_DIR, dt_now + ".png")
        cv2.imwrite(save_path, img)

        #print("save", save_path)
    return render_template('result.html', Path = file_name('./templates/images'))

# 変換結果 ページ
@app.route('/result', methods=['POST'])
def result():
    return redirect('result.html')

# 使い方 ページ
@app.route('/manual')
def manual():
    return render_template('manual.html')

# メイン関数
if __name__ == '__main__':
    # Serverのバージョンを表示
    __version__()
    # モード指定
    app.debug = True
    # 実行
    app.run(host='0.0.0.0', port=8080)
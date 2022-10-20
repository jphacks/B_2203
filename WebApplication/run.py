import os
import cv2
import dlib
import shutil
import numpy as np
import urllib.parse
import face_clipper
from scipy.misc import face
from DummyGAN import run_dummy
from mask2white import mask2white
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# 入力画像の保存先
INPUT_DIR = "./input_image"
# 出力画像の保存先
OUTPUT_DIR = "./templates/images"
# CycleGAN側で指定している入力画像のフォルダ
GANIN_DIR = "./CycleGAN/test_img"
# CycleGAN側で指定している出力画像のフォルダ
GANOUT_DIR = "./results/CycleGAN/mask2nonmask_results_succes/test_latest/images/fake_image"

# dlib(顔検出)のインスタンス変数
detector = dlib.get_frontal_face_detector()

# Flask 定義
app = Flask( __name__, static_folder="./templates" )

# versition管理
def __version__():
    print(" * Versition:1.02")
    return "1.02"

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

# 画像を入力する画面
@app.route('/')
def index():
    # 入力画像の保存先を生成・削除
    if not os.path.exists(INPUT_DIR):
        os.mkdir(INPUT_DIR)
    else:
        # 入力画像の保存先を削除
        shutil.rmtree(INPUT_DIR)
        # 入力画像の保存先を生成
        os.mkdir(INPUT_DIR)

    # 出力画像の保存先を生成・削除
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        # 出力画像の保存先を削除
        shutil.rmtree(OUTPUT_DIR)
        # 出力画像の保存先を生成
        os.mkdir(OUTPUT_DIR)

    # GAN入力画像の保存先を生成・削除
    if not os.path.exists(GANIN_DIR):
        os.mkdir(GANIN_DIR)
    else: # 前の画像を削除
        # 入力画像の保存先を削除
        shutil.rmtree(GANIN_DIR)
        # 入力画像の保存先を生成
        os.mkdir(GANIN_DIR)
    
    # GAN出力画像の保存先を削除
    if os.path.exists(GANOUT_DIR):
        shutil.rmtree(GANOUT_DIR)

    return render_template('index.html')

# 画像変換処理 ページ
@app.route('/upload', methods=['POST'])
def upload():
    # 警告フォーム化する
    if 'image' not in request.files:
        # ファイルが選択されていない場合
        print('ファイルが選択されていません')
        return redirect('/')
    
    # imageファイルのリクエストが存在する場合
    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 入力された画像を入力の指定先フォルダへ保存
        cv2.imwrite(f'{INPUT_DIR}/input.png', img)

        # 顔の切り抜き(返り値：背景画像, 切り抜いた顔画像, 切り抜いた画像の座標)
        back_img, face_list, cornar_list = face_clipper.get_face(INPUT_DIR + '/input.png', detector)
            
        # フェイスマスクの白色化処理（事前処理）
        mask2white(face_list, GANIN_DIR)

        # 外部からGANを実行(CycleGANプログラムの設計者の思想に合わせた)
        run('python ./CycleGAN/mask_delete.py --dataroot ./CycleGAN/test_img --name ./CycleGAN/mask2nonmask_results_succes --model test --model_suffix _A --num_test 500 --no_dropout')

        # CycleGANで変換した画像の取得
        face_clipper.insert_translated_face(back_img, cornar_list, GANOUT_DIR, OUTPUT_DIR)

        return render_template('result.html', Path = file_name('./templates/images'))
    else:
        return render_template('index.html')

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
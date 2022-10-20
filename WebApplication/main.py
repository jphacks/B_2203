# %%
import os
import cv2
import string
import random
import shutil
import numpy as np
from datetime import datetime
from DummyGAN import run_dummy
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# 外部プログラムを実行するためのメソッド
def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

if __name__ == '__main__':
    # 外部からGANを実行(CycleGANプログラムの設計者の思想に合わせた)
    run('python ./CycleGAN/mask_delete.py --dataroot ./CycleGAN/test_img --name ./CycleGAN/mask2nonmask_results_succes --model test --model_suffix _A --num_test 500 --no_dropout')
# %%

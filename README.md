# **SUGAO**(**S**aving **U**nmasking face **G**enerated by **A**I **O**peration)

![](/figs/top_image2.svg)

## 製品概要

### 背景(製品開発のきっかけ，課題等）

<!--担当：中野-->

皆様，一度はこのように思った経験ありませんか？

「あれ，この人のマスクを外したときの顔ってこんな感じなんだ」

コロナ禍でマスクの着用が定着し，マスク顔がデフォルトとなっているこの時代，マスクを外したときの顔に新鮮さを覚える人は少なくないと思います．マスクによって口元の表情が読み取りにくくなり，感染防止のために写真を撮る機会も減少しました．

私たちはこの課題に対して，AI 技術を活用し，「日常からマスクを払拭できるような面白いアプリを開発したい！！」と考えました．

私たちの開発したアプリケーションが写真を撮る機会を促進し，皆様の『新たな思い出の 1 ページ！！』となれば幸いです．

### 説明動画（YouTubeへのリンク）

[![](/figs/demo_link.svg)](https://youtu.be/JpdHhGm5G4c)

### 製品説明（具体的な製品の説明）

<!--担当：三好-->

本製品は，フェイスマスク着用時の写真撮影を支援する目的とした，フェイスマスクを撮影された写真から AI による素顔の予測を行う Web アプリケーションである．

具体的には，インターネット上に公開された Web ページにデバイス（スマートフォン・パソコンなど）からアクセスし，フェイスマスクを着用した写真を選択する．選択された画像は事前処理を行い画像生成系 AI"CycleGAN"によってフェイスマスク部分の素顔を予測し，予測された画像（フェイスマスクが除去された画像）が出力される．また，出力された結果画像を保存することも可能である．

#### 特長

<!--担当：森下-->

##### 1. マスクを外さずにマスクを外した写真を撮れる

##### 2. スマホで撮影した写真をその場で変換できる

##### 3. Web アプリなのでインストールの必要がない

#### システム概要

本製品は以下の図のようなシステム構成を設計し，実装した．

![System](/figs/system.svg)

#### 使用方法

本製品の使用方法は以下の手順の通りである．<br>
① デバイス（スマートフォン・パソコンなど）から Web ページにアクセス(スマートフォンの場合 Chrome 推奨)<br>
② "ファイル選択"ボタンを押し，デバイス内に保存された画像を選択<br>
③ "変換"ボタンを押す<br>
④ Web ページ上に結果画像が表示される<br>
⑤ 表示された画像を任意で保存(保存機能あり)<br>

![Users](/figs/usermanual.svg)

#### 画像変換の仕組み

![image](/figs/system_of_covert_img.png)

#### CycleGAN（画像生成 AI）の学習方法

今回 CycleGAN の学習には，CelebA と呼ばれる顔画像のデータセットを用いた．このデータセットをもとにマスクをつけた人の画像（収集方法は「注力したこと」に記載）とマスクを着けていない人の画像を約 3,000 枚ずつ用いて学習を行った（学習時間：2 日半程度）．

### 解決出来ること

- 写真を撮る機会を促進できる
- 日常の写真からマスクを取り除くことができる

### 注力したこと（こだわり等）

#### 1. Cycle GAN を学習させるために必要なマスクをつけた人の画像集め

課題として，マスクを着けていな人の画像データを収取することが困難であった．
そこで，Cycle GAN が双方向変換(mask → nonmask, nonmask → mask)であることを利用した．数百枚程度の少ないデータ学習させた nonmask→mask 変換の GAN を用いることでマスクを着けていない人の画像から，マスクをつけた人の画像データを 3,000 枚程度入手することに成功した．

#### 2.ウォーターフォール開発による個性を活かした開発

本システム開発ではウォーターフォール開発を採用しており，チームメンバの個々の専門性を活かした役割分担を行うことで開発効率の向上を図った．

#### 3.CSS を用いたデバイス間のデザイン切り替え

利用者のデバイスによって Web ページのレイアウトが崩れてしまうことを防止するため，画面サイズに応じて適切に変化するよう CSS を記述した．

### 今後の展望

- 顔の検出では dlib を使っているため，マスクを着けている状態では顔が認識されにくい．今後はマスクをつけた状態でも顔の検出がされるような CNN を用いる（学習させる）ことで，検出精度を高めていきたい．

- 画像の事前処理は，HSV による閾値を用いた OpenCV の輪郭抽出で行っているため，今のところ白いマスク以外には非対応であり，背景が白いと画像変換がうまくいかない場合が多い．そのため，Mask_RCNN(画像から segmentation mask を生成できるモデル)等のモデルを fine tuning することにより，どんな色のマスクや背景にも対応したマスクの輪郭抽出を行えるようにしたい．

- 時間の都合上，数千枚程度の画像を用いた CycleGAN の学習しか行えなかったが，画像データを可能な限り増やし，よりリアルな輪郭を生成できるモデルの学習を行いたい．

- 現状多重アクセスへの対策が未対応である．ログイン機能などを設けることにより，複数のアクセスに対応できるような工夫を行いたい．

## 開発技術

### 活用した技術

<!--担当：全員-->

#### API・データ

- CelebA(人の顔画像のデータセット)

#### フレームワーク・ライブラリ・モジュール

<!--担当：森下-->

##### フロントエンド

- HTML
- CSS

<!--担当：三好-->

##### バックエンド

- Python
- Flask
- ngrok

<!--担当：中野-->

##### ライブラリ・学習モデル

- Dlib
- OpenCV
- Pytorch
- Cycle GAN

#### デバイス

- PC
- スマートフォン
- タブレット

### 独自技術

#### ハッカソンで開発した独自機能・技術

##### 限られた時間における効率的な CycleGAN の学習方法の開発

- CycleGAN が画像の変換を学習する過程で，生成された画像からマスク下の顔のランドマーク検出を行い，その結果を重みとして Loss に乗算することで，顔の輪郭生成に特化したモデルを作成した．これにより数千枚程度の画像データで顔の輪郭を生成できるモデルの学習が可能となった．

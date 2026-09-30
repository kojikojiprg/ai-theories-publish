---
title: "ViT と画像パッチ埋め込み / Vision Transformer and Patch Embedding(実装・実験編 1/4)"
---

この記事は後編(実装・実験編 1/4)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-theory)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/019_vision_transformer-practice-2)。

## 4. 実装方針 / Implementation Policy

**`src/`に切り出す(スクラッチ実装、本トピックで新規作成)**:

- `src/layers/patch_embedding.py`: `PatchEmbedding`(`nn.Conv2d(kernel_size=P, stride=P)`で実装し、平坦化 + 線形射影の形の重み
  $E$ を返す`linear_projection_weight()`を持つ)、`extract_patches()`(パッチの平坦化)、`permute_patches()`(パッチ単位の並べ替え)、
  `build_2d_sinusoidal_position_embedding()`(Beyer et al. の実装に従う)。
- `src/models/vit.py`: `VisionTransformer`。002 の`EncoderBlock`を`norm_first=True`で積層し、活性化は 004 の`gelu_exact`、
  正規化は`LayerNormalization`とする(原論文どおり)。位置埋め込みの方式を`"learned"`/`"sinusoidal_2d"`/`"none"`で切り替える。
  [CLS] トークンと最終正規化層を持ち、`forward(..., return_attention_weights=True)`で各層の注意の重みを返す。
  非埋め込みパラメータ数を数える`count_vit_non_embedding_parameters()`。
- `src/data/image.py`: CIFAR-10 の取得`load_cifar10()`(torchvision、キャッシュは`.cache/cifar10/`)、クラス均衡な検証集合の
  切り出し`split_class_balanced_validation()`、クラス均衡で入れ子の部分集合`make_nested_class_balanced_subsets()`、エポックの境界を
  またいで一定の大きさのバッチを切り出す`EpochShuffledBatchSampler`、GPU 上のデータ拡張`random_crop_and_flip()`。
- `src/training/classification.py`: 分類の学習ループ`train_image_classifier()`と評価`evaluate_image_classifier()`。007 の
  `AdamW`・`compute_warmup_cosine_learning_rate()`・gradient clipping、011 の`DynamicLossScaler`を使う。データの順序・データ拡張の
  乱数(CPU の生成器)と初期化の乱数を別々の生成器にする。

**ノートブック内に直接書く(019 固有)**:

- 比較対象の CNN: He et al. [7] の 4.2 節の CIFAR-10 用の構成(層数 $6n + 2$ で $n = 9$ の ResNet-56)を、ノートブック内に
  スクラッチ実装する(`CifarResNet`)。比較対象であって主題ではないので、共通モジュールにはしない。1 枚あたりの積和演算の数の
  閉形式が、モジュールのフックで数えた値と一致することを 5.4 節で確かめる。
- 各実験の条件の定義・学習率の較正・判定・可視化・スケーリングの計測、計算量の閉形式。

**既存モジュールの変更**: なし(`EncoderBlock`・`LayerNormalization`・`gelu_exact`・`AdamW`・
`compute_warmup_cosine_learning_rate()`・`DynamicLossScaler`はそのまま使う)。`DynamicLossScaler.unscale_gradients()`の代わりに、
同じ除算をまとめて行う`unscale_gradients_and_compute_norm()`を新しいモジュールに置いた理由は、そのモジュールの docstring と
5.4 節に記す(結果が bit 単位で一致することを確かめる)。

**アップロード方針**: Hugging Face Hub へのアップロードはしない。学習したモデルは条件比較のためのもので、後続のトピックの入力に
ならない(020 は画像とテキストの encoder を共同で学習するので、019 から再利用するのはコードのみ)。

**生成物の置き場所**:

| 生成物 | 置き場所 |
|---|---|
| CIFAR-10(torchvision が取得するアーカイブと展開したファイル) | `.cache/cifar10/`(入力から決定的に再生成できる。セッション内でのみ再利用する) |
| 検証集合・部分集合の添字、GPU 上の画像のテンソル | メモリ上のみ(固定のシードから決定的に再生成できる) |
| 学習したモデル | メモリ上のみ(評価の直後に破棄する) |
| 判定の記録 | 判定と前提条件を計算した各セルの出力 |

外部から取得するのは CIFAR-10 のみである(torchvision の`CIFAR10(download=True)`が
`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`から取得し、MD5 を照合する)。**torchvision の用途は CIFAR-10 の取得のみ**
であり、モデルの定義や学習済みの重みには使わない(CNN はノートブック内のスクラッチ実装)。



## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/05_vision_language/019_vision_transformer.ipynb

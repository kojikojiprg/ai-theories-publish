---
title: "トークナイザと部分語分割(実装・実験編 2/3)"
---

この記事は後編(実装・実験編 2/3)です。前の内容は [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer-practice-1)。続きは [こちら](https://zenn.dev/kojikojiprg/books/ai-theories-roadmap/viewer/005_tokenizer-practice-3)。

## 6. 実験 / Experiments

以下の実験はすべて Google Colab 無料枠(CPU で十分実行できる規模。学習を伴う実験は 001〜004 と異なり GPU を必要としない)で実行可能な規模で構成する。

**再現性についての注記**: 001〜004 の実験は、学習過程に乱数(重みの初期化、ミニバッチのサンプリングなど)を含むため、複数シードでの反復によって結果の安定性を確認していた。本トピックの BPE の学習・符号化、および Unigram 言語モデルの Viterbi 最尤分割は、乱数を一切使わない **決定的(deterministic)** な処理である(3.2 節のタイブレーク規則、3.5 節の動的計画法)。したがって複数シードでの反復は行わない。

ただし、アルゴリズムが決定的であっても **入力コーパスが実行のたびに変化すれば結果は再現しない**。したがって、本トピックの再現性は入力・アルゴリズムの両面から成り立つ。

- **アルゴリズム面**: タイブレーク規則を固定していること(3.2 節: 頻度同点時は対のタプルの辞書式順序で一意に決定する)。この規則を固定しない実装では、同じ入力コーパスからでも実行のたびに異なるマージ順序が得られうる。
- **入力面**: 英語(Tiny Shakespeare、`load_tiny_shakespeare`)は固定の配布ファイルであるため不変である。日本語(`load_japanese_corpus`)は、5.2 節で述べた通りタイトルとリビジョン ID の両方を固定して取得するため、記事本文が今後編集されても本ノートブックが取得するコーパスは変わらない。**コード(本リポジトリ自身の`src/`、`load_code_corpus`)はこの 2 つとは異なり不変ではない**。`src/`は 006 以降のトピック追加のたびに拡張され、`load_code_corpus`はファイルをパスの辞書順に連結するため、新しいファイルが加わると`TRAIN_SIZE`で切り出す先頭部分の内容が将来のコミットで変化しうる。コードドメインの役割は「空白・記号が多い第 3 のドメイン」として英語・日本語との質的な対比を得ることにあり、絶対値の長期的な再現を目的としていないため、この不変性の欠如は許容する。

以上により、5 節の実装(タイブレーク規則の固定)と、英語・日本語の入力の固定(コードは上記の理由により対象外)を合わせて、本トピックの実験の大部分が決定的であるといえる。

以下、実験ごとに検証事項と判定基準を先に宣言してから実行する(004 と同じ方式)。

### 実験1: BPE のマージ過程の観察

英語コーパスの一部(数 KB)で語彙サイズ 300 程度の BPE を学習し、最初の 30 個程度のマージ規則を表示する。図2(3.2 節)で示した過程が、実際のコーパスでどのように起こるかを定性的に観察する。

**観察すること**: マージ規則の前半が高頻度の 2 文字組、後半にかけて単語・接尾辞相当のシンボルに近づく傾向が見られるか。

本実験は定性的な観察のみを目的とし、**判定基準を伴う定量的な検証は行わない**。


```python
tiny_corpus = corpus_en[:5_000]
print(f"学習コーパス: {len(tiny_corpus)} 文字")

tiny_bpe = learn_bpe(tiny_corpus, vocab_size=300, byte_level=False, chunk_split_mode="whitespace")
print(f"達成した語彙サイズ: {len(tiny_bpe.vocab)}")
print(f"マージ規則の総数: {len(tiny_bpe.merges)}")
print()
print("最初の 30 個のマージ規則(学習順):")
for i, (s1, s2) in enumerate(tiny_bpe.merges[:30]):
    print(f"  {i + 1:2d}: {s1!r:>8} + {s2!r:<8} -> {s1 + s2!r}")
```

    学習コーパス: 5000 文字
    達成した語彙サイズ: 300
    マージ規則の総数: 247
    
    最初の 30 個のマージ規則(学習順):
       1:      ' ' + 't'      -> ' t'
       2:      'h' + 'e'      -> 'he'
       3:      ' ' + 'a'      -> ' a'
       4:      'o' + 'u'      -> 'ou'
       5:      'i' + 't'      -> 'it'
       6:      'e' + 'n'      -> 'en'
       7:      ' ' + 'w'      -> ' w'
       8:      's' + 't'      -> 'st'
       9:      'r' + 'e'      -> 're'
      10:     ' t' + 'he'     -> ' the'
      11:      'i' + 'n'      -> 'in'
      12:      'l' + 'l'      -> 'll'
      13:      ' ' + 'm'      -> ' m'
      14:      'a' + 't'      -> 'at'
      15:      ' ' + 's'      -> ' s'
      16:     '\n' + '\n'     -> '\n\n'
      17:      ' ' + 'c'      -> ' c'
      18:      'o' + 'n'      -> 'on'
      19:      ' ' + 'b'      -> ' b'
      20:      ' ' + 'y'      -> ' y'
      21:      'o' + 'r'      -> 'or'
      22:     ' y' + 'ou'     -> ' you'
      23:      'e' + 'r'      -> 'er'
      24:      'i' + 'r'      -> 'ir'
      25:      'i' + 's'      -> 'is'
      26:      ' ' + 'C'      -> ' C'
      27:      ' ' + 'f'      -> ' f'
      28:      ' ' + 'p'      -> ' p'
      29:     'it' + 'i'      -> 'iti'
      30:      'u' + 's'      -> 'us'


**観察の指針**: マージ規則の前半は、頻出する 2 文字の組(`th`、`he`、`in` など英語で頻出するバイグラム)が中心になり、後半にかけて `the`、`and` のような頻出単語全体、あるいは `ing`、`ed` のような接尾辞に相当するシンボルが現れてくるはずである。実行後、実際に得られたマージ規則を確認し、この傾向が見られたかどうかを 7 節で述べる。

### 実験2: 語彙サイズと fertility の関係

3 ドメイン(英語・日本語・コード)× 語彙サイズ $\{512, 1024, 2048, 4096, 8192\}$ でバイトレベル BPE(`byte_level=True`、`chunk_split_mode="whitespace"`)を学習し、ホールドアウト部分の fertility を測定する。

**検証すること**: 圧縮効果は語彙サイズに対して収穫逓減(diminishing returns)する。

**判定基準**: 語彙サイズを 2 倍にしたときの fertility 減少幅($\mathrm{fertility}(V) - \mathrm{fertility}(2V)$)が、掃引の全区間で単調に縮小する。

**注記**: コーパスから作れるユニークなペアが尽きると、目標語彙サイズに達する前に学習が終了することがある(3.2 節)。以下の実装では、要求した語彙サイズ(`requested`)と実際に達成した語彙サイズ(`achieved`)の両方を記録し、両者が乖離した場合はそれ自体を結果として報告する。


```python
VOCAB_SIZES = [512, 1024, 2048, 4096, 8192]

# {domain: {requested_vocab_size: (achieved_vocab_size, fertility)}}
experiment2_results: dict[str, dict[int, tuple[int, float]]] = {}
bpe_tokenizers: dict[
    tuple[str, int], BPETokenizer
] = {}  # (domain, vocab_size) -> BPETokenizer、後続の実験で再利用する

t_total = time.time()
for domain, train_text in train_texts.items():
    experiment2_results[domain] = {}
    holdout = holdout_texts[domain]
    for vocab_size in VOCAB_SIZES:
        t0 = time.time()
        tokenizer = learn_bpe(
            train_text, vocab_size=vocab_size, byte_level=True, chunk_split_mode="whitespace"
        )
        bpe_tokenizers[(domain, vocab_size)] = tokenizer
        tokens = tokenizer.encode(holdout)
        fertility = compute_fertility(len(tokens), len(holdout))
        experiment2_results[domain][vocab_size] = (len(tokenizer.vocab), fertility)
        print(
            f"{domain:>8} requested={vocab_size:5d} achieved={len(tokenizer.vocab):5d} "
            f"fertility={fertility:.4f} ({time.time() - t0:.1f}s)"
        )
print(f"合計実行時間: {time.time() - t_total:.1f} 秒")
```

     English requested=  512 achieved=  512 fertility=0.4586 (0.4s)
     English requested= 1024 achieved= 1024 fertility=0.3415 (0.6s)
     English requested= 2048 achieved= 2048 fertility=0.2847 (1.3s)
     English requested= 4096 achieved= 4096 fertility=0.2631 (2.5s)
     English requested= 8192 achieved= 6554 fertility=0.2503 (3.0s)
    Japanese requested=  512 achieved=  512 fertility=1.4069 (21.4s)
    Japanese requested= 1024 achieved= 1024 fertility=1.0352 (31.0s)
    Japanese requested= 2048 achieved= 2048 fertility=0.8507 (41.6s)
    Japanese requested= 4096 achieved= 4096 fertility=0.7275 (52.4s)
    Japanese requested= 8192 achieved= 8192 fertility=0.6682 (72.5s)
        Code requested=  512 achieved=  512 fertility=0.6213 (1.8s)
        Code requested= 1024 achieved= 1024 fertility=0.4524 (2.3s)
        Code requested= 2048 achieved= 2048 fertility=0.3452 (3.9s)
        Code requested= 4096 achieved= 4096 fertility=0.2937 (8.3s)
        Code requested= 8192 achieved= 8192 fertility=0.2723 (12.1s)
    合計実行時間: 255.2 秒



```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for domain, results in experiment2_results.items():
    vocab_sizes = sorted(results.keys())
    fertilities = [results[v][1] for v in vocab_sizes]
    ax.plot(vocab_sizes, fertilities, marker="o", label=domain)
ax.set_xscale("log", base=2)
ax.set_xlabel("Vocabulary size V (log scale)")
ax.set_ylabel("Fertility (tokens / char)")
ax.set_title("Fertility vs. vocabulary size by domain")
ax.grid(alpha=0.3)
ax.legend()
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_30_0.png)
    



```python
# 実験2の判定: 語彙サイズを2倍にしたときの fertility 減少幅が単調に縮小するか
for domain, results in experiment2_results.items():
    vocab_sizes = sorted(results.keys())
    fertilities = [results[v][1] for v in vocab_sizes]
    deltas = [fertilities[i] - fertilities[i + 1] for i in range(len(fertilities) - 1)]
    monotonic = all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))
    print(f"{domain:>8}: fertility={[f'{f:.4f}' for f in fertilities]}")
    print(f"{'':>8}  減少幅={[f'{d:.4f}' for d in deltas]}  単調縮小={monotonic}")
```

     English: fertility=['0.4586', '0.3415', '0.2847', '0.2631', '0.2503']
              減少幅=['0.1171', '0.0568', '0.0216', '0.0127']  単調縮小=True
    Japanese: fertility=['1.4069', '1.0352', '0.8507', '0.7275', '0.6682']
              減少幅=['0.3717', '0.1845', '0.1232', '0.0593']  単調縮小=True
        Code: fertility=['0.6213', '0.4524', '0.3452', '0.2937', '0.2723']
              減少幅=['0.1689', '0.1072', '0.0515', '0.0213']  単調縮小=True


### 実験3: 初期語彙方式と未知語

学習コーパスに存在しない文字(絵文字、稀な漢字など)を含むホールドアウト文字列を用意し、`byte_level`の真偽で未知語率を比較する。

**検証すること**: バイトレベル初期語彙は未知語を原理的に発生させない。

**判定基準**: バイトレベルで未知語率が厳密に 0、文字レベル初期語彙で厳密に正。


```python
# 実験3: 未知語(Out-of-Vocabulary)の検証
# 学習コーパス(日本語ドメイン)に存在しない文字(絵文字・極めて稀な漢字)を含む文字列を用意する。
# 日本語コーパスは Wikipedia から取得するため内容が変わりうる。固定の候補文字が
# 偶然コーパスに出現していても実験全体が停止しないよう、候補プールから学習コーパスに
# 存在しない文字だけを動的に選ぶ(文中の一般的な単語は当然コーパスにも出現するため、
# 「学習コーパスに存在しない」ことを検証するのは絵文字・稀な漢字の部分のみに限定する)。
candidate_pool = "😀🎉🚀🌸龘齉靐彟龗鱻麤鸞爨癵驫"
truly_novel_chars = "".join(ch for ch in candidate_pool if ch not in train_texts["Japanese"])
assert len(truly_novel_chars) >= 4, "学習コーパスに存在しない文字が十分に確保できない"
novel_chars_text = f"これは{truly_novel_chars}絶対に出現しない文字列だ。"
print(f"検証に使う未出現文字: {truly_novel_chars}")

char_level_tokenizer = learn_bpe(
    train_texts["Japanese"], vocab_size=2048, byte_level=False, chunk_split_mode="whitespace"
)
byte_level_tokenizer = bpe_tokenizers[("Japanese", 2048)]  # 実験2で学習済み

char_tokens = char_level_tokenizer.encode(novel_chars_text)
byte_tokens = byte_level_tokenizer.encode(novel_chars_text)

char_unk_rate = compute_unknown_rate(char_tokens)
byte_unk_rate = compute_unknown_rate(byte_tokens)

print(f"文字レベル初期語彙: 未知語率={char_unk_rate:.3f}  トークン例={char_tokens[:12]}")
print(f"バイトレベル初期語彙: 未知語率={byte_unk_rate:.3f}  トークン数={len(byte_tokens)}")
print(
    f"実験3判定: バイトレベル未知語率==0 -> {byte_unk_rate == 0.0}, "
    f"文字レベル未知語率>0 -> {char_unk_rate > 0.0}"
)
```

    検証に使う未出現文字: 😀🎉🚀🌸龘齉靐彟龗鱻麤鸞爨癵驫
    文字レベル初期語彙: 未知語率=0.536  トークン例=['これは', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>']
    バイトレベル初期語彙: 未知語率=0.000  トークン数=60
    実験3判定: バイトレベル未知語率==0 -> True, 文字レベル未知語率>0 -> True


### 実験4: 初期語彙方式と語彙サイズの相互作用(日本語)

3.3 節で述べた通り、日本語のユニーク文字数は目標語彙サイズ(512 や 1024 など)を上回りうる。この場合、文字レベル初期語彙は最初からその文字数だけで目標語彙サイズを超えてしまい、**マージが一切発生しない**(`achieved vocab`が学習コーパスのユニーク文字数とほぼ一致し、目標語彙サイズを上回る)。この区間でバイトレベルと文字レベルを比較すると、両者は **異なる語彙予算での比較** になり(バイトレベルは要求どおりの語彙サイズで学習される一方、文字レベルはそれより大きい実効語彙サイズで学習される)、「バイトレベルが不利なのは UTF-8 の 3 バイト表現のコストゆえか、単に語彙予算が違うからか」を分離できない。

そこで、実験2 と同じ`VOCAB_SIZES`(512, 1024, 2048, 4096, 8192)で **日本語ドメインの文字レベル BPE を掃引** し、実験2 で得た日本語のバイトレベル fertility 曲線と重ね描きする。語彙サイズを掃引することで、文字レベルの`achieved vocab`が要求値と一致し始める点(= ユニーク文字数を超え、マージが発生し始める点)を境に、比較可能な区間とそうでない区間を視覚的に区別する。

**検証すること**: バイトレベル初期語彙と文字レベル初期語彙のどちらが日本語で有利かは、語彙サイズに依存する。

**判定基準**: 語彙サイズを掃引したとき、日本語の fertility の大小関係(バイトレベル 対 文字レベル)が語彙サイズによって入れ替わる区間が存在するか。入れ替わりが観測されれば「どちらが優れているか」は語彙サイズに依存するトレードオフであり、全域で一方が有利であればその旨をそのまま報告する。

マージが発生しない区間(文字レベルの`achieved vocab`が要求語彙サイズを上回っている区間)の比較は、語彙予算・その使われ方のいずれも揃っておらず、UTF-8 の 3 バイト表現のコストの有無を分離できないため **解釈不能** である。要求語彙サイズがユニーク文字数を上回り、両者ともマージ予算を持つ比較可能な区間で初めて、トレードオフの検証として意味を持つ。**「A は B より優れている」という形ではなく、どのようなトレードオフが生じるかを説明する形で 7 節の考察を書く。**

**fertility 比較の測定上の非対称性についての注記**: fertility による比較には測定上の非対称性がある。文字レベル初期語彙は、ホールドアウトに学習コーパス未出現の文字が含まれていても、それを未知語(Out-of-Vocabulary)トークン **1 個** として出力する(実験3 参照)。一方バイトレベル初期語彙は未知語を発生させないため(実験3)、同じ文字を UTF-8 の 2〜3 バイト分のトークンで表現する。つまり文字レベルは「表現できない文字」を最も安いコストで処理していることになり、その分だけ fertility が構造的に有利に出うる。**この偏りの大きさを評価するため、以下のセルでは語彙サイズごとに文字レベルのホールドアウト未知語率も測定する**(判定基準はここまでに述べたものから変更しない)。未知語率が無視できる大きさであれば fertility の比較はそのまま解釈でき、無視できない大きさであればその分を割り引いて解釈する必要がある。

**未知語を除去した対照条件について**: 未知語率の測定だけでは、それが fertility の差にどれだけ寄与しているかは分からない。そこで、文字レベルの語彙に単一文字として含まれない文字をホールドアウトから除去した対照条件でも fertility を測定する。この条件では文字レベルは未知語を出さないため、バイトレベルとの差は未知語の扱いに起因しない、符号化効率そのものの差になる。元のホールドアウトでの差(バイトレベル − 文字レベル)が、この対照条件での差と比べて大きい場合、その差分は未知語の扱いに起因する見かけのものであることになる。**この対照比較も判定基準を変更するものではなく、判定結果を機構レベルで説明するための追加測定である。**

**語彙予算の配分についての注記**: バイトレベル初期語彙は 256 バイトから出発するため、UTF-8 で 3 バイトを占める日本語の文字を単一トークンにするには、1 文字あたり 2 回のマージを要する(3.3 節)。したがって、バイトレベルの語彙の相当部分は **文字の再構成** に費やされており、真の部分語(subword)の学習に使える予算はその分少なくなる。この予算配分を直接測定するため、以下のセルでは各語彙サイズにおけるバイトレベル語彙の文字被覆率(学習コーパスのユニーク文字のうち、単一トークンとして表現できるものの割合)と、文字の再構成に費やされたマージ回数の概算を診断量として測定する。**この診断量も判定基準の一部ではなく、判定結果を機構として説明するためのものである。**


```python
# 実験4: 日本語ドメインで文字レベル BPE を実験2 と同じ語彙サイズで掃引する。
# achieved_vocab・fertility に加え、ホールドアウトの未知語率も記録する
# (文字レベル初期語彙が未知語を 1 トークンで処理することによる fertility への
# 影響を評価するため。上の宣言セルの「測定上の非対称性についての注記」を参照)。
#
# 対照条件: 文字レベルの語彙に単一文字として含まれない文字をホールドアウトから
# 除去した文字列でも、文字レベル・バイトレベル(実験2で学習済み)の両方で
# fertility を測定する。この条件では文字レベルは未知語を出さないため、両者の差は
# 未知語の扱いに起因しない、符号化効率そのものの差になる
# (宣言セルの「未知語を除去した対照条件について」を参照)。
#
# ASCII 文字はバイトレベル初期語彙の基底シンボル(256 バイト)にそのまま含まれる
# ため、マージなしで常に被覆される。学習コーパスに出現する ASCII 文字の集合は
# 語彙サイズによらず一定であり、これを基準に「非 ASCII 文字の被覆数」を
# 逆算できる(宣言セルの「予算配分」についての注記を参照)。
japanese_train_ascii_chars = {ch for ch in set(train_texts["Japanese"]) if ord(ch) < 128}

experiment4_results: dict[int, tuple[int, float, float]] = {}
# {requested_vocab_size: (achieved_vocab_size, fertility, unknown_rate)}
experiment4_unk_removed_results: dict[int, tuple[int, float, float]] = {}
# {requested_vocab_size: (removed_char_count, char_fertility_filtered, byte_fertility_filtered)}
experiment4_coverage_results: dict[int, tuple[int, int, float, int, float]] = {}
# {requested_vocab_size: (num_covered, num_total, coverage,
#                          estimated_reconstruction_merges, reconstruction_merge_share)}
holdout = holdout_texts["Japanese"]
for vocab_size in VOCAB_SIZES:
    tokenizer = learn_bpe(
        train_texts["Japanese"],
        vocab_size=vocab_size,
        byte_level=False,
        chunk_split_mode="whitespace",
    )
    byte_tokenizer = bpe_tokenizers[("Japanese", vocab_size)]  # 実験2で学習済み

    tokens = tokenizer.encode(holdout)
    fertility = compute_fertility(len(tokens), len(holdout))
    unknown_rate = compute_unknown_rate(tokens)
    experiment4_results[vocab_size] = (len(tokenizer.vocab), fertility, unknown_rate)
    print(
        f"requested={vocab_size:5d} achieved={len(tokenizer.vocab):5d} "
        f"fertility={fertility:.4f}  未知語率={unknown_rate:.4f}"
    )

    # 文字被覆率(バイトレベル)。ASCII 文字は常に被覆されるため、非 ASCII 文字の
    # 被覆数は「被覆数 - 学習コーパス中の ASCII 文字数」で概算できる。
    # 非 ASCII 文字(UTF-8 で 3 バイト)は 1 文字あたり 2 回のマージを要するという
    # 近似で、文字の再構成に費やされたマージ回数の下限を見積もる(概算であることに
    # 注意。実際にはマージがちょうど同じ文字の3バイトだけを対象に進むとは限らない)。
    num_covered, num_total, coverage = compute_character_coverage(
        byte_tokenizer.vocab, train_texts["Japanese"]
    )
    num_nonascii_covered = num_covered - len(japanese_train_ascii_chars)
    estimated_reconstruction_merges = 2 * num_nonascii_covered
    total_merges = len(byte_tokenizer.vocab) - 256
    reconstruction_merge_share = estimated_reconstruction_merges / total_merges
    experiment4_coverage_results[vocab_size] = (
        num_covered,
        num_total,
        coverage,
        estimated_reconstruction_merges,
        reconstruction_merge_share,
    )
    print(
        f"  [文字被覆率(バイトレベル)] {num_covered}/{num_total}={coverage:.1%}  "
        f"再構成に費やしたマージ回数(概算)={estimated_reconstruction_merges}"
        f"/{total_merges}(総マージ回数の {reconstruction_merge_share:.1%})"
    )

    # 対照: 文字レベルの被覆率(常に 1.0 になるはず。バイトレベルだけが
    # 負っているコストであることを示す対照として出力する)。
    char_num_covered, char_num_total, char_coverage = compute_character_coverage(
        tokenizer.vocab, train_texts["Japanese"]
    )
    print(
        f"  [文字被覆率(文字レベル、対照)] {char_num_covered}/{char_num_total}={char_coverage:.1%}"
    )

    filtered_holdout = "".join(ch for ch in holdout if ch in tokenizer.vocab)
    removed_char_count = len(holdout) - len(filtered_holdout)
    char_fertility_filtered = compute_fertility(
        len(tokenizer.encode(filtered_holdout)), len(filtered_holdout)
    )
    byte_fertility_filtered = compute_fertility(
        len(byte_tokenizer.encode(filtered_holdout)), len(filtered_holdout)
    )
    experiment4_unk_removed_results[vocab_size] = (
        removed_char_count,
        char_fertility_filtered,
        byte_fertility_filtered,
    )
    print(
        f"  [未知語除去後] 除去文字数={removed_char_count}  "
        f"char fertility={char_fertility_filtered:.4f}  "
        f"byte fertility={byte_fertility_filtered:.4f}  "
        f"差={byte_fertility_filtered - char_fertility_filtered:.4f}"
    )
```

    requested=  512 achieved= 1715 fertility=1.0000  未知語率=0.0175
      [文字被覆率(バイトレベル)] 188/1715=11.0%  再構成に費やしたマージ回数(概算)=236/256(総マージ回数の 92.2%)
      [文字被覆率(文字レベル、対照)] 1715/1715=100.0%
      [未知語除去後] 除去文字数=262  char fertility=1.0000  byte fertility=1.3862  差=0.3862
    requested= 1024 achieved= 1715 fertility=1.0000  未知語率=0.0175
      [文字被覆率(バイトレベル)] 438/1715=25.5%  再構成に費やしたマージ回数(概算)=736/768(総マージ回数の 95.8%)
      [文字被覆率(文字レベル、対照)] 1715/1715=100.0%
      [未知語除去後] 除去文字数=262  char fertility=1.0000  byte fertility=1.0137  差=0.0137
    requested= 2048 achieved= 2048 fertility=0.8167  未知語率=0.0214
      [文字被覆率(バイトレベル)] 764/1715=44.5%  再構成に費やしたマージ回数(概算)=1388/1792(総マージ回数の 77.5%)
      [文字被覆率(文字レベル、対照)] 1715/1715=100.0%
      [未知語除去後] 除去文字数=262  char fertility=0.8133  byte fertility=0.8286  差=0.0153
    requested= 4096 achieved= 4096 fertility=0.6742  未知語率=0.0259
      [文字被覆率(バイトレベル)] 1026/1715=59.8%  再構成に費やしたマージ回数(概算)=1912/3840(総マージ回数の 49.8%)
      [文字被覆率(文字レベル、対照)] 1715/1715=100.0%
      [未知語除去後] 除去文字数=262  char fertility=0.6683  byte fertility=0.7043  差=0.0360
    requested= 8192 achieved= 8192 fertility=0.6291  未知語率=0.0278
      [文字被覆率(バイトレベル)] 1239/1715=72.2%  再構成に費やしたマージ回数(概算)=2338/7936(総マージ回数の 29.5%)
      [文字被覆率(文字レベル、対照)] 1715/1715=100.0%
      [未知語除去後] 除去文字数=262  char fertility=0.6224  byte fertility=0.6443  差=0.0218



```python
# 実験2 で得た日本語のバイトレベル曲線、実験4 の文字レベル曲線(元のホールドアウト)、
# および文字レベル・未知語除去後の曲線を重ね描きする。
# 文字レベル側(元のホールドアウト)は achieved vocab を注釈として付け、要求値と
# 乖離している点(= マージが発生していない点)を区別できるようにする。
byte_vocab_sizes = sorted(experiment2_results["Japanese"].keys())
byte_fertilities = [experiment2_results["Japanese"][v][1] for v in byte_vocab_sizes]

char_vocab_sizes = sorted(experiment4_results.keys())
char_fertilities = [experiment4_results[v][1] for v in char_vocab_sizes]
char_achieved = [experiment4_results[v][0] for v in char_vocab_sizes]
char_unknown_rates = [experiment4_results[v][2] for v in char_vocab_sizes]
char_filtered_fertilities = [experiment4_unk_removed_results[v][1] for v in char_vocab_sizes]

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.plot(byte_vocab_sizes, byte_fertilities, marker="o", label="byte-level", color="tab:blue")
ax.plot(char_vocab_sizes, char_fertilities, marker="s", label="char-level", color="tab:orange")
ax.plot(
    char_vocab_sizes,
    char_filtered_fertilities,
    marker="^",
    linestyle="--",
    label="char-level (未知語除去後)",
    color="tab:green",
)
for v, f, a in zip(char_vocab_sizes, char_fertilities, char_achieved, strict=True):
    ax.annotate(
        f"achieved={a}",
        (v, f),
        fontsize=7,
        color="tab:orange",
        textcoords="offset points",
        xytext=(4, 6),
    )
ax.set_xscale("log", base=2)
ax.set_xlabel("Vocabulary size V (log scale, requested)")
ax.set_ylabel("Fertility (tokens / char) — Japanese")
ax.set_title("Byte-level vs. char-level initial vocabulary — Japanese")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.show()
```

    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 26410 (\N{CJK UNIFIED IDEOGRAPH-672A}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 30693 (\N{CJK UNIFIED IDEOGRAPH-77E5}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 35486 (\N{CJK UNIFIED IDEOGRAPH-8A9E}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 38500 (\N{CJK UNIFIED IDEOGRAPH-9664}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 21435 (\N{CJK UNIFIED IDEOGRAPH-53BB}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)
    /usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 24460 (\N{CJK UNIFIED IDEOGRAPH-5F8C}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_36_1.png)
    



```python
# 実験4の判定: achieved vocab が要求値と一致し始める語彙サイズ(マージが発生し始める点)、
# および fertility の大小関係(バイトレベル対文字レベル)が入れ替わる区間の有無
print("要求値 vs 実効値(achieved vocab):")
for v, a in zip(char_vocab_sizes, char_achieved, strict=True):
    if a == v:
        status = "要求値と一致(マージが発生し、比較可能な区間)"
    else:
        status = f"要求値を超過(マージ 0 回、コーパスのユニーク文字数 = {a}、比較不能な区間)"
    print(f"  requested={v:5d}  achieved={a:5d}  {status}")

print()
byte_gt_char = [b > c for b, c in zip(byte_fertilities, char_fertilities, strict=True)]
print("語彙サイズ:", char_vocab_sizes)
print("fertility: byte > char ->", byte_gt_char)

crossings = [i for i in range(len(byte_gt_char) - 1) if byte_gt_char[i] != byte_gt_char[i + 1]]
if crossings:
    for i in crossings:
        print(
            f"入れ替わりを検出: 語彙サイズ {char_vocab_sizes[i]} と {char_vocab_sizes[i + 1]} の間"
        )
elif all(byte_gt_char):
    print("入れ替わりなし。全域で文字レベルが有利(fertility が小さい)。")
elif not any(byte_gt_char):
    print("入れ替わりなし。全域でバイトレベルが有利(fertility が小さい)。")
else:
    print("byte_gt_char の系列が単調でない(入れ替わりの検出条件を要確認)。")

print()
print("文字レベル初期語彙のホールドアウト未知語率(fertility 比較の非対称性の評価用):")
for v, u in zip(char_vocab_sizes, char_unknown_rates, strict=True):
    print(f"  requested={v:5d}  未知語率={u:.4f}")

print()
print("--- 対照比較: 未知語除去の有無による byte-char fertility 差への影響 ---")
print("(差の絶対値だけでなく、その語彙サイズの fertility に対する相対値も併記する)")
for i, v in enumerate(char_vocab_sizes):
    _removed, char_filtered, byte_filtered = experiment4_unk_removed_results[v]
    original_diff = byte_fertilities[i] - char_fertilities[i]
    filtered_diff = byte_filtered - char_filtered
    unk_contribution = original_diff - filtered_diff
    unk_contribution_share = unk_contribution / original_diff

    print(f"  requested={v:5d}")
    print(
        f"    fertility(元のホールドアウト)  char={char_fertilities[i]:.4f}  "
        f"byte={byte_fertilities[i]:.4f}"
    )
    print(
        f"      差={original_diff:.4f}(char fertility 比 {original_diff / char_fertilities[i]:.1%})"
    )
    print(f"    fertility(未知語除去後)  char={char_filtered:.4f}  byte={byte_filtered:.4f}")
    print(f"      差={filtered_diff:.4f}(char fertility 比 {filtered_diff / char_filtered:.1%})")
    print(
        f"    未知語の扱いに起因する見かけの寄与={unk_contribution:.4f}"
        f"(元の差の {unk_contribution_share:.1%})"
    )
```

    要求値 vs 実効値(achieved vocab):
      requested=  512  achieved= 1715  要求値を超過(マージ 0 回、コーパスのユニーク文字数 = 1715、比較不能な区間)
      requested= 1024  achieved= 1715  要求値を超過(マージ 0 回、コーパスのユニーク文字数 = 1715、比較不能な区間)
      requested= 2048  achieved= 2048  要求値と一致(マージが発生し、比較可能な区間)
      requested= 4096  achieved= 4096  要求値と一致(マージが発生し、比較可能な区間)
      requested= 8192  achieved= 8192  要求値と一致(マージが発生し、比較可能な区間)
    
    語彙サイズ: [512, 1024, 2048, 4096, 8192]
    fertility: byte > char -> [True, True, True, True, True]
    入れ替わりなし。全域で文字レベルが有利(fertility が小さい)。
    
    文字レベル初期語彙のホールドアウト未知語率(fertility 比較の非対称性の評価用):
      requested=  512  未知語率=0.0175
      requested= 1024  未知語率=0.0175
      requested= 2048  未知語率=0.0214
      requested= 4096  未知語率=0.0259
      requested= 8192  未知語率=0.0278
    
    --- 対照比較: 未知語除去の有無による byte-char fertility 差への影響 ---
    (差の絶対値だけでなく、その語彙サイズの fertility に対する相対値も併記する)
      requested=  512
        fertility(元のホールドアウト)  char=1.0000  byte=1.4069
          差=0.4069(char fertility 比 40.7%)
        fertility(未知語除去後)  char=1.0000  byte=1.3862
          差=0.3862(char fertility 比 38.6%)
        未知語の扱いに起因する見かけの寄与=0.0207(元の差の 5.1%)
      requested= 1024
        fertility(元のホールドアウト)  char=1.0000  byte=1.0352
          差=0.0352(char fertility 比 3.5%)
        fertility(未知語除去後)  char=1.0000  byte=1.0137
          差=0.0137(char fertility 比 1.4%)
        未知語の扱いに起因する見かけの寄与=0.0215(元の差の 61.1%)
      requested= 2048
        fertility(元のホールドアウト)  char=0.8167  byte=0.8507
          差=0.0340(char fertility 比 4.2%)
        fertility(未知語除去後)  char=0.8133  byte=0.8286
          差=0.0153(char fertility 比 1.9%)
        未知語の扱いに起因する見かけの寄与=0.0187(元の差の 55.1%)
      requested= 4096
        fertility(元のホールドアウト)  char=0.6742  byte=0.7275
          差=0.0533(char fertility 比 7.9%)
        fertility(未知語除去後)  char=0.6683  byte=0.7043
          差=0.0360(char fertility 比 5.4%)
        未知語の扱いに起因する見かけの寄与=0.0172(元の差の 32.4%)
      requested= 8192
        fertility(元のホールドアウト)  char=0.6291  byte=0.6682
          差=0.0391(char fertility 比 6.2%)
        fertility(未知語除去後)  char=0.6224  byte=0.6443
          差=0.0218(char fertility 比 3.5%)
        未知語の扱いに起因する見かけの寄与=0.0172(元の差の 44.1%)


### 実験5: 空白による事前分割のネガティブコントロール

3 ドメイン × `chunk_split_mode` $\in \{$`whitespace`, `none`$\}$(語彙サイズは 2048 に固定)で、チャンク長の分布統計と fertility を比較する。

**検証すること**: 空白による事前分割は言語依存であり、日本語では機能しない。

**判定基準**: `whitespace`時のチャンク長中央値が、日本語で英語・コードの 10 倍以上。かつ英語・コードでは事前分割の有無による fertility 差が小さい(ネガティブコントロール)。**判定基準はこの 2 点から変更しない。**

英語・コードが陰性対照(negative control)として機能していること、すなわち「事前分割の方式を変えても英語・コードの fertility はほとんど変わらない」ことを 7 節の考察で明示する。**5.1 節の修正(空白をチャンク先頭に保持する方式)により、`whitespace`モードのチャンクは空白の文字数を含むようになった。** 以前の実装(空白を破棄する方式)では`none`モードのみが空白の文字数を fertility に含めており、両モードの fertility を同じ基準で比較できていなかった。この修正により、この交絡が解消されているかどうかも合わせて確認する。

**インデントを持つコードについての注記**: 上記の修正により、`whitespace`モードのチャンクは先頭の空白を含むようになった。このため、インデントを持つコードでは、チャンク長が「単語の長さ」ではなく「インデントの深さ」を反映しうる。これは日本語の「語境界の不在」とは異なる要因でチャンク長を押し上げうるということであり、判定基準の前半(チャンク長中央値の比較)がコードで不成立になった場合、それが「事前分割がコードでも機能している(語がある程度の長さを持つ)」ことを意味するのか、単に「インデントが計上されているだけ」なのかを区別する必要がある。判定基準そのものは変えず、先頭の連続空白を除いたチャンク長中央値(`compute_chunk_length_statistics`の`_without_leading_whitespace`系のキー)を診断量として併せて測定し、この区別に使う。

なお、判定基準の前半(チャンク長中央値の比較)と後半(事前分割の有無による fertility 差)は独立した測定である。前半が不成立でも後半(陰性対照としての実質的な機能)が成立していることはありうるため、7 節では **両者を分けて評価する**。

**陰性対照としての証拠力についての注記**: 判定基準の後半は「英語・コードの fertility 差が小さいこと」しか要求していない。しかし陰性対照(negative control)が証拠として機能するためには、対照群(英語・コード)で効果が出ず、処理群(日本語)で効果が出るという **対比** が必要である。日本語の fertility 差が英語・コードと同程度の大きさであれば、この測定は「事前分割の言語依存性」について何の証拠力も持たない(判定基準は形式的に満たされていても、対比が存在しないため)。7 節ではこの対比が実際に成立しているかどうかを評価する。

日本語には空白がほとんどないため、`whitespace`モードは実質的に改行でしか分割されず、`none`モードとほぼ同一の動作になる。この意味で、日本語の fertility 差が小さいことは(仮にそうなったとしても)測定結果というより、この構成上の帰結である可能性がある。

**コードコーパスの非不変性についての注記**: 6 節で述べた通り、コード(`load_code_corpus`)は本コミット時点のリポジトリ内容に依存し、将来のコミットで`src/`が拡張されると内容が変化しうる。判定基準の前半はコードのチャンク長中央値に依存するため、**`src/`の内容が変われば判定が変わりうる**。


```python
CHUNK_VOCAB_SIZE = 2048
experiment5_chunk_stats: dict[str, dict[str, dict]] = {}
experiment5_fertility: dict[str, dict[str, float]] = {}

for domain, train_text in train_texts.items():
    experiment5_chunk_stats[domain] = {}
    experiment5_fertility[domain] = {}
    holdout = holdout_texts[domain]
    for mode in ["whitespace", "none"]:
        chunks = pretokenize(train_text, mode)
        stats = compute_chunk_length_statistics(chunks)
        experiment5_chunk_stats[domain][mode] = stats

        if mode == "whitespace":
            tokenizer = bpe_tokenizers[(domain, CHUNK_VOCAB_SIZE)]  # 実験2で学習済み
        else:
            tokenizer = learn_bpe(
                train_text, vocab_size=CHUNK_VOCAB_SIZE, byte_level=True, chunk_split_mode=mode
            )
        fertility = compute_fertility(len(tokenizer.encode(holdout)), len(holdout))
        experiment5_fertility[domain][mode] = fertility

        print(
            f"{domain:>8} mode={mode:>10} median_chunk_len={stats['median']:6.1f} "
            f"(先頭空白除く={stats['median_without_leading_whitespace']:6.1f})  "
            f"mean={stats['mean']:6.1f}  fertility={fertility:.4f}"
        )
```

     English mode=whitespace median_chunk_len=   5.0 (先頭空白除く=   4.0)  mean=   5.6  fertility=0.2847
     English mode=      none median_chunk_len=  37.0 (先頭空白除く=  37.0)  mean=  31.3  fertility=0.2981
    Japanese mode=whitespace median_chunk_len=  52.0 (先頭空白除く=  50.0)  mean= 102.4  fertility=0.8507
    Japanese mode=      none median_chunk_len=  69.5 (先頭空白除く=  69.5)  mean= 116.8  fertility=0.8298
        Code mode=whitespace median_chunk_len=   8.0 (先頭空白除く=   5.0)  mean=  11.1  fertility=0.3452
        Code mode=      none median_chunk_len=  39.0 (先頭空白除く=  32.0)  mean=  38.5  fertility=0.3201



```python
# 実験5の判定(判定基準・判定式は変更しない)
ja_median_ws = experiment5_chunk_stats["Japanese"]["whitespace"]["median"]
en_median_ws = experiment5_chunk_stats["English"]["whitespace"]["median"]
code_median_ws = experiment5_chunk_stats["Code"]["whitespace"]["median"]
print(f"日本語チャンク長中央値(whitespace): {ja_median_ws}")
print(f"英語チャンク長中央値(whitespace): {en_median_ws}")
print(f"コードチャンク長中央値(whitespace): {code_median_ws}")
print(
    f"判定(日本語 >= 英語 x10): {ja_median_ws} >= {en_median_ws} * 10(={en_median_ws * 10}) "
    f"-> {ja_median_ws >= en_median_ws * 10}"
)
print(
    f"判定(日本語 >= コード x10): {ja_median_ws} >= {code_median_ws} * 10(={code_median_ws * 10}) "
    f"-> {ja_median_ws >= code_median_ws * 10}"
)

print()
print("fertility 差(絶対値と、fertility 自体の大きさに対する相対値):")
for domain in ["English", "Japanese", "Code"]:
    ws = experiment5_fertility[domain]["whitespace"]
    none_ = experiment5_fertility[domain]["none"]
    diff = abs(ws - none_)
    relative_diff = diff / ws
    print(
        f"{domain:>8}: whitespace={ws:.4f}  none={none_:.4f}  差={diff:.4f}  "
        f"相対差={relative_diff:.1%}"
    )

print()
print(
    "--- 診断: 先頭の連続空白がチャンク長にどれだけ寄与しているか(判定基準は上のまま変更しない) ---"
)
print(
    "コードの中央値が大きい原因が先頭空白(インデント)であれば、除外後の中央値は英語と同程度になるはずである。"
)
print("日本語の中央値が大きい原因が語境界の不在であれば、除外後もほとんど変わらないはずである。")
for domain in ["English", "Japanese", "Code"]:
    with_ws = experiment5_chunk_stats[domain]["whitespace"]["median"]
    without_ws = experiment5_chunk_stats[domain]["whitespace"]["median_without_leading_whitespace"]
    print(
        f"{domain:>8}: 先頭空白を含む中央値={with_ws:6.1f}  除いた中央値={without_ws:6.1f}  "
        f"差(先頭空白の寄与)={with_ws - without_ws:6.1f}"
    )
```

    日本語チャンク長中央値(whitespace): 52.0
    英語チャンク長中央値(whitespace): 5.0
    コードチャンク長中央値(whitespace): 8.0
    判定(日本語 >= 英語 x10): 52.0 >= 5.0 * 10(=50.0) -> True
    判定(日本語 >= コード x10): 52.0 >= 8.0 * 10(=80.0) -> False
    
    fertility 差(絶対値と、fertility 自体の大きさに対する相対値):
     English: whitespace=0.2847  none=0.2981  差=0.0134  相対差=4.7%
    Japanese: whitespace=0.8507  none=0.8298  差=0.0209  相対差=2.5%
        Code: whitespace=0.3452  none=0.3201  差=0.0251  相対差=7.3%
    
    --- 診断: 先頭の連続空白がチャンク長にどれだけ寄与しているか(判定基準は上のまま変更しない) ---
    コードの中央値が大きい原因が先頭空白(インデント)であれば、除外後の中央値は英語と同程度になるはずである。
    日本語の中央値が大きい原因が語境界の不在であれば、除外後もほとんど変わらないはずである。
     English: 先頭空白を含む中央値=   5.0  除いた中央値=   4.0  差(先頭空白の寄与)=   1.0
    Japanese: 先頭空白を含む中央値=  52.0  除いた中央値=  50.0  差(先頭空白の寄与)=   2.0
        Code: 先頭空白を含む中央値=   8.0  除いた中央値=   5.0  差(先頭空白の寄与)=   3.0



```python
grouped = {
    mode: {domain: experiment5_chunk_stats[domain][mode]["median"] for domain in train_texts}
    for mode in ["whitespace", "none"]
}
plot_grouped_bar(
    grouped,
    title="Median chunk length by domain and pre-tokenization mode",
    ylabel="Median chunk length (characters)",
    xlabel="Domain",
    log_scale=True,
)
plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_41_0.png)
    


### 実験6: スクラッチ Viterbi 分割の検証と BPE との比較

1. 日本語・英語の数十文について、`viterbi_segment`による分割が sentencepiece 自身の分割と完全一致することを確認する。
2. 同一コーパス・同一語彙サイズで BPE と Unigram 言語モデルの分割を比較し、一致率を測る。
3. 不一致箇所の具体例をいくつか表示し、定性的な傾向を読み取る。

**検証すること**: Unigram 言語モデルは BPE と体系的に異なる分割を与える。

**判定基準**: トークン境界の Jaccard 係数(`compute_segmentation_agreement_rate`)の平均が 1.0 未満であり、かつ一致率が低い文に定性的な傾向(どのような箇所で分割が分かれるか)が読み取れる。

**なぜ完全一致率ではなく境界の Jaccard 係数か**: 5.1 節の修正により、自作 BPE(`chunk_split_mode="whitespace"`)も SentencePiece と同様に空白を保持した可逆な符号化になった。したがって両者の符号化の慣習の違いは、もはや「空白を保持するかどうか」ではなく、**空白をどう表現するか**(SentencePiece は`▁`という専用記号に置き換える、自作 BPE は半角空白のまま扱う)という表層形の違いだけになる。この違いだけでも、文単位の完全一致(`compute_exact_match_rate`)を使うと、分割の大部分が同じ位置で切れていても表層形の流儀の違いだけで一致率が 0 に張り付いてしまい、一致の度合いを測れない。境界位置集合の Jaccard 係数を使うことで、部分的な一致の度合いを連続値として評価する。

**語彙サイズについての注記**: sentencepiece の Unigram 言語モデル学習は、語彙サイズが学習コーパスの文字種数(日本語)や、コーパスから抽出できる候補部分文字列数の上限(英語)を外れるとエラーになる。両ドメインで学習可能な語彙サイズとして 2048 を用いる(実験2・実験5と共通)。

**BPE との比較における`byte_level`についての注記**: 実験6-2 で BPE と Unigram 言語モデルを比較する際は、BPE を`byte_level=False`(文字レベル初期語彙)で学習し直す。バイトレベル BPE の部分語は UTF-8 バイトの表現であり、sentencepiece の Unicode 部分語とは構成上表層形が一致しえないため、同一の表現形式(Unicode 文字)で比較する必要がある。

**Viterbi 分割の文字カバレッジについての注記**: sentencepiece は既定で`character_coverage`を 100% 未満(まれな文字を除外)に設定するため、学習語彙が 1 文字も持たない文字を含む文はそもそも分割できない(自作の`viterbi_segment`は未知語へのフォールバックを持たない、3.5 節)。この検証では、学習語彙が単一文字として持つ文字のみからなるホールドアウト文を対象に選ぶ。


```python
UNIGRAM_VOCAB_SIZE = 2048

unigram_tokenizers = {}
for domain in ["English", "Japanese"]:
    t0 = time.time()
    unigram_tokenizers[domain] = train_unigram_model(
        train_texts[domain], vocab_size=UNIGRAM_VOCAB_SIZE, model_prefix=CACHE_DIR / f"spm_{domain}"
    )
    vocab_n = len(unigram_tokenizers[domain].vocab)
    print(f"{domain}: Unigram 言語モデルの学習完了({time.time() - t0:.1f}s, 語彙数={vocab_n})")
```

    English: Unigram 言語モデルの学習完了(0.1s, 語彙数=2047)
    Japanese: Unigram 言語モデルの学習完了(0.5s, 語彙数=2047)



```python
# 1. Viterbi 分割と sentencepiece 自身の分割の一致率
# candidates/covered が想定外に少ない(0 件を含む)場合でもノートブック全体が
# 停止しないよう、対象文が確保できないドメインはスキップし、後続セルが参照する
# 変数には安全な既定値(空リスト・NaN)を入れておく。
viterbi_validation = {}
for domain in ["English", "Japanese"]:
    uni = unigram_tokenizers[domain]
    single_chars = {p for p in uni.vocab if len(p) == 1}

    candidates = [s for s in holdout_texts[domain].split("\n") if s.strip()]
    covered = [s for s in candidates if all(ch in single_chars for ch in uni.normalize(s))]
    sentences = covered[:30]

    if not sentences:
        print(
            f"{domain}: 検証対象の文が確保できなかった"
            f"(候補 {len(candidates)} 文中カバー {len(covered)} 文)。この節はスキップする。"
        )
        viterbi_validation[domain] = (float("nan"), [], [], [], [])
        continue

    library_result = [uni.encode_with_library(s) for s in sentences]
    viterbi_result = [uni.encode_with_viterbi(s) for s in sentences]
    rate, mismatches = compute_exact_match_rate(library_result, viterbi_result)
    viterbi_validation[domain] = (rate, mismatches, sentences, library_result, viterbi_result)

    print(
        f"{domain}: 対象文数={len(sentences)}(候補 {len(candidates)} 文中カバー {len(covered)} 文)"
    )
    print(f"{domain}: Viterbi 一致率={rate:.3f}  不一致件数={len(mismatches)}")
    for i in mismatches[:3]:
        print(f"  不一致例 [{i}]  文={sentences[i]!r}")
        print(f"    sentencepiece: {library_result[i]}")
        print(f"    viterbi_segment: {viterbi_result[i]}")
```

    English: 対象文数=30(候補 485 文中カバー 477 文)
    English: Viterbi 一致率=1.000  不一致件数=0
    Japanese: 対象文数=30(候補 313 文中カバー 199 文)
    Japanese: Viterbi 一致率=1.000  不一致件数=0


**不一致が生じた場合の注記**: 上記セルで不一致件数が 0 でない場合、原因として (a) `character_coverage` により一部の文字が語彙から除外されている、(b) sentencepiece 内部の Viterbi 実装が数値誤差や同点処理で自作実装と異なる経路を選んでいる、のいずれかが考えられる。実行結果を見て、実際にどちらであったかを 7 節で報告する(不一致を考察でごまかさない)。


```python
# 2. 同一語彙サイズでの BPE と Unigram 言語モデルの分割比較
# BPE は byte_level=False(文字レベル初期語彙)で学習し直す。バイトレベル BPE の
# 部分語は UTF-8 バイトの表現であり、sentencepiece の Unicode 部分語とは構成上
# 表層形が一致しえないため、同一の表現形式(Unicode 文字)で比較する。
#
# 5.1 節の修正により自作 BPE も空白を保持するようになったため、両者の違いは
# 空白の表現方法だけになった(sentencepiece は ▁、自作 BPE は半角空白のまま)。
# sentencepiece 側の出力を半角空白表現に変換すれば、連結結果は自作 BPE の
# 連結結果(= 元の文そのもの)と一致するはずである。ただし sentencepiece は
# 文頭に「ダミーの接頭辞」として ▁ を 1 つ余分に付与する(UnigramTokenizer.normalize、
# 3.5 節)ため、これは実際の空白ではなく先頭の 1 文字だけ除去し、残りの ▁ を
# 半角空白に置換する。文の正規化(" ".join(s.split()))は、比較に使う文が
# タブなど半角空白以外の空白文字を含まないようにし、▁ との変換を単純化するために
# 残す(必須ではないが安全側に倒す)。


def _sentencepiece_tokens_to_space_form(tokens):
    tokens = list(tokens)
    if tokens and tokens[0].startswith("▁"):
        tokens[0] = tokens[0][1:]  # 文頭のダミー接頭辞(実際の空白ではない)を除く
    despaced = [t.replace("▁", " ") for t in tokens]
    return [t for t in despaced if t]


bpe_vs_unigram = {}
for domain in ["English", "Japanese"]:
    _, _, sentences, _, _ = viterbi_validation[domain]
    if not sentences:
        print(f"{domain}: 実験6-1 で対象文が確保できなかったため、この節はスキップする。")
        bpe_vs_unigram[domain] = ([], [], [], [])
        continue

    bpe_tok = learn_bpe(
        train_texts[domain],
        vocab_size=UNIGRAM_VOCAB_SIZE,
        byte_level=False,
        chunk_split_mode="whitespace",
    )
    uni = unigram_tokenizers[domain]

    normalized_sentences = [" ".join(s.split()) for s in sentences]
    bpe_result = [bpe_tok.encode(s) for s in normalized_sentences]
    library_result = [
        _sentencepiece_tokens_to_space_form(uni.encode_with_library(s))
        for s in normalized_sentences
    ]

    agreements = [
        compute_segmentation_agreement_rate(b, u)
        for b, u in zip(bpe_result, library_result, strict=True)
    ]
    bpe_vs_unigram[domain] = (agreements, normalized_sentences, bpe_result, library_result)
    print(
        f"{domain}: トークン境界 Jaccard 係数  平均={sum(agreements) / len(agreements):.3f}  "
        f"最小={min(agreements):.3f}  最大={max(agreements):.3f}  (n={len(agreements)})"
    )
```

    English: トークン境界 Jaccard 係数  平均=0.602  最小=0.000  最大=1.000  (n=30)
    Japanese: トークン境界 Jaccard 係数  平均=0.820  最小=0.000  最大=1.000  (n=30)



```python
# 3. 一致率(Jaccard 係数)が低い文の具体例
# 係数が低い順(分割の食い違いが大きい順)に並べ替え、上位 4 文を表示する。
for domain in ["English", "Japanese"]:
    agreements, sentences, bpe_result, library_result = bpe_vs_unigram[domain]
    if not agreements:
        print(f"=== {domain}: 対象文なし ===")
        continue
    order = sorted(range(len(agreements)), key=lambda i: agreements[i])
    print(f"=== {domain}(Jaccard 係数 平均={sum(agreements) / len(agreements):.3f}) ===")
    for i in order[:4]:
        print(f"  文: {sentences[i]!r}  (Jaccard 係数={agreements[i]:.3f})")
        print(f"    BPE:     {bpe_result[i]}")
        print(f"    Unigram: {library_result[i]}")
    print()
```

    === English(Jaccard 係数 平均=0.602) ===
      文: 'CORIOLANUS:'  (Jaccard 係数=0.000)
        BPE:     ['C', 'O', 'RIOLAN', 'US:']
        Unigram: ['CORIOLANUS', ':']
      文: 'CORIOLANUS:'  (Jaccard 係数=0.000)
        BPE:     ['C', 'O', 'RIOLAN', 'US:']
        Unigram: ['CORIOLANUS', ':']
      文: 'appear,'  (Jaccard 係数=0.333)
        BPE:     ['a', 'pp', 'ear', ',']
        Unigram: ['appear', ',']
      文: "man's voice."  (Jaccard 係数=0.333)
        BPE:     ['m', 'an', "'s", ' voic', 'e.']
        Unigram: ['man', "'", 's', ' voice', '.']
    
    === Japanese(Jaccard 係数 平均=0.820) ===
      文: '地理'  (Jaccard 係数=0.000)
        BPE:     ['地', '理']
        Unigram: ['地理']
      文: '気候'  (Jaccard 係数=0.000)
        BPE:     ['気候']
        Unigram: ['気', '候']
      文: '自治体'  (Jaccard 係数=0.000)
        BPE:     ['自治', '体']
        Unigram: ['自治体']
      文: '自然公園'  (Jaccard 係数=0.667)
        BPE:     ['自', '然', '公', '園']
        Unigram: ['自然', '公', '園']
    


### 実験7: 語彙サイズと計算量のトレードオフ(理論計算)

実験2で得た fertility から、固定の文字数を処理する際の系列長 $n$ を各語彙サイズについて求め、$d_{\text{model}}$ を固定した仮想的なモデルについて、埋め込み行列のパラメータ数 $V \times d_{\text{model}}$、Attention の演算量 $O(n^2 d_{\text{model}})$、順伝播ネットワークの演算量 $O(n \, d_{\text{model}}^2)$ を算出する(3.7 節)。

**これは実測(実際の実行時間の計測)ではなく、理論式による計算である。** 以下のセルは fertility という実測値を入力に使うが、算出する量そのものは実行時間の計測結果ではない。3 ドメインそれぞれで、埋め込みパラメータ数の増加曲線と系列長由来の演算量の減少曲線がどのあたりで交差するか(相対的な大小関係がどう変わるか)を確認する。


```python
D_MODEL = 768  # 仮想的なモデルの隠れ次元(固定)
TOTAL_CHARS = 1_000_000  # 処理対象として仮定する文字数 C(固定)

experiment7_results = {}
for domain, results in experiment2_results.items():
    vocab_sizes = sorted(results.keys())
    achieved_vocabs = [results[v][0] for v in vocab_sizes]
    fertilities = [results[v][1] for v in vocab_sizes]
    seq_lengths = [f * TOTAL_CHARS for f in fertilities]  # n = fertility x C

    embedding_params = [v * D_MODEL for v in achieved_vocabs]
    attention_flops = [n**2 * D_MODEL for n in seq_lengths]
    ffn_flops = [n * D_MODEL**2 for n in seq_lengths]

    experiment7_results[domain] = {
        "vocab_sizes": vocab_sizes,
        "seq_lengths": seq_lengths,
        "embedding_params": embedding_params,
        "attention_flops": attention_flops,
        "ffn_flops": ffn_flops,
    }
    print(f"=== {domain} ===")
    for i, v in enumerate(vocab_sizes):
        print(
            f"  V={v:5d}  n={seq_lengths[i]:9.0f}  embed_params={embedding_params[i]:11,.0f}  "
            f"attn_flops={attention_flops[i]:.3e}  ffn_flops={ffn_flops[i]:.3e}"
        )
```

    === English ===
      V=  512  n=   458600  embed_params=    393,216  attn_flops=1.615e+14  ffn_flops=2.705e+11
      V= 1024  n=   341467  embed_params=    786,432  attn_flops=8.955e+13  ffn_flops=2.014e+11
      V= 2048  n=   284667  embed_params=  1,572,864  attn_flops=6.223e+13  ffn_flops=1.679e+11
      V= 4096  n=   263067  embed_params=  3,145,728  attn_flops=5.315e+13  ffn_flops=1.552e+11
      V= 8192  n=   250333  embed_params=  5,033,472  attn_flops=4.813e+13  ffn_flops=1.477e+11
    === Japanese ===
      V=  512  n=  1406867  embed_params=    393,216  attn_flops=1.520e+15  ffn_flops=8.298e+11
      V= 1024  n=  1035200  embed_params=    786,432  attn_flops=8.230e+14  ffn_flops=6.106e+11
      V= 2048  n=   850667  embed_params=  1,572,864  attn_flops=5.558e+14  ffn_flops=5.017e+11
      V= 4096  n=   727467  embed_params=  3,145,728  attn_flops=4.064e+14  ffn_flops=4.291e+11
      V= 8192  n=   668200  embed_params=  6,291,456  attn_flops=3.429e+14  ffn_flops=3.941e+11
    === Code ===
      V=  512  n=   621267  embed_params=    393,216  attn_flops=2.964e+14  ffn_flops=3.664e+11
      V= 1024  n=   452400  embed_params=    786,432  attn_flops=1.572e+14  ffn_flops=2.668e+11
      V= 2048  n=   345200  embed_params=  1,572,864  attn_flops=9.152e+13  ffn_flops=2.036e+11
      V= 4096  n=   293667  embed_params=  3,145,728  attn_flops=6.623e+13  ffn_flops=1.732e+11
      V= 8192  n=   272333  embed_params=  6,291,456  attn_flops=5.696e+13  ffn_flops=1.606e+11



```python
for domain, r in experiment7_results.items():
    plot_dual_axis_curves(
        r["vocab_sizes"],
        left_curves={"Embedding params (V x d_model)": r["embedding_params"]},
        right_curves={
            "Attention compute (n^2 d_model)": r["attention_flops"],
            "FFN compute (n d_model^2)": r["ffn_flops"],
        },
        xlabel="Vocabulary size V (log scale)",
        left_ylabel="Embedding parameters",
        right_ylabel="Compute (FLOPs, arbitrary units)",
        title=f"Vocabulary size vs. parameters/compute trade-off — {domain}",
        log_x=True,
    )
    plt.show()
```


    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_50_0.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_50_1.png)
    



    
![png](https://raw.githubusercontent.com/kojikojiprg/ai-theories-publish/main/images/005_tokenizer/output_50_2.png)
    




## 元ノートブック(実装の全文はこちら)

https://github.com/kojikojiprg/ai-theories/blob/main/theories/02_pretraining/005_tokenizer.ipynb

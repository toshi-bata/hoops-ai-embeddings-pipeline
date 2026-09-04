# HOOPS AI 1.1 ベンチマーク (AWS EC2) - CPU vs GPU

マシン: **AWS g6.8xlarge**（AMD EPYC 16コア + NVIDIA L4）。データセット: `mechcad` ヘビー機械CADコーパス - Step 1 のワーカースイープ、Step 2 の CPU vs GPU 学習、そして Step 3 の埋め込み + FAISS索引化（チュートリアルの事前学習モデル `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` を使用）のすべてで、同じ10,715ファイルの全コーパスを使用。
ベンチマーク実行: 成功 23 件、失敗/スキップ 0 件。生データ: `results/results.csv`、各実行ログは `logs/`。

もう一方のマシンの対になるレポートも参照（`REPORT_local.*` / `REPORT_aws.*`）。

## 結論（要約）

結論を先に述べる。裏付けとなる数値と設定ごとのスイープは各Stepのセクションを参照。

- **Step 1（エンコード）と Step 3（埋め込み + 索引化）はCPUバウンド** — GPUはアイドルで、GPUインストールの恩恵はない。Step 1 のスループットはワーカー数に応じて物理コア数付近まで向上し、その後は頭打ち。Step 3 はより少ないワーカー数でピークに達し、それ以上増やすと*低下*する — 各ワーカープロセスが独自のCPU計算（intra-op）スレッド群を起動するため、少数のワーカーを超えると総スレッド数がコア数を大きく上回り、そのCPUスレッドの過剰割り当て（RAM やファイルディスクリプタの負荷ではなく）が全ワーカーを遅くするため。
- **Step 2（学習）が唯一のGPUバウンドな段階**であり、GPUが効果を発揮する唯一の場所。本モデルは対比学習のため batch_size を変えると学習されるモデル自体が変わる。したがって公平な速度比較は batch_size を固定して行う：チュートリアル既定の batch_size=64（同一モデル・同一10 epoch）で、**10,715ファイルのヘビーデータ**（AWS g6.8xlarge・NVIDIA L4）では GPUは16コアCPUの約**4.9倍**高速（497 vs 101 s/epoch）。比較は総wall（固定の起動コストを含む）ではなく `s/epoch` で行う。
- _表の読み方:_ 各スイープ内の速度向上は、そのグループで実測した最小設定を基準に示す（最小ワーカー数、Step 2 では最小バッチサイズ）。並列効率は速度向上を理想的な線形速度向上で割った値。

### テスト環境

#### AWS g6.8xlarge (EC2)

ヘビースケール実行環境: 約1万ファイルの Step 1 ワーカースイープと Step 2 CPU vs GPU 学習をこの環境で実行。両インストールは同一インスタンス上で、異なるのは torch wheel のみ（共通行はセル結合）。
<table>
<thead><tr><th>項目</th><th>CPU1.1</th><th>GPU1.1</th></tr></thead>
<tbody>
<tr><td>CPU</td><td colspan='2'>AMD EPYC 7R13 Processor</td></tr>
<tr><td>コア</td><td colspan='2'>16 物理 / 32 論理</td></tr>
<tr><td>RAM</td><td colspan='2'>121.2 GB</td></tr>
<tr><td>GPU (nvidia-smi)</td><td colspan='2'>NVIDIA L4, 23034 MiB, 580.173.02, 8.9</td></tr>
<tr><td>torch</td><td>2.9.1+cpu (cuda None)</td><td>2.9.1+cu130 (cuda 13.0)</td></tr>
<tr><td>cuda_available</td><td>False</td><td>True</td></tr>
<tr><td>hoops_ai</td><td colspan='2'>1.1.1</td></tr>
<tr><td>OS / Python</td><td colspan='2'>Linux-7.0.0-1011-aws-x86_64-with-glibc2.39 / 3.12.3</td></tr>
</tbody></table>

## Step 1 - CADエンコード (DataPrep)

### max_workers スケーリング

**CPU1.1, n=10715 files** (基準 max_workers=8)

<table>
<thead><tr><th>max_workers</th><th>時間 (s)</th><th>ファイル/s</th><th>速度向上</th><th>並列効率</th><th>最大RSS (MB)</th><th>失敗</th></tr></thead>
<tbody>
<tr><td>8</td><td>3134.7</td><td>3.42</td><td>1.00x</td><td>100%</td><td>14266</td><td>216</td></tr>
<tr><td>12</td><td>2190.3</td><td>4.89</td><td>1.43x</td><td>95%</td><td>20619</td><td>216</td></tr>
<tr><td>16</td><td>1778.5</td><td>6.02</td><td>1.76x</td><td>88%</td><td>26946</td><td>216</td></tr>
<tr><td>20</td><td>1649.5</td><td>6.50</td><td>1.90x</td><td>76%</td><td>33361</td><td>216</td></tr>
<tr><td>24</td><td>1586.7</td><td>6.75</td><td>1.98x</td><td>66%</td><td>39610</td><td>216</td></tr>
<tr><td>28</td><td>1538.2</td><td>6.97</td><td>2.04x</td><td>58%</td><td>45737</td><td>216</td></tr>
<tr class='peak'><td>32</td><td>1495.6</td><td>7.16</td><td>2.10x</td><td>52%</td><td>53532</td><td>216</td></tr>
<tr><td>36</td><td>1508.4</td><td>7.10</td><td>2.08x</td><td>46%</td><td>59292</td><td>216</td></tr>
<tr><td>40</td><td>1522.6</td><td>7.04</td><td>2.06x</td><td>41%</td><td>63636</td><td>216</td></tr>
</tbody></table>

**最速: max_workers = 32**（1495.6 s、7.16 ファイル/s）- このグループで最短。（max_workers = 28 でその3%以内 - 最小コストでピーク性能。）

アムダール則フィット (max_workers=8..40): 直列成分 f = **6.5%**。ワーカーをいくら増やしても 漸近的な上限は 15.4倍。おおよそ max_workers = 58 で その上限の80%に到達する。

### CPU vs GPU（venvは影響するか？）

**Step 1 はCPUバウンドで、GPUインストールの恩恵はない。** max_workers=32 で、CPU 1495.6 s vs GPU 1483.5 s: GPUインストールはCPUの **1.01倍** のスループット（実行ごとのノイズの範囲内＝ほぼ同等）。エンコードはCPUワーカー上のHOOPS Exchange + numpyで、モデルの順伝播を一切行わないためtorch wheelは無関係。このパリティこそが Step 2/3 の CPU vs GPU 比較の妥当性を担保する — 両インストールはデバイス依存の段階以外はすべて同一である。

同一 max_workers でのCPU vs GPU（10,715ファイルの全コーパスで両インストールを max_workers=32 で実行）:

<table>
<thead><tr><th>max_workers</th><th>CPU 時間 (s)</th><th>GPU 時間 (s)</th><th>GPU速度向上</th></tr></thead>
<tbody>
<tr><td>32</td><td class='peak'>1495.6</td><td class='peak'>1483.5</td><td>1.01x</td></tr>
</tbody></table>
_ハイライト: max_workers=32 の両インストール — エンコードはGPUを一切使わないため、両者は実行ごとのノイズの範囲内（ほぼ同等）。GPU速度向上 = CPU時間 / GPU時間。_

## Step 2 - 学習

### 学習速度 - CPU vs GPU（1万ファイルのヘビーデータ）

**AWS g6.8xlarge**（AMD EPYC・物理16コア + **NVIDIA L4** GPU）で、上記 Step 1 / Step 3 のワーカースイープと同じ全コーパス **10,715ファイル**のヘビーな機械CADコーパス（`mechcad`）に対して実行。両デバイスとも**同一のデータセットポインタ**・batch_size=64・num_workers=0・matmul=high・固定seed でスクラッチから10 epoch学習するため、両実行は*同一モデル*を生成し、wall時間の差は純粋なCPU vs GPUの速度テストになる。

**なぜ batch_size を固定するか。** 本埋め込みモデルは*対比学習*（SimCLR / NT-Xent）で学習される。バッチがバッチ内の負例を供給するため、batch_size を変えると*別のモデル*が学習される。batch_size=64（チュートリアル既定）に固定すればモデルは同一に保たれ、device 速度だけを切り出せる。この規模では1 epochは約204バッチ。

**投入数 vs 学習数。** 10,715ファイルは*投入*数。Step 1 のエンコードは、HOOPS AI の既知の**非決定的な約2%の再エンコード失敗**（典型例 `division by zero` や `Data key 'graph' not found`。しかも失敗するファイルは実行ごとに*異なる*）を許容する。今回は **216** ファイルが落ち、モデルは実際には **10,499件の成功分**で学習されている。これは事前にファイルリストから除外した24件とは別物で、失敗が再現しないため事前フィルタで除ききれない。

#### 同一設定でのCPU vs GPU（batch_size=64）

**10,715ファイルで（同一モデル・同一10 epoch・同一seed、変えるのは device のみ）、L4 GPUは16コアCPUの 4.9倍 高速:** CPU 497.3 vs GPU 101.1 s/epoch（8.3 vs 1.7 分/epoch）。10 epochでは学習wall時間が CPU 83分 vs GPU 17分。

<table>
<thead><tr><th>デバイス</th><th>s/epoch</th><th>分/epoch</th><th>学習wall (分)</th><th>最大RSS (MB)</th><th>最大GPU (MB)</th></tr></thead>
<tbody>
<tr><td>CPU (EPYC 16コア)</td><td>497.3</td><td>8.29</td><td>82.9</td><td>7299</td><td>-</td></tr>
<tr class='peak'><td>GPU (NVIDIA L4)</td><td>101.1</td><td>1.68</td><td>16.8</td><td>2553</td><td>6180</td></tr>
</tbody></table>

_速度向上 = CPU s/epoch / GPU s/epoch = 497.3 / 101.1 = **4.9倍**。重いテンソル演算がデバイス側で動くため、GPUはホストRAM（最大RSS）も大幅に少ない。


**知見**

- **比較は総wallではなく s/epoch で。** split/セットアップは固定コスト。限界学習コストは CPU 497.3 vs GPU 101.1 s/epoch（**4.9倍**）。長時間実行の見積りは s/epoch から行う。
- **L4のメモリ余裕:** batch_size=64で最大GPU 6180 MB — L4の23 GBに対して十分小さく、より大きなバッチやモデルでも収まる。
- **ホストRAM:** CPU学習の最大RSSは 7.1 GB、GPU実行は 2.5 GB — デバイスへのオフロードはホストのメモリ負荷も下げる。

## Step 3 - 埋め込み + FAISS索引化

### num_workers スケーリング

**CPU1.1, n=10715 files** (基準 num_workers=4)

<table>
<thead><tr><th>num_workers</th><th>時間 (s)</th><th>ファイル/s</th><th>速度向上</th><th>並列効率</th><th>最大RSS (MB)</th><th>失敗</th></tr></thead>
<tbody>
<tr><td>4</td><td>6219.5</td><td>1.72</td><td>1.00x</td><td>100%</td><td>7545</td><td>0</td></tr>
<tr><td>8</td><td>3486.1</td><td>3.07</td><td>1.78x</td><td>89%</td><td>14167</td><td>0</td></tr>
<tr class='peak'><td>12</td><td>3377.6</td><td>3.17</td><td>1.84x</td><td>61%</td><td>20802</td><td>0</td></tr>
<tr><td>14</td><td>3432.3</td><td>3.12</td><td>1.81x</td><td>52%</td><td>23866</td><td>0</td></tr>
<tr><td>16</td><td>4124.8</td><td>2.60</td><td>1.51x</td><td>38%</td><td>28208</td><td>0</td></tr>
<tr><td>20</td><td>4762.8</td><td>2.25</td><td>1.31x</td><td>26%</td><td>32422</td><td>0</td></tr>
<tr><td>24</td><td>4974.9</td><td>2.15</td><td>1.25x</td><td>21%</td><td>38949</td><td>0</td></tr>
<tr><td>28</td><td>5056.4</td><td>2.12</td><td>1.23x</td><td>18%</td><td>45960</td><td>0</td></tr>
<tr><td>32</td><td>5139.8</td><td>2.08</td><td>1.21x</td><td>15%</td><td>55738</td><td>0</td></tr>
<tr><td>36</td><td>5202.2</td><td>2.06</td><td>1.20x</td><td>13%</td><td>61025</td><td>0</td></tr>
</tbody></table>

_**ファイル/s** 列は入力CADファイル数を数える。この 10,715 ファイルは **37,548 個のB-rep形状** に展開され（約3.5 形状/ファイル - ヘビーコーパスはアセンブリが多い）、Step 3 は各形状を埋め込む: ピークでは約11.1 形状/s。_


**最速: num_workers = 12**（3377.6 s、3.17 ファイル/s）- このグループで最短。

アムダール則フィット (num_workers=4..36): 直列成分 f = **52.6%**。ワーカーをいくら増やしても 漸近的な上限は 1.9倍。おおよそ num_workers = 4 で その上限の80%に到達する。

### CPU vs GPU

**Step 3 の索引化もCPUバウンドで、GPUの恩恵はない。** 各プラットフォームのピークで、CPU 3377.6 s (num_workers=12) vs GPU 3282.1 s (num_workers=12): GPUインストールはCPUの **1.03倍** のスループット（つまり高速）。索引化の全実行で最大GPUメモリは0 MB - `embed_shape_batch` はB-repエンコードをCPUワーカー上で実行するため、ここではGPUを足しても効果がない。

同一 num_workers でのCPU vs GPU:

<table>
<thead><tr><th>num_workers</th><th>CPU 時間 (s)</th><th>GPU 時間 (s)</th><th>GPU速度向上</th></tr></thead>
<tbody>
<tr><td>12</td><td class='peak'>3377.6</td><td class='peak'>3282.1</td><td>1.03x</td></tr>
</tbody></table>
_ハイライト: 各プラットフォームのピーク（CPU num_workers=12、GPU num_workers=12）。GPU速度向上 = CPU時間 / GPU時間。1.00x未満はGPUインストールの方が遅いことを意味する。_

## 注意事項

- 単一マシン・各構成1回のみの実行。約5%未満の差は実行ごとのノイズの範囲内。小さな差に結論が依存する場合は再実行すること。
- ヘビーな `mechcad` ファイルは HOOPS AI の既知の非決定的な約2%の再エンコード失敗を誘発し、実行ごとに数十ファイルが脱落する。スループットは正常処理できたファイルに対する値。
- マシンのバックグラウンド負荷（STEPファイルをスキャンする アンチウイルス等）は、GPU段階よりCPUバウンド段階に強く影響する。

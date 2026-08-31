# HOOPS AI Embeddings Pipeline: CPU vs GPU

*(English: [README.md](README.md))*

HOOPS AI embeddingsワークフロー――CADファイルのフォルダを**エンコード**し、埋め込みモデルを**学習**し、類似検索用にその結果を**インデックス化**する――のための再利用可能な3ステップパイプライン。加えて、ノートPCからAWS GPUインスタンスまで、1万ファイル超のコーパスでCPU/GPUを比較するために使ったツール群。

このリポジトリはもともとその場しのぎのベンチマークハーネスとして始まり、次の3つに発展した。

1. **3本のCLIスクリプト**(`bench_step1_dataprep.py` / `bench_step2_training.py` / `bench_step3_indexing.py`)。対応するHOOPS AI embeddingsのデモノートブックをラップしたもので、それぞれ明示的な`--accelerator`・対象フォルダ・ワーカー数・タイムリミットのパラメータを持つ。
2. **`run_pipeline.py`**。3ステップを*1回ずつ*、順番に、CADファイルのフォルダに対して通す本番用ランナー。スイープもマトリクスもなく、単純にencode -> train -> index。
3. **スイープハーネス**(Windows向け`run_benchmark.ps1`・`run_local_sweep.ps1`、AWS/Ubuntu向け`run_heavy_batch.sh`・`run_heavy_scaling.sh`)。同じ3本のスクリプトを`(accelerator, workers, batch_size, ...)`の多数の組み合わせで実行し、結果をHTMLレポートにまとめる――[`reports/`](reports/)参照。

上記すべてを手作業で構築せずに済むよう、GPU付きEC2インスタンス(ドライバ・venv・このリポジトリ)を自動構築するAWS CDKプロジェクトは[`cdk/`](cdk/)を参照。

> **これはサンプル/ベンチマークコードであり、製品ではない。** HOOPS AI embeddingsワークフローを自動化するための妥当な出発点ではあるが、堅牢化されたパイプラインではない――気にするデータを対象に使う前に「既知の落とし穴」節を読むこと。

## 前提条件

これは、CPU用ビルド/GPU用ビルドのvenvが満たしているべき状態の説明である――既にHOOPS AI SDKのセットアップ済み環境(HOOPS AI自身のセットアップ手順に沿ったもの)があり`hoops_ai`がimportできるなら、以下はおそらく既に満たされているので、下のQuickstartへ進んでよい。以下のコマンドが関係してくるのは、venvをゼロから作る場合(新しいマシン上で、あるいは`cdk/`経由――そちらはCPU/GPUのpipインストールを自動的に行う)である。

- アクセラレータごとのvenvに`hoops-ai[all]`をインストールしていること。[公式pipインストール手順](https://docs.techsoft3d.com/hoops/ai/getting_started/install_pip.html)の通り(Python 3.12。hoops-ai自身のパッケージ用`--extra-index-url`と、対応するtorchビルド用`--extra-index-url`の両方が必要):
  ```bash
  pip install "hoops-ai[all]" --extra-index-url https://packages.techsoft3d.com/pip \
      --extra-index-url https://download.pytorch.org/whl/cpu     # CPU venv
  pip install "hoops-ai[all]" --extra-index-url https://packages.techsoft3d.com/pip \
      --extra-index-url https://download.pytorch.org/whl/cu130   # GPU venv (CUDA 13.0)
  ```
  インストール自体に認証情報は不要――ライセンスキーは実行時のSDK有効化にのみ必要(下記`.env`参照)。`cdk/`はこの2つのコマンドを自動的に実行する。
- `pip install -r requirements.txt`(`psutil`だけ。それ以外は上記の通り)。
- **`.env`ファイル** -- [`.env.example`](.env.example)を`.env`(リポジトリ直下、gitignore対象)にコピーして中身を埋める。この1ファイルがここにあるすべてから読まれる:`bench_step*.py`/`run_pipeline.py`のあらゆる実行は`bench_common.py`の`load_dotenv()`経由で`HOOPS_AI_LICENSE`と`HOOPS_AI_CKPT`をここから拾い、`run_benchmark.ps1`/`preflight.ps1`/`run_local_sweep.ps1`も`resolve_local_env.ps1`経由で`CPU_PY`/`GPU_PY`/`HOOPS_AI_CKPT`をここから拾う――PowerShell専用の別設定ファイルは存在しない。探索順序(最初に見つかったものを使用):`./.env`、`../.env`(1つ上の階層――このリポジトリが`CPU1.1`/`GPU1.1`の隣に置かれている場合はSDKインストールのルート)、`../CPU1.1/.env`、`../GPU1.1/.env`。
  - `HOOPS_AI_LICENSE` -- すべてのスクリプトで必須。ファイルを持ちたくなければ直接環境変数としてexportしてもよい。
  - `HOOPS_AI_CKPT` -- `*/packages/trained_ml_models/`配下での`ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`の自動探索が見つけられない場合のみ必要(例えば、このリポジトリがSDKインストールディレクトリの中にネストされていない場合――通常はそうなっているはず)。実行ごとに`--ckpt`で上書きも可能。
  - `CPU_PY` / `GPU_PY` -- 上記のWindows用PowerShellスクリプト3本でのみ必要。HOOPS AI SDKのインストールフォルダ名は標準化されていない(あるマシンでは`V1.1`、別のマシンではリネーム後の`CPU1.1`など)ため、このリポジトリ側では推測しない。実行ごとに`-CpuPy`/`-GpuPy`で上書きも可能。Linux側の`run_heavy_batch.sh`/`run_heavy_scaling.sh`も同じ`CPU_PY`/`GPU_PY`という名前を使うが、`.env`経由ではなく素のシェル環境変数として扱う(exportするか、`.env`を自分でsourceする)。
- モニタの繋がっていないヘッドレスなLinuxボックス(EC2インスタンスなど)の場合:Xディスプレイが必要。HOOPS AIはオフスクリーン処理でもXディスプレイを要求する。`run_heavy_batch.sh`と`run_heavy_scaling.sh`は`$DISPLAY`が未設定なら`Xvfb :99`を自動起動する。`cdk/`のCDKスタックもsystemdサービスとしてこれをインストールする。

## Quickstart: パイプラインを1回、自分のデータに対して実行する

> **必ずCPUまたはGPU用venv自身の`python.exe`を使うこと。素の`python`ではない。**
> `hoops_ai`はそのvenvの中にインストールされている(Windowsなら例えば`C:\SDK\HOOPS_AI\CPU1.1\.venv\Scripts\python.exe`、Linuxなら`$CPU_PY`/`$GPU_PY`が指す先)――このリポジトリのワークフローにはグローバルな環境や「activate」の手順は無い。以下のコマンド例はすべて、適切なインタプリタに置き換えている前提。`ModuleNotFoundError: No module named 'hoops_ai'`が出たら、これが原因。
> 特に`run_pipeline.py`は3ステップすべてを*起動したインタプリタそのもの*で実行するので、`--accelerator gpu`を使うにはGPU用venvのpython.exeで起動する必要がある。

```bash
python run_pipeline.py \
  --run-name my_corpus --source-dir /path/to/cad_files \
  --accelerator gpu --workers 16 --epochs 10 --batch-size 32
```

これは`--source-dir`を再帰的にスキャンし、拡張子でCADファイルを拾う――STEP、IGES、Parasolid、ACIS、CATIA V4/V5/V6、SolidWorks、Inventor、NX/Creo/Pro-E、Solid Edge、JT、Rhino、PRC:[HOOPS Exchangeのドキュメント](https://docs.techsoft3d.com/hoops/exchange/start/supported-formats.html)がフルB-repを持つとしているフォーマットであり、STLのようなメッシュのみのフォーマットは含まない(HOOPS AIのエンコードには実際のB-repが必要)。正確な一覧は`bench_common.py`の`CAD_EXTENSIONS`を参照。`--extensions .stp,.catpart,...`で上書きも可能。その後、発見したリストを固定し、step 1(エンコード、データセットを保持)、step 2(学習、デフォルトでは同梱チェックポイントからウォームスタート――`--no-warm-start`でスクラッチから学習)、step 3(埋め込み+FAISSインデックス構築、元のウォームスタート重みではなくstep 2で*今学習した*チェックポイントを使う)を実行する。デフォルトでパーツごとにPNG+`.scs`シーンキャッシュもレンダリングする(`--no-gen-images`でスキップ可能)。これはqt_sandboxの「Add Folder」が生成するものと同じで、qt_sandbox側ではそもそも省略できない。各ステップは`results/results.csv`に1行ずつ追記し、成果物は`out/<run-name>/`配下に置かれる:

- `<run-name>.ckpt` -- 学習済みチェックポイントのフラットなコピー(Lightning自身は`training/.../flowtrainer/...`のさらに深い階層に書き出す)
- `indexing/<run-name>.faiss` + `.meta` -- 保存されたインデックス
- `indexing/<run-name>/` -- パーツごとの`.png`/`.scs`アセット。`--source-dir`のフォルダ構成をそのまま配下に再現する

qt_sandboxのようなデスクトップクライアントで使うにはこれらすべてをコピーする――先にチェックポイントをロードしてからインデックスを開くこと(別モデルの埋め込みは比較できないため)。

パラメータの全リスト(ワーカー数、ファイル単位のタイムリミット、DataLoaderワーカー数、インデックス時のPNGレンダリングなど)は`python run_pipeline.py --help`を参照。

## 3ステップ単体での実行

| | Step 1 -- encode | Step 2 -- train | Step 3 -- index |
|---|---|---|---|
| スクリプト | `bench_step1_dataprep.py` | `bench_step2_training.py` | `bench_step3_indexing.py` |
| CPU/GPU | 暗黙(どのvenvで実行するか) | `--accelerator {cpu,gpu}` | 自動検出。インストール済みのtorchビルドがそのカード向けのカーネルを実際に実行できない場合はCPUにフォールバックし、正しくラベル付けする |
| 対象フォルダ | `--source-dir`、再帰的にスキャン(または`--filelist`、後述) | `--dataset-pointer`(step 1の出力を指す) | `--source-dir`、再帰的にスキャン(または`--filelist`) |
| ワーカー数 | `--max-workers` | `--num-workers`(DataLoader) | `--num-workers` |
| ファイル単位のタイムリミット | `--time-limit-s` | -- (学習は設計上epoch単位で区切られ、壁時計ベースではない) | `--time-limit` |

Step 1とstep 3はどちらも`--source-dir`でプレーンなフォルダを受け付ける:その配下(サブフォルダを含む)にあるファイルのうち、拡張子が`CAD_EXTENSIONS`(`bench_common.py`――HOOPS Exchangeのドキュメントがフルなb-repを持つとしているフォーマット。手元のデータやライセンスに合わなければ`--extensions .stp,.catpart,...`で実行ごとに上書き可能)に含まれるものすべてが対象になり、決定的な順序でソートされ、`filelists/<env-tag>_discovered.txt`に固定保存される。これにより繰り返し実行しても同じファイルを同じように処理する。厳密で使い回し可能な手作業のリストが必要な場合は、`--source-dir`の代わりに`--filelist <path>`を渡す――これは下記のスイープハーネスが行っていることで、スイープ内のすべての設定が同一の入力で比較されるようにするためである。

各スクリプトは単独でも実行できる。例:

```bash
python bench_step1_dataprep.py --env-tag gpu --max-workers 16 \
    --source-dir /path/to/cad --keep-output

python bench_step2_training.py --env-tag gpu --accelerator gpu \
    --batch-size 32 --new-epochs 10

python bench_step3_indexing.py --env-tag gpu --num-workers 16 \
    --source-dir /path/to/cad --save-index
```

`run_pipeline.py`は、まさにこの3つの呼び出しを(データセットのポインタ、続いてチェックポイントのポインタで)つなぎ合わせた薄いラッパーであり、中間のJSONファイルを自分で追いかける必要がない。なお`bench_step2_training.py`は、単体で呼んでも`run_pipeline.py`経由で呼んでも、学習済みチェックポイントを(Lightning自身の深いパスに加えて)常に`out/<env-tag>/<env-tag>.ckpt`にもコピーし、`results/checkpoint_<env-tag>.json`に両方のパスを記録する――分かりやすい場所に置かれたパスは常に存在する。`bench_step3_indexing.py`の`--index-name`を使えば、デフォルトの`<env-tag>_n<n_files>`ではなく同じ命名でインデックスを保存できる――`--gen-images`と組み合わせると、パーツごとの`.png`/`.scs`アセットの置き場所(`<index-name>/`、`<index-name>.faiss`の直下の隣)も制御し、qt_sandbox自身のレイアウトに一致させる。`run_pipeline.py`と異なり、こちら(`bench_step3_indexing.py`単体)では`--gen-images`は**デフォルトOFF**のまま:このスクリプトはベンチマークハーネスの部品も兼ねており、このフラグはもともとレンダリング(大きく一定のコスト)を`num_workers`の傾向から切り離して測るためのものだった――詳細は下記「公平な比較のための設計判断」参照。

## ベンチマーク:パイプライン全体でのCPU vs GPU

上記のスクリプトは、スイープハーネスが呼び出しているものでもある。1計測につき1プロセスを起動するので、CUDAコンテキスト/ワーカープール/アロケータの状態が実行間で漏れることはない。エントリポイントは2つ:

- **`run_benchmark.ps1`**(Windows、ローカルマシン)-- フェーズ分割された時間予算付きスイープ:プローブ -> step 1の`max_workers`スイープ -> フルエンコード(両環境) -> step 2の`accelerator x batch_size x num_workers`マトリクス -> step 3の`num_workers`スイープ+フルインデックス構築 -> レポート。`-DryRun`、`-Unattended`(夜間実行:確認プロンプト無し、スリープ抑止、1計測がハングしても全体の予算を食い尽くさないためのハードなper-runタイムアウト)、`-Phases`に対応。
- **`run_heavy_batch.sh`** / **`run_heavy_scaling.sh`**(AWS/Ubuntu、大規模コーパス)-- 同じ考え方を1万ファイル超の規模で:`run_heavy_batch.sh`はstep 1を1回(CPU、データセットを保持)実行してからstep 2をCPU vs GPUで実行し、両側が*同じ*モデルを学習するようにする。`run_heavy_scaling.sh`は代表的なサブセットに対してstep 1/3のワーカー数をスイープする。

結果はすべて`results/results.csv`(gitignore対象――CADファイルのパスを含みうるため)に書き込まれ、`make_report.py --lang en --scope {local,aws,both}`でそれを`results/REPORT_*.md/.html`に変換する。[`reports/`](reports/)配下に既にコミットされている2つのレポートは、このリポジトリの元になったCPU1.1/GPU1.1のノートPCおよびg6.8xlarge/L4での実行結果である――自分の`results.csv`ができたら、同じツールで自分のレポートを再生成すること。

## 公平な比較のための設計判断

CPU-vs-GPU、ワーカー数の数字が単に速いだけでなく比較可能であるように、いくつかの選択を意図的に行っている:

- **スイープ内では、都度スキャンし直すのではなく固定したファイルリストを使う。** `--source-dir`は1回限りの実行には便利だが(上記参照)、*スイープ*内のすべての設定は同一の明示的な`--filelist`を指すようにしてあり、全設定が同一ファイルを同一順序で処理する。元のデモコードの`random.shuffle`はそうしないと実行ごとの計測時間を歪めてしまう。
- **サイズで層化したサブセット。** より小さいスイープ用サブセットは、ファイルサイズでソートして全域から均等な間隔で抽出しており、`[::k]`のような単純なスライスではない――単純なスライスは最大級のファイルを切り落とし、サブセットを実態より軽く(スイープの数字を実態より良く)見せてしまう。
- **1設定=1プロセス。** 各計測は新しいPythonプロセスを起動するので、CUDAコンテキスト・ワーカープール・アロケータの状態が前の実行から漏れて次の計測の数字に混ざることがない。
- **Step 2は`early_stopping=False`、固定のバッチ順序(`train_shuffle=False`、`train_seed=1234`)、固定のepoch数を使う。** これにより壁時計時間の差が純粋なデバイス/パラメータの比較になり、lossカーブがどこで平坦化したかという偶然のアーティファクトにならない。

## 既知の落とし穴

| 症状 | 原因・対処 |
|---|---|
| `HOOPS_AI_LICENSE environment variable is required` | 上記の`.env`を作成する |
| `Could not find ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` | `--ckpt`または`HOOPS_AI_CKPT`を明示的に指定する |
| `--accelerator gpu requested but torch.cuda.is_available() is False` | CPU専用のvenvで実行している。GPU側を使う |
| `torch.cuda.is_available()`が`True`を返すのに、GPUでの実行がすべて`no kernel image is available for execution on the device`で失敗する | インストールされているtorchビルドが、そのGPUのcompute capability向けのカーネルを含んでいない(sm_75以降しか含まないtorchビルドを、古いPascal世代のカードで使った際に確認)。自分のGPUに合ったCUDA/カーネル対応のtorchビルドを入れる(`torch.cuda.get_device_capability()`をそのwheelが対応するアーキテクチャと突き合わせる)か、CPUのみでベンチマークする――「このGPUは現行のHOOPS AIのtorch要件に対応していない」という結論自体が有用な成果になる。`bench_common.cuda_usable()`は`is_available()`を信用せず実際にmatmulを起動して検証し、step 3は使用不能なGPUを自動的にマスク(`CUDA_VISIBLE_DEVICES=""`)するので、全ファイルで単に失敗し続けることはない |
| step 2が`CUDA out of memory`を起こす | その`batch_size`がVRAM上限を超えている。スイープハーネスは`FAILED`として記録し次へ進む |
| step 1/3のスループットが実行ごとにばらつく | Windowsでリアルタイムのウイルス対策がSTEPファイルをスキャンしている可能性。コーパスと出力フォルダを除外設定に入れる |
| スイープが`-DryRun`の見積もりよりかなり遅い | 見積もりはPhase 1の実測値に置き換わるまで、1ファイルあたり約34コア秒という前提を使っている |
| ヘッドレスなLinuxボックスで最初のCADファイルの時点でハングまたはエラーになる | Xディスプレイが無い――HOOPS AIはオフスクリーンでもXディスプレイを要求する。`Xvfb :99`を起動して`export DISPLAY=:99`する(重量級コーパス用の2つのシェルスクリプトはどちらも自動的にこれを行う) |
| step 1が`RAM limit reached ... Restarting workers`や`worker still alive after kill attempt`を繰り返すだけで一向に進まない | 実際に使える空きRAMに対して`--max-workers`が多すぎる。kill仕切れなかったワーカーがRAMを保持したまま新しいワーカーが起動されるため、再起動のたびに空きRAMが回復するどころか減っていく。`--max-workers`を減らす(数十ファイル程度なら12は要らない)、再試行前にタスクマネージャー/`ps`で前回の中断実行から残っているワーカープロセスが無いか確認する、他のRAMを食うアプリケーションを閉じる |

## ソースコード

このリポジトリは出発点として提供するものであり、本番品質のコードではない――気にするデータやインフラを対象に使う前に、(特に`run_pipeline.py`や`cdk/`のCDKスタックを)よく確認すること。AWS側のビルド/セットアップの詳細はここには重複して書かず、[`cdk/README.md`](cdk/README.md)に記載している。

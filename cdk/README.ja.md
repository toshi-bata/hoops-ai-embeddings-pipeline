# cdk — HOOPS AI embeddings pipeline 用のGPUマシンを構築する

*(English: [README.md](README.md))*

HOOPS AI embeddingsの3ステップパイプライン(encode / train / index)を、本番規模あるいはベンチマーク規模でCPU/GPU両方で実行し、同一ハードウェア上で比較できるようにする単一のEC2インスタンスを構築するAWS CDK(TypeScript)プロジェクト。姉妹プロジェクト[`hvw-cdk`](https://github.com/toshi-bata/hvw-cdk)のデプロイパターンを踏襲している。

## `cdk deploy`が作成するもの

- 最小構成のVPC(パブリックサブネット1つ、NATゲートウェイなし)。
- **SSH(22)のみ**を開けたセキュリティグループ。`allowedSshCidr`で制限。それ以外は一切公開しない――このマシンはサービスではなくバッチジョブを動かすため。
- **g6.8xlarge**(デフォルト)のUbuntu 24.04 LTSインスタンス。user-dataで以下をブートストラップ:
  - NVIDIAドライバ(`nvidia-driver-580-server`)
  - `:99`でsystemdサービスとして動くXvfb(HOOPS AIはヘッドレス/オフスクリーン処理でもXディスプレイを必要とする)
  - 2つのPython venv、`~/CPU1.1/.venv`と`~/GPU1.1/.venv`。それぞれに[公式インストール手順](https://docs.techsoft3d.com/hoops/ai/getting_started/install_pip.html)通り`hoops-ai[all]`(対応するtorchビルドも含む)をpipインストール済み――インストール自体にダウンロード・ライセンスは不要
  - `~/bench`にcloneされるこのリポジトリ
- 300 GiBの暗号化GP3ルートボリューム(デフォルト。`volumeSize`で上書き可能)――CADコーパス・両方のvenv・チェックポイント・パイプライン出力を見込んだサイズ。
- SSMコアポリシー付きのIAMロール。SSHをインターネットに公開せずSession Manager経由で接続できる。

## 意図的に行わないこと

- **CADコーパスは転送しない。** 顧客データであるため、インスタンス起動後に自分で`scp`/`rsync`でコピーする(下記参照)。
- **HOOPS AIのライセンスは有効化しない。** `hoops-ai[all]`は両方のvenvに無条件でインストールされる(pipインストール自体に認証情報は不要)が、実行時にライセンスキーを有効化しないとSDKは何も動作しない。`hoopsAiLicense`(context)/`HOOPS_AI_LICENSE`(環境変数)を渡せば`~/bench/.env`に自動で書き込まれる。渡さなければ、インスタンス起動後に自分でそのファイルを作成する。
- **インスタンスの再起動は自動で行わない。** NVIDIAドライバのインストール後、`nvidia-smi`がGPUを認識するには1回の再起動が必要(下記「初回ログイン」参照)。

## 前提条件

- Node.js / npm(Node 22で動作確認済み)
- AWS CDK v2(`npx cdk`、`package.json`にバージョン固定)
- 対象アカウントのAWS認証情報(`aws configure`またはSSOプロファイル)
- HOOPS AIのライセンスキー(pipインストール自体は公開されており、キーは実行時のSDK有効化にのみ必要)

## 設定(`cdk deploy -c key=value`で上書き)

| contextキー | デフォルト | 説明 |
|---|---|---|
| `instanceType` | `g6.8xlarge` | EC2インスタンスタイプ |
| `volumeSize` | `300` | ルートEBSサイズ(GiB) |
| `allowedSshCidr` | `0.0.0.0/0` | ポート22への接続を許可するCIDR――**自分のIPに絞ることを推奨**(下記参照) |
| `keyName` | (なし) | 既存のEC2キーペア名。省略するとSession Managerを使う |
| `repoUrl` | このリポジトリのGitHub URL | インスタンス上の`~/bench`にcloneされるリポジトリ。環境変数`HOOPS_AI_PIPELINE_REPO_URL`でも指定可 |
| `hoopsAiLicense` | (なし) | ライセンスキー。インスタンス上の`~/bench/.env`に書き込まれる。環境変数`HOOPS_AI_LICENSE`でも指定可 |

### `allowedSshCidr`用に自分のIPを調べる

```powershell
(Invoke-RestMethod https://checkip.amazonaws.com).Trim()   # 例: 203.0.113.42
```

結果を`-c allowedSshCidr=<自分のIP>/32`として使う。

### デプロイ

これらのコマンドは、Node/npmとAWS CLIが入っている場所――通常は自分のWindowsマシン――から実行する。EC2インスタンス自体の上ではなく、そこからAWSへデプロイする形。PowerShellで示しており、bash相当の環境変数構文は行内に注記している。

```powershell
cd cdk
npm install

# デプロイ先アカウント/リージョンのAWS認証情報――CDKはここからアカウント・
# リージョンを自動解決する。CDK_DEFAULT_ACCOUNT/REGIONを手動で設定しても
# 意味がない(このステップで失敗する場合は下記「トラブルシューティング」参照)
$env:AWS_PROFILE = "<your-profile>"
aws sso login --profile <your-profile>   # SSOセッションが失効している場合のみ。下のaws stsが通るなら不要
aws sts get-caller-identity

# 初回のみ
npx cdk bootstrap aws://<ACCOUNT_ID>/ap-northeast-1

# 任意: ライセンスを自動有効化する場合(渡さなければ~/bench/.envは自分で作成する)
$env:HOOPS_AI_LICENSE = '<your-license-key>'

npx cdk deploy -c keyName=<your-key-pair> -c allowedSshCidr=<your-ip>/32 --require-approval never
```

bashの場合: `$env:AWS_PROFILE = "..."`のような行を`export AWS_PROFILE=...`/`export HOOPS_AI_LICENSE='...'`等に置き換える。

`userDataCausesReplacement: true`のため、`assets/user-data.sh`を編集して再デプロイすると**インスタンスが置き換わる**(`InstanceId`が変わる)。これによりブートストラップは常にクリーンな状態から再実行される。

### トラブルシューティング: SSO/プロファイル関連のエラー

SSO(IAM Identity Center)のプロファイルを使っている場合、`aws sso login --profile <name>`が成功しても、それだけで以降のコマンドがそのプロファイルを使うわけではない。同じターミナル内で、以下すべてのコマンド(`aws sts get-caller-identity`、`npx cdk bootstrap`、`npx cdk deploy`)について、`$env:AWS_PROFILE`(または各コマンドへの明示的な`--profile <name>`)がログインしたプロファイルと一致している必要がある。一致していない(別のプロファイルを指している、または未設定)場合、次のいずれかのエラーになる――どちらも「今ログインしたセッションが本当に失効した」ことを意味するわけではない:

- `aws`コマンド: `The SSO session associated with this profile has expired or is otherwise invalid` ―― ほとんどの場合、`$env:AWS_PROFILE`が今ログインしたプロファイルとは別の(場合によっては本当に失効している)プロファイルを指していることが原因で、今回のログイン自体が失敗したわけではない。
- `cdk deploy`/`cdk synth`: `Unable to resolve AWS account to use` ―― CDKが認証情報を全く解決できなかったという意味で、原因は同じ。`$env:CDK_DEFAULT_ACCOUNT`/`CDK_DEFAULT_REGION`を手動で設定しても直らない。`$env:AWS_PROFILE`を直し、CDK自身に解決させること。

`cdk deploy`を再実行する前に、`aws sts get-caller-identity`(`--profile`無しで、現在の`$env:AWS_PROFILE`をそのまま反映する)で、期待するアカウント/ロールが返ってくることを確認すること。

また、同じSSO開始URL/リージョンを共有していても、あるプロファイルでのSSOログインが他のプロファイルにそのまま引き継がれるとは限らない。使うプロファイルごとに`aws sso login --profile <name>`を行うこと(1回のログインで全部済むと決めつけない)。

## 初回ログイン

```bash
ssh -i <path-to-key>.pem ubuntu@<PublicDnsName>       # または: aws ssm start-session --target <InstanceId>

# 1. GPUドライバが読み込まれているか確認――エラーなら1回再起動して再試行
nvidia-smi

# 2. デプロイ直後ならブートストラップログを確認
sudo tail -f /var/log/cloud-init-output.log

# 3. CADコーパスを転送する(自分のマシンから、EC2上ではなく)
rsync -avz --progress /local/path/to/corpus/ ubuntu@<PublicDnsName>:~/dataset/

# 4. hoops_aiがimportでき、(hoopsAiLicenseを渡していれば)ライセンス有効化されているか確認
~/CPU1.1/.venv/bin/python -c "import hoops_ai; print(hoops_ai.__version__)"
```

デプロイ時に`hoopsAiLicense`を渡さなかった場合は、パイプラインを実行する前に`~/bench/.env`を自分で作成し、`HOOPS_AI_LICENSE='<your-license-key>'`を書いておく――詳細はリポジトリ直下READMEの「前提条件」を参照。

## パイプラインの実行

コーパスの配置が終わったら、`run_pipeline.py`(1回通しの本番実行)と`run_heavy_batch.sh`/`run_heavy_scaling.sh`(CPU-vs-GPUスイープ)の使い方はリポジトリ直下の[README](../README.ja.md)を参照。

## 後始末

```bash
npx cdk destroy
```

インスタンスとそのEBSボリュームを削除する。事前にCADコーパスの元の場所に書いたもの、あるいはインスタンスからコピーして持ち出したものには影響しない。インスタンスのディスクにしか無かったものは失われる。

## ファイル構成

```
cdk/
├── bin/embeddings-cdk.ts         # CDKアプリのエントリポイント
├── lib/embeddings-cdk-stack.ts   # スタック定義(VPC/SG/EC2/IAM)
├── assets/user-data.sh           # EC2ブートストラップ(ドライバ、Xvfb、venv+hoops-aiインストール、リポジトリclone)
└── README.md
```

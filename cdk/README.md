# cdk — provision a GPU box for the HOOPS AI embeddings pipeline

*(日本語版: [README.ja.md](README.ja.md))*

AWS CDK (TypeScript) project that provisions a single EC2 instance to run the
3-step HOOPS AI embeddings pipeline (encode / train / index) at production or
benchmark scale, on CPU and GPU, so the two can be compared on identical
hardware. Modeled on the companion [`hvw-cdk`](https://github.com/toshi-bata/hvw-cdk)
project's deploy pattern.

## What `cdk deploy` creates

- A minimal VPC (single public subnet, no NAT gateway).
- A security group that opens **only SSH (22)**, restricted to `allowedSshCidr`.
  Nothing else is exposed -- this box runs batch jobs, not a service.
- A **g6.8xlarge** (default) Ubuntu 24.04 LTS instance, bootstrapped via
  user-data with:
  - the NVIDIA driver (`nvidia-driver-580-server`)
  - Xvfb running as a systemd service on `:99` (HOOPS AI needs an X display
    even for headless/offscreen work)
  - two Python venvs, `~/CPU1.1/.venv` and `~/GPU1.1/.venv`, each with
    `hoops-ai[all]` (and the matching torch build) pip-installed per the
    [official install docs](https://docs.techsoft3d.com/hoops/ai/getting_started/install_pip.html)
    -- no download/license needed for the install itself
  - this repo, cloned into `~/bench`
- A 300 GiB encrypted GP3 root volume (default; override with `volumeSize`) --
  sized for the CAD corpus, both venvs, checkpoints and pipeline output.
- An IAM role with the SSM core policy, so you can connect via Session
  Manager without opening SSH to the world.

## What it deliberately does NOT do

- **It does not transfer your CAD corpus.** That's customer data; copy it
  yourself with `scp`/`rsync` after the instance is up (see below).
- **It does not activate a HOOPS AI license.** `hoops-ai[all]` installs into
  both venvs unconditionally (the pip install needs no credentials), but the
  SDK won't do anything useful without a license key activated at runtime.
  Pass one via `hoopsAiLicense` (context) / `HOOPS_AI_LICENSE` (env) to have
  it written to `~/bench/.env` automatically, or create that file yourself
  after the instance is up.
- **It does not reboot the instance for you.** The NVIDIA driver install
  needs one reboot before `nvidia-smi` reports the GPU (see "First login"
  below).

## Prerequisites

- Node.js / npm (tested with Node 22)
- AWS CDK v2 (`npx cdk`, pinned in `package.json`)
- AWS credentials for the target account (`aws configure` or SSO profile)
- A HOOPS AI license key (the pip install itself is public; the key is only
  needed to activate the SDK at runtime)

## Configuration (override with `cdk deploy -c key=value`)

| context key | default | description |
|---|---|---|
| `instanceType` | `g6.8xlarge` | EC2 instance type |
| `volumeSize` | `300` | root EBS size (GiB) |
| `allowedSshCidr` | `0.0.0.0/0` | CIDR allowed to reach port 22 -- **restrict this to your own IP** (see below) |
| `keyName` | (none) | existing EC2 key pair name; omit to use Session Manager instead |
| `repoUrl` | this repo's GitHub URL | repo cloned into `~/bench` on the instance. Env var `HOOPS_AI_PIPELINE_REPO_URL` also works |
| `hoopsAiLicense` | (none) | license key, written to `~/bench/.env` on the instance. Env var `HOOPS_AI_LICENSE` also works |

### Find your IP for `allowedSshCidr`

```powershell
(Invoke-RestMethod https://checkip.amazonaws.com).Trim()   # e.g. 203.0.113.42
```

Use the result as `-c allowedSshCidr=<your-ip>/32`.

### Deploy

These commands are run from wherever you have Node/npm and the AWS CLI --
typically your own Windows machine, deploying to AWS, not on the EC2 instance
itself. Shown in PowerShell; bash-equivalent env var syntax is noted inline.

```powershell
cd cdk
npm install

# AWS credentials for the target account/region -- CDK resolves the account
# and region from THIS, automatically; don't set CDK_DEFAULT_ACCOUNT/REGION
# by hand, it won't help (see "Troubleshooting" below if this step fails)
$env:AWS_PROFILE = "<your-profile>"
aws sso login --profile <your-profile>   # only if your SSO session expired; skip if aws sts below already works
aws sts get-caller-identity

# first time only
npx cdk bootstrap aws://<ACCOUNT_ID>/ap-northeast-1

# optional: auto-activate the license (skip to leave ~/bench/.env for you to create by hand)
$env:HOOPS_AI_LICENSE = '<your-license-key>'

npx cdk deploy -c keyName=<your-key-pair> -c allowedSshCidr=<your-ip>/32 --require-approval never
```

`cdk deploy` returns once the CloudFormation stack finishes creating -- which
is *before* user-data.sh has finished running on the instance (driver
install, `apt upgrade`, and `pip install "hoops-ai[all]"` alone take several
minutes). Checking `nvidia-smi` / `hoops_ai` right after `cdk deploy` returns
will look broken when it's really just still running. To have the terminal
block until bootstrap has actually finished, capture the outputs and SSH in
to wait on `cloud-init` (requires `keyName`; there's no SSM equivalent of
this one-liner):

```powershell
npx cdk deploy -c keyName=<your-key-pair> -c allowedSshCidr=<your-ip>/32 `
    --require-approval never --outputs-file outputs.json

$dns = (Get-Content outputs.json | ConvertFrom-Json).HoopsAiEmbeddingsPipelineStack.PublicDnsName
ssh -i <path-to-key>.pem -o StrictHostKeyChecking=no ubuntu@$dns "cloud-init status --wait; cloud-init status --long"
```

The prompt won't return until bootstrap is done; the final `cloud-init
status --long` line tells you `done` or `error` (and which module failed, if
any) without needing a separate SSH session.

bash: replace the `$env:AWS_PROFILE = "..."`-style lines with
`export AWS_PROFILE=...` / `export HOOPS_AI_LICENSE='...'` / etc.

`userDataCausesReplacement: true` means editing `assets/user-data.sh` and
redeploying **replaces the instance** (new `InstanceId`), so bootstrap always
reruns from a clean box.

### Troubleshooting: SSO / profile errors

If you're on an SSO (IAM Identity Center) profile, `aws sso login --profile
<name>` succeeding does NOT mean subsequent commands automatically use that
profile. `$env:AWS_PROFILE` (or an explicit `--profile <name>` on every
command) has to match the profile you logged into, in the SAME terminal, for
every command above -- `aws sts get-caller-identity`, `npx cdk bootstrap`,
`npx cdk deploy`. If it doesn't match (wrong profile, or unset), you'll see
one of these, neither of which means your login actually expired:

- `aws` commands: `The SSO session associated with this profile has expired
  or is otherwise invalid` -- almost always means `$env:AWS_PROFILE` points
  at a different (possibly genuinely expired) profile than the one you just
  logged into, not that the fresh login failed.
- `cdk deploy`/`cdk synth`: `Unable to resolve AWS account to use` -- CDK
  couldn't resolve credentials at all, same root cause. Setting
  `$env:CDK_DEFAULT_ACCOUNT`/`CDK_DEFAULT_REGION` by hand does not fix this;
  fix `$env:AWS_PROFILE` instead and let CDK resolve them itself.

Confirm with `aws sts get-caller-identity` (no `--profile` flag, so it
reflects whatever `$env:AWS_PROFILE` currently is) before re-running
`cdk deploy` -- it should print the account/role you expect, not an error.

Also note SSO login does not reliably carry over between different profiles
even when they share the same SSO start URL/region -- log into each profile
you use (`aws sso login --profile <name>`) rather than assuming one login
covers all of them.

## First login

```bash
ssh -i <path-to-key>.pem ubuntu@<PublicDnsName>       # or: aws ssm start-session --target <InstanceId>

# 1. verify the GPU driver loaded -- if this errors, reboot once and retry
nvidia-smi

# 2. watch the bootstrap log if you just deployed
sudo tail -f /var/log/cloud-init-output.log

# 3. copy your CAD corpus in (from your own machine, not the EC2 box)
rsync -avz --progress /local/path/to/corpus/ ubuntu@<PublicDnsName>:~/dataset/

# 4. confirm hoops_ai is importable and (if you passed hoopsAiLicense) licensed
~/CPU1.1/.venv/bin/python -c "import hoops_ai; print(hoops_ai.__version__)"
```

If you didn't pass `hoopsAiLicense` at deploy time, create `~/bench/.env`
yourself with `HOOPS_AI_LICENSE='<your-license-key>'` before running the
pipeline -- see the repo root README's "Prerequisites".

## Running the pipeline

See the repo root [README](../README.md) for `run_pipeline.py` (single
end-to-end run) and `run_heavy_batch.sh` / `run_heavy_scaling.sh`
(CPU-vs-GPU sweeps) usage once the corpus is in place.

## Teardown

```bash
npx cdk destroy
```

This deletes the instance and its EBS volume. Anything you only ever wrote to
the CAD corpus location or copied off the box beforehand is unaffected;
anything left only on the instance's disk is gone.

## File layout

```
cdk/
├── bin/embeddings-cdk.ts         # CDK app entry point
├── lib/embeddings-cdk-stack.ts   # stack definition (VPC/SG/EC2/IAM)
├── assets/user-data.sh           # EC2 bootstrap (driver, Xvfb, venvs + hoops-ai install, repo clone)
└── README.md
```

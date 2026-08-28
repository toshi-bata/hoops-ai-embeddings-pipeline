# cdk — provision a GPU box for the HOOPS AI embeddings pipeline

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
  - two Python venvs, `~/CPU1.1/.venv` (CPU-only torch) and `~/GPU1.1/.venv`
    (CUDA 13.0 torch)
  - this repo, cloned into `~/bench`
- A 300 GiB encrypted GP3 root volume (default; override with `volumeSize`) --
  sized for the CAD corpus, both venvs, checkpoints and pipeline output.
- An IAM role with the SSM core policy, so you can connect via Session
  Manager without opening SSH to the world.

## What it deliberately does NOT do

- **It does not transfer your CAD corpus.** That's customer data; copy it
  yourself with `scp`/`rsync` after the instance is up (see below).
- **It does not install the HOOPS AI SDK** (the `hoops_ai` package) unless you
  pass a short-lived download URL at deploy time (`sdkUrl` context / env
  `HOOPS_AI_SDK_URL`). It's licensed software and is never baked into this
  public repo.
- **It does not reboot the instance for you.** The NVIDIA driver install
  needs one reboot before `nvidia-smi` reports the GPU (see "First login"
  below).

## Prerequisites

- Node.js / npm (tested with Node 22)
- AWS CDK v2 (`npx cdk`, pinned in `package.json`)
- AWS credentials for the target account (`aws configure` or SSO profile)
- A licensed HOOPS AI SDK build for Linux, and a license key

## Configuration (override with `cdk deploy -c key=value`)

| context key | default | description |
|---|---|---|
| `instanceType` | `g6.8xlarge` | EC2 instance type |
| `volumeSize` | `300` | root EBS size (GiB) |
| `allowedSshCidr` | `0.0.0.0/0` | CIDR allowed to reach port 22 -- **restrict this to your own IP** (see below) |
| `keyName` | (none) | existing EC2 key pair name; omit to use Session Manager instead |
| `repoUrl` | this repo's GitHub URL | repo cloned into `~/bench` on the instance. Env var `HOOPS_AI_PIPELINE_REPO_URL` also works |
| `sdkUrl` | (none) | presigned download URL for the HOOPS AI SDK archive. Installs the SDK automatically when set. Env var `HOOPS_AI_SDK_URL` also works |
| `hoopsAiLicense` | (none) | license key, written to `~/bench/.env` on the instance (only takes effect together with `sdkUrl`). Env var `HOOPS_AI_LICENSE` also works |

### Find your IP for `allowedSshCidr`

```bash
curl -s https://checkip.amazonaws.com   # e.g. 14.3.142.47
```

Use the result as `-c allowedSshCidr=<your-ip>/32`.

### Deploy

```bash
cd cdk
npm install

# AWS credentials for the target account/region
export AWS_PROFILE=<your-profile>
export CDK_DEFAULT_REGION=ap-northeast-1
aws sts get-caller-identity

# first time only
npx cdk bootstrap aws://<ACCOUNT_ID>/ap-northeast-1

# presigned SDK URL contains a query string with an ampersand -- quote it
export HOOPS_AI_SDK_URL='https://.../hoops-ai-sdk-linux.tar.gz?X-Amz-...'
export HOOPS_AI_LICENSE='<your-license-key>'

npx cdk deploy -c keyName=<your-key-pair> -c allowedSshCidr=<your-ip>/32 --require-approval never
```

If you omit `HOOPS_AI_SDK_URL`, the instance still comes up with both venvs
and the repo cloned -- you'll just need to `pip install` the SDK into
`~/CPU1.1/.venv` and `~/GPU1.1/.venv` by hand afterwards.

`userDataCausesReplacement: true` means editing `assets/user-data.sh` or
`assets/install-sdk.sh` and redeploying **replaces the instance** (new
`InstanceId`), so bootstrap always reruns from a clean box.

## First login

```bash
ssh -i <path-to-key>.pem ubuntu@<PublicDnsName>       # or: aws ssm start-session --target <InstanceId>

# 1. verify the GPU driver loaded -- if this errors, reboot once and retry
nvidia-smi

# 2. watch the bootstrap log if you just deployed
sudo tail -f /var/log/cloud-init-output.log

# 3. copy your CAD corpus in (from your own machine, not the EC2 box)
rsync -avz --progress /local/path/to/corpus/ ubuntu@<PublicDnsName>:~/dataset/
```

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
├── assets/user-data.sh           # EC2 bootstrap (driver, Xvfb, venvs, repo clone)
├── assets/install-sdk.sh         # optional SDK download + install (when sdkUrl is set)
└── README.md
```

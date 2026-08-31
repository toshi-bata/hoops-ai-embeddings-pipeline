import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';

/**
 * Provisions a single GPU EC2 box to run the HOOPS AI embeddings pipeline
 * (encode -> train -> index) at production/benchmark scale, following the
 * pattern established by the companion `hvw-cdk` project.
 *
 * What this stack creates:
 *   - A minimal VPC with a single public subnet (no NAT gateway).
 *   - A security group opening only 22 (SSH), restricted to `allowedSshCidr`.
 *     No web server runs on this box, so nothing else is exposed.
 *   - A g6.8xlarge (default) Ubuntu 24.04 LTS EC2 instance, bootstrapped via
 *     user-data with the NVIDIA driver, Xvfb (HOOPS AI needs an X display
 *     even for offscreen/headless work), and two Python venvs -- CPU1.1 and
 *     GPU1.1 -- each with `hoops-ai[all]` (and the matching torch build)
 *     pip-installed per
 *     https://docs.techsoft3d.com/hoops/ai/getting_started/install_pip.html,
 *     and this repo cloned into ~/bench.
 *   - An IAM role with the SSM core policy, so you can connect via Session
 *     Manager without opening SSH to the world.
 *   - A large encrypted GP3 root volume (default 300 GiB) for the CAD
 *     corpus, venvs, checkpoints and benchmark output.
 *
 * What this stack deliberately does NOT do:
 *   - It does not transfer your CAD corpus. That is customer data; copy it
 *     yourself with scp/rsync after the instance is up (see README).
 *   - It does not activate a HOOPS AI license. The pip install itself needs
 *     no credentials, but the SDK is inert without a license key at runtime.
 *     Pass one via `hoopsAiLicense` (context) / HOOPS_AI_LICENSE (env) to
 *     have it written to ~/bench/.env automatically, or create that file
 *     yourself after the instance is up.
 *   - The NVIDIA driver install typically needs one reboot before
 *     `nvidia-smi` reports the GPU; see README for the one-time
 *     verification step.
 */
export class EmbeddingsBenchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ------------------------------------------------------------------
    // Configuration (override with `cdk deploy -c key=value`)
    // ------------------------------------------------------------------
    const instanceTypeCtx = (this.node.tryGetContext('instanceType') as string) ?? 'g6.8xlarge';
    const volumeSizeGiB = Number(this.node.tryGetContext('volumeSize') ?? 300);
    const sshCidr = (this.node.tryGetContext('allowedSshCidr') as string) ?? '0.0.0.0/0';
    const keyName = this.node.tryGetContext('keyName') as string | undefined;
    const repoUrl =
      (this.node.tryGetContext('repoUrl') as string) ??
      process.env.HOOPS_AI_PIPELINE_REPO_URL ??
      'https://github.com/toshi-bata/hoops-ai-embeddings-pipeline.git';

    // ------------------------------------------------------------------
    // Networking: minimal public VPC (single AZ, no NAT to keep cost down)
    // ------------------------------------------------------------------
    const vpc = new ec2.Vpc(this, 'BenchVpc', {
      maxAzs: 1,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
    });

    // ------------------------------------------------------------------
    // Security group: SSH only. This box runs batch jobs, not a service.
    // ------------------------------------------------------------------
    const sg = new ec2.SecurityGroup(this, 'BenchSecurityGroup', {
      vpc,
      description: 'HOOPS AI embeddings pipeline benchmark box: SSH only',
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.ipv4(sshCidr), ec2.Port.tcp(22), 'SSH');

    // ------------------------------------------------------------------
    // IAM role (Session Manager access, no bastion required)
    // ------------------------------------------------------------------
    const role = new iam.Role(this, 'BenchInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // ------------------------------------------------------------------
    // AMI: Ubuntu Server 24.04 LTS (amd64) via Canonical's public SSM param.
    // The NVIDIA driver is installed by user-data, not baked into the AMI,
    // so this resolves correctly in any region without an AMI-ID lookup.
    // ------------------------------------------------------------------
    const ubuntu = ec2.MachineImage.fromSsmParameter(
      '/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id',
      { os: ec2.OperatingSystemType.LINUX },
    );

    // ------------------------------------------------------------------
    // User-data bootstrap script
    // ------------------------------------------------------------------
    const rawUserDataScript = fs.readFileSync(
      path.join(__dirname, '..', 'assets', 'user-data.sh'),
      'utf8',
    );
    // user-data.sh's own git-clone step needs BENCH_REPO_URL to already be
    // set when IT runs. UserData.custom(content).addCommands(...) APPENDS
    // new lines after the custom content (CustomUserData just joins `lines`
    // in push order) -- it does not inject them before it. Calling
    // addCommands() here for the export would define BENCH_REPO_URL only
    // after the whole script, including the clone step, had already run,
    // silently skipping the clone (the `if [[ -n "${BENCH_REPO_URL:-}" ]]`
    // check would see it unset and do nothing -- no error, so this is easy
    // to miss until you notice ~/bench is empty). Splice the export in right
    // after the shebang line instead, so it's set before anything else runs.
    const userDataScript = rawUserDataScript.replace(
      /^(#!.*\n)/,
      `$1export BENCH_REPO_URL='${repoUrl}'\n`,
    );
    const userData = ec2.UserData.custom(userDataScript);

    // Optional license key: write ~/bench/.env once the repo has been cloned
    // (this step is appended AFTER the whole script above runs, which is
    // fine here since it only needs $BENCH_DIR to already exist -- unlike
    // BENCH_REPO_URL, nothing earlier in user-data.sh depends on this).
    // hoops-ai itself installs unconditionally above; this only controls
    // whether the license is auto-activated or left for you to add by hand.
    const licenseKey =
      (this.node.tryGetContext('hoopsAiLicense') as string) ?? process.env.HOOPS_AI_LICENSE;
    if (licenseKey) {
      // user-data.sh runs with `set -x`; disable tracing before emitting the
      // license key so it's never written to /var/log/cloud-init-output.log
      // in clear text. Write as root then chown, since `>>` on a not-yet-
      // existing file would otherwise create it as root (same trap the
      // BENCH_DIR mkdir hit -- see user-data.sh step 4).
      userData.addCommands(
        'set +x',
        `echo "HOOPS_AI_LICENSE='${licenseKey}'" >> /home/ubuntu/bench/.env`,
        'chown ubuntu:ubuntu /home/ubuntu/bench/.env',
        'set -x',
      );
    }

    // Optional existing EC2 key pair for SSH access.
    const keyPair = keyName
      ? ec2.KeyPair.fromKeyPairName(this, 'BenchKeyPair', keyName)
      : undefined;

    // ------------------------------------------------------------------
    // EC2 instance
    // ------------------------------------------------------------------
    const instance = new ec2.Instance(this, 'BenchInstance', {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceTypeCtx),
      machineImage: ubuntu,
      securityGroup: sg,
      role,
      userData,
      userDataCausesReplacement: true,
      keyPair,
      blockDevices: [
        {
          deviceName: '/dev/sda1',
          volume: ec2.BlockDeviceVolume.ebs(volumeSizeGiB, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
    });

    // ------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------
    new cdk.CfnOutput(this, 'InstanceId', { value: instance.instanceId });
    new cdk.CfnOutput(this, 'PublicDnsName', {
      value: instance.instancePublicDnsName,
      description: 'Changes if the instance is replaced (e.g. by a user-data edit)',
    });
    new cdk.CfnOutput(this, 'SshCommand', {
      value: keyName
        ? `ssh -i <path-to-${keyName}.pem> ubuntu@${instance.instancePublicDnsName}`
        : '(no keyName provided) use AWS Session Manager: aws ssm start-session --target ' +
          instance.instanceId,
    });
    new cdk.CfnOutput(this, 'NextSteps', {
      value:
        'ssh in, run `nvidia-smi` to confirm the GPU driver loaded (reboot once if not), ' +
        'copy your CAD corpus into ~/dataset (scp/rsync), then see the repo README for ' +
        'run_pipeline.py / run_heavy_batch.sh usage',
    });
  }
}

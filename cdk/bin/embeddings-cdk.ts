#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { EmbeddingsBenchStack } from '../lib/embeddings-cdk-stack';

const app = new cdk.App();
new EmbeddingsBenchStack(app, 'HoopsAiEmbeddingsPipelineStack', {
  // Bind to the account/region from your AWS CLI/profile so the Ubuntu AMI
  // SSM parameter resolves in the correct region at deploy time.
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

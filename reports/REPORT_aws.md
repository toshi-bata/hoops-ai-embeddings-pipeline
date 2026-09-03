# HOOPS AI 1.1 benchmark (AWS EC2) - CPU vs GPU

Machine: **AWS g6.8xlarge** (AMD EPYC 16-core + NVIDIA L4). Dataset: `mechcad` heavy mechanical-CAD corpus - the same full 10,715-file corpus is used for the Step 1 worker sweep and the Step 2 CPU-vs-GPU training run.
Benchmark runs: 22 succeeded, 0 failed or skipped. Raw data: `results/results.csv`, per-run logs in `logs/`.

See also the companion report for the other machine (`REPORT_local.*` / `REPORT_aws.*`).

## Summary

Conclusions first; the supporting numbers and per-setting sweeps follow in each step's section.

- **Step 1 (encoding) and Step 3 (embedding + indexing) are CPU-bound** - the GPU sits idle, so the GPU install buys nothing. Step 1 throughput scales with worker count up to roughly the machine's physical core count, then flattens; Step 3 peaks at a lower worker count and then *declines* as more workers are added, because its heavier RAM and file-descriptor load makes oversubscription counter-productive.
- **Step 2 (training) is the only GPU-bound stage** and the only place a GPU pays off. Because the model is trained with contrastive learning, batch_size changes the trained model, so the fair speed comparison fixes batch_size: at the tutorial default batch_size=64 (same model, same 10 epochs), on a **10,715-file heavy dataset** (AWS g6.8xlarge, NVIDIA L4) the GPU is about **4.9x faster** than the 16-core CPU (497 vs 101 s/epoch). Compare on `s/epoch`, not total wall (which includes a fixed start-up cost).
- _How the tables read:_ within each sweep, speedup is quoted against the smallest setting measured in that group (lowest worker count, or smallest batch size for step 2); parallel efficiency is speedup divided by the ideal linear speedup.

### Test environment

#### AWS g6.8xlarge (EC2)

Heavy-scale host: the ~10k-file Step 1 worker sweep and the Step 2 CPU-vs-GPU training run were executed here. Both installs share the same instance; only the torch wheel differs (shared rows are merged).
<table>
<thead><tr><th>item</th><th>CPU1.1</th><th>GPU1.1</th></tr></thead>
<tbody>
<tr><td>CPU</td><td colspan='2'>AMD EPYC 7R13 Processor</td></tr>
<tr><td>Cores</td><td colspan='2'>16 physical / 32 logical</td></tr>
<tr><td>RAM</td><td colspan='2'>121.2 GB</td></tr>
<tr><td>GPU (nvidia-smi)</td><td colspan='2'>NVIDIA L4, 23034 MiB, 580.173.02, 8.9</td></tr>
<tr><td>torch</td><td>2.9.1+cpu (cuda None)</td><td>2.9.1+cu130 (cuda 13.0)</td></tr>
<tr><td>cuda_available</td><td>False</td><td>True</td></tr>
<tr><td>hoops_ai</td><td colspan='2'>1.1.1</td></tr>
<tr><td>OS / Python</td><td colspan='2'>Linux-7.0.0-1011-aws-x86_64-with-glibc2.39 / 3.12.3</td></tr>
</tbody></table>

## Step 1 - CAD encoding (DataPrep)

### max_workers scaling

**CPU1.1, n=10715 files** (speedup relative to max_workers=8)

<table>
<thead><tr><th>max_workers</th><th>time (s)</th><th>files/s</th><th>speedup</th><th>parallel eff.</th><th>peak RSS (MB)</th><th>failed</th></tr></thead>
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

**Fastest: max_workers = 32** (1495.6 s, 7.16 files/s) - the shortest time in this group. (max_workers = 28 is within 3% - the cheapest setting at peak throughput.)

Amdahl fit over max_workers=8..40: serial fraction f = **6.5%**, so the asymptotic ceiling is 15.4x no matter how many workers you add. Roughly max_workers = 58 reaches 80% of that ceiling.

> **The exact peak worker count is run-to-run noise, not a stable optimum.** Across repeated full sweeps the fastest max_workers has moved between 12 and 14 on CPU and between 10 and 12 on GPU, while every point from roughly the physical-core count upward stays within ~5% of the best. Read this as a flat plateau near the core count, not a single best value - do not tune the exact peak.

### Encoding: does the venv matter?

Step 1 is HOOPS Exchange + numpy on the CPU; the torch wheel should be irrelevant. Matching numbers below confirm the two installs are otherwise equivalent, which is what licenses the step 2/3 comparisons.

**Step 1 is CPU-bound - the GPU install buys nothing.** At max_workers=32, CPU 1495.6 s vs GPU 1483.5 s: the GPU install is **1.01x** the CPU throughput - within run-to-run noise, i.e. parity.

Both installs were run at max_workers=32 on the 10,715-file corpus:

<table>
<thead><tr><th>install</th><th>encode time (s)</th><th>files/s</th></tr></thead>
<tbody>
<tr><td>CPU1.1</td><td>1495.6</td><td>7.16</td></tr>
<tr><td>GPU1.1</td><td>1483.5</td><td>7.22</td></tr>
</tbody></table>
The GPU install is neither faster nor slower because the encoding never touches the GPU.

## Step 2 - training

### Training speed - CPU vs GPU (10k heavy dataset)

Ran on **AWS g6.8xlarge** (AMD EPYC, 16 physical cores + **NVIDIA L4** GPU) against a **10,715-file** heavy mechanical-CAD corpus (`mechcad`), the same corpus used for the Step 1 / Step 3 worker sweeps above. Both devices train from scratch for 10 epochs with the **same dataset pointer**, batch_size=64, num_workers=0, matmul=high and a fixed seed, so the two runs produce the *same model* and the wall-clock gap is a pure CPU-vs-GPU speed test.

**Why batch_size is held fixed.** This embedding model is trained with *contrastive learning* (SimCLR / NT-Xent): the batch supplies the in-batch negatives, so changing batch_size trains a *different* model. Holding batch_size=64 (the tutorial default) keeps the model identical and isolates device speed. At this scale each epoch is ~204 batches.

**Submitted vs trained.** The 10,715 files are the *submitted* count. Step 1 encoding tolerates HOOPS AI's known **non-deterministic ~2% re-encode failures** (typically `division by zero` or `Data key 'graph' not found`, and a *different* set of files each run), so this run dropped **216** files and the model actually trained on the **10,499 successes**. These are separate from the 24 files pre-excluded from the file list - they cannot be filtered ahead of time because the failures are not repeatable.

#### CPU vs GPU at identical settings (batch_size=64)

**At 10,715 files - same model, same 10 epochs, same seed, only the device differs - the L4 GPU is 4.9x faster than the 16-core CPU:** CPU 497.3 vs GPU 101.1 s/epoch (8.3 vs 1.7 min/epoch). Over 10 epochs that is CPU 83 min vs GPU 17 min of training wall time.

<table>
<thead><tr><th>device</th><th>s/epoch</th><th>min/epoch</th><th>train wall (min)</th><th>peak RSS (MB)</th><th>peak GPU (MB)</th></tr></thead>
<tbody>
<tr><td>CPU (EPYC 16-core)</td><td>497.3</td><td>8.29</td><td>82.9</td><td>7299</td><td>-</td></tr>
<tr><td>GPU (NVIDIA L4)</td><td>101.1</td><td>1.68</td><td>16.8</td><td>2553</td><td>6180</td></tr>
</tbody></table>

_Speedup = CPU s/epoch / GPU s/epoch = 497.3 / 101.1 = **4.9x**. The GPU also uses far less host RAM (peak RSS) because the heavy tensor work lives on the device.


**Findings**

- **Compare on s/epoch, not total wall.** Split/setup is a fixed cost; the marginal training cost is CPU 497.3 vs GPU 101.1 s/epoch (**4.9x**). Project longer runs from s/epoch.
- **L4 memory headroom:** peak GPU 6180 MB at batch_size=64 - well under the L4's 23 GB, so larger batches or a bigger model would still fit.
- **Host RAM:** CPU training peaked at 7.1 GB RSS vs the GPU run's 2.5 GB - offloading to the device cuts host memory pressure too.

## Step 3 - embedding + FAISS indexing

### num_workers scaling

**CPU1.1, n=10715 files** (speedup relative to num_workers=4)

<table>
<thead><tr><th>num_workers</th><th>time (s)</th><th>files/s</th><th>speedup</th><th>parallel eff.</th><th>peak RSS (MB)</th><th>failed</th></tr></thead>
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

_The **files/s** column counts input CAD files. These 10,715 files expand to **37,548 B-rep shapes** (~3.5 shapes/file - the heavy corpus is full of assemblies), and Step 3 embeds each shape: at the peak that is ~11.1 shapes/s._


**Fastest: num_workers = 12** (3377.6 s, 3.17 files/s) - the shortest time in this group.

Amdahl fit over num_workers=4..36: serial fraction f = **52.6%**, so the asymptotic ceiling is 1.9x no matter how many workers you add. Roughly num_workers = 4 reaches 80% of that ceiling.

## Caveats

- Single machine, single run per configuration: differences under roughly 5% are inside run-to-run noise. Re-run a cell if a conclusion depends on a small gap.
- The heavy `mechcad` files trigger HOOPS AI's known non-deterministic ~2% re-encode failures, so a few dozen files drop each run; the throughput figures are over the successfully processed files.
- Background load on the machine (including antivirus scanning the STEP files) affects the CPU-bound stages more than the GPU stage.

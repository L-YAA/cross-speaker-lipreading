# Cross-Speaker Lipreading with Personality-Aware Encoder Modulation

A research-oriented deep learning project for cross-speaker visual speech recognition (VSR), featuring a two-stage framework for speaker personality extraction and AdaIN-based encoder modulation.

## Overview

This repository focuses on improving lipreading robustness under unseen-speaker settings. The project integrates:

- RAFT-based optical flow preprocessing
- Static and dynamic speaker personality feature extraction
- Stage1 personality representation learning
- Stage2 lipreading with AdaIN-modulated visual encoder
- Hydra + PyTorch Lightning training / evaluation
- Checkpoint averaging and beam-search decoding

## Architecture

![Architecture](assets/architecture.png)

The overall pipeline consists of two major stages:

### Stage1: Speaker Personality Feature Extraction
- A static branch extracts appearance-related speaker cues from video frames.
- A dynamic branch models motion-style characteristics from optical flow.
- The objective is to learn disentangled speaker personality representations.

### Stage2: Personality-Aware Lipreading
- Personality codes `ps` and `pd` are computed from the input video.
- The visual encoder is modulated through AdaIN before positional encoding.
- The modified encoder is integrated into an ESPNet-based Conformer lipreading pipeline.

## Repository Structure

```text
cross-speaker-lipreading/
├── assets/
├── configs/
├── scripts/
├── src/
│   ├── stage1/
│   ├── stage2/
│   ├── datamodule/
│   └── experimental/
├── docs/
└── third_party/

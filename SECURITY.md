# Security policy

## Reporting a vulnerability

Please report a suspected vulnerability privately rather than in a public issue: email the maintainers at the
address in [`pyproject.toml`](pyproject.toml), or use the "Report a vulnerability" button on the repository's
Security tab where private vulnerability reporting is enabled. Say what you found, how to reproduce it and which
version it affects. This is a research project kept by a small group, so we answer as soon as we can.

## Supported versions

Only the latest release receives fixes. Until 1.0 the API can change between minor versions.

## Loading a checkpoint runs code

`DeFoGModel.load`, `CrossAttentionAdapter.load`, `PropertyHead.load` and `LearnedSizeDistribution.load` call
`torch.load(..., weights_only=False)`, because the files carry configuration dictionaries, and **unpickling executes
code**. Only load checkpoints you trust, and never one from an unknown source. This is how PyTorch checkpoints that
hold Python objects work, so it is not a vulnerability in this package, but it is the thing to be careful with. The
file format and the compatibility rules are in
[`docs/concepts/adapters.md`](docs/concepts/adapters.md#file-format-compatibility-and-trust).

# Limits

What this package does not do, and where it can go wrong without raising an error. The detailed list, with the
evidence behind each item, is on [Results and caveats](results-and-caveats.md).

- **Molecules are the validated domain.** The core is ordinary graph code, but nothing else has been measured.
- **A few scalar properties.** The recipes train adapters for logP, TPSA, QED and SAScore. Feynman-Kac steering is
  recorded for logP and QED.
- **No trained models ship with the package.** You train the base model, the adapters, the property heads and the size
  models yourself, and the base and the adapters need a GPU.
- **An adapter is tied to its base.** To its vocabulary for certain, and in practice to its weights.
- **An adapter is less accurate than training end to end.** In the recorded comparison, a model trained end to end on
  the property had a lower logP error than an adapter alone. Feynman-Kac steering brings the adapter close, and that
  comparison cannot say which of the two is better. What the adapter buys is one frozen base that serves every
  property.
- **Steering costs diversity.** Feynman-Kac steering lowers the error to the target at the price of less variety among
  the molecules.
- **The gains do not add up.** The gain from RL fine-tuning and the gain from Feynman-Kac steering are not additive.
- **A graph stores no formal charge.** The property of a source SMILES and of the molecule that was decoded from a
  graph can differ, so every recorded result is labelled by the decoded molecule, and the adapter, its head and its
  size model must agree on that.
- **Use kekulized molecules.** The aromatic bond class loses information, and most hard validity failures on an
  aromatic base were kekulization errors.
- **A run can look fine and be wrong.** A Feynman-Kac run that never resampled is just the unguided sampler. An RL run
  can raise its reward while the distribution narrows. Read the sampler's trace, and check the whole distribution, not
  one number.

Every measured number, with its protocol and what it does not support, is on
[Results and caveats](results-and-caveats.md).

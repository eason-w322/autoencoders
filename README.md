# CS182 Homework 8: Autoencoders (Coding Question)

This notebook (`q_autoencoder.ipynb`) explores **self-supervised representation learning with autoencoders**. Each autoencoder is trained to reconstruct its input **without labels**, and then the quality of its learned features is measured with a **linear probe**: a single linear layer trained on the frozen encoder's outputs to classify the data.

## What I implemented

| Component | Description |
|---|---|
| `Autoencoder._build_decoder` | Decoder that mirrors the encoder (e.g. `16 → 64 → 256 → 784`), with activations only *between* linear layers |
| `Autoencoder.forward` | `x → encoder → z → decoder → x̂` |
| `Autoencoder.loss` | Squared reconstruction error summed over features and averaged over the batch: `((x - x̂)**2).sum() / len(x)` |
| `DenoisingAutoencoder.train_step` | Adds i.i.d. Gaussian noise (`std = noise_std`) to the input; the loss is still computed against the **clean** input |
| `MaskedAutoencoder.train_step` | Zeros each input entry independently with probability `mask_prob`; the loss is computed against the **clean** input |
| `Experiment._get_model_accuracy` | Encodes test data with the frozen encoder, classifies with the linear probe, and reports accuracy |
| `plot_single` | Plots the mean curve over repeated runs with a ±1 std shaded band |

## How evaluation works

1. Train the autoencoder for one epoch on the **training split** (labels ignored).
2. Freeze the encoder and encode the **validation split** into latent codes.
3. Train a fresh linear probe (`nn.Linear(latent_dim, num_classes)`) on those codes and their labels.
4. Report the probe's accuracy on the **test split**.

Steps 2–4 run once before training (baseline) and again after every epoch.

## Experiment 1: Linear autoencoders on synthetic data

**Setup**
- 100-dimensional Gaussian data, randomly rotated, with **20 high-variance directions** (variance 10) and 80 low-variance directions (variance 1).
- Binary label: whether the sum of the 20 high-variance coordinates is positive.
- Linear autoencoders (`nn.Identity` activation, one layer each way) with latent sizes **5, 20, 100, 500**; vanilla, denoising and masked variants; 3 repeats each.
- The probe is trained on the full labeled validation set.

**Findings**
- The vanilla autoencoder's validation reconstruction loss levels off exactly where PCA / Eckart–Young predicts (the variance of the discarded directions):

  | Latent dim | Predicted best loss | Observed plateau |
  |---|---|---|
  | 5 | 15·10 + 80·1 = 230 | ≈ 230 |
  | 20 | 80·1 = 80 | ≈ 80 |
  | 100 | 0 | ≈ 0 |
  | 500 | 0 | ≈ 0 (reached fastest) |

- A linear autoencoder trained by gradient descent recovers the PCA subspace on its own.
- Lower reconstruction loss does **not** mean more useful features: latent sizes 100 and 500 reach near-zero loss simply by copying the input.
- With plenty of labeled data for the probe, dimension reduction gives **no advantage** in probe accuracy, because the raw input is already linearly separable.

## Experiment 2: Nonlinear autoencoder on MNIST with few labels

**Setup**
- All MNIST training images used for unsupervised pretraining; only **200 labeled examples** used to train the probe.
- Denoising autoencoder, `hidden_dims = [100, 100]` (784 → 100 → ReLU → 100), ReLU, `noise_std = 0.5`, `lr = 5e-4`, batch size 100, 25 epochs, 2 repeats, 1000 probe epochs.

**Result**
- **Average final-epoch linear probe accuracy: 79.5%** (target ≥ 79%; random guessing ≈ 10%).
- Each run took about 78 seconds on a Colab T4 GPU.

**Takeaway:** when labels are scarce, self-supervised pretraining pays off. Learning to remove noise forces the encoder to capture the structure of digits, so a simple linear classifier can reach about 80% accuracy from just 200 labeled examples.

## Running the notebook

1. Open `q_autoencoder.ipynb` in Google Colab.
2. Select a GPU runtime: **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** MNIST downloads automatically on first run.

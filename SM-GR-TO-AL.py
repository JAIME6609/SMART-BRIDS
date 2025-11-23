# -*- coding: utf-8 -*-
"""
Combined SmartGrid + Autoencoders + TDA Pipeline
================================================

This script integrates, without removing functionality, the content of:

1) SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-03.py
   (SmartGrid-TDA-AE-Topological-Pipeline-FULL-COMPARISON_v3.py)

2) SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-04.py
   (SmartGrid-TDA-AE-Topological-Pipeline-FULL-COMPARISON.py)

Both pipelines are preserved as two separate experiment functions:

- run_smartgrid_version3()
- run_smartgrid_full_comparison()

so that all plots, metrics, exports, and topological analyses from both
original scripts remain available within a single unified file.

All comments, docstrings and printed messages are now in English.
"""

# ============================================================================
# GLOBAL IMPORTS AND SHARED UTILITIES
# ============================================================================

import os
import sys
import gc
import subprocess
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # for headless environments (e.g., servers, Colab)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

# scikit-learn (used in the full-comparison version)
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# Package installation helper (shared by both pipelines)
# ----------------------------------------------------------------------------
def _ensure_package(pkg_name: str) -> bool:
    """
    Try to import a package; if not found, try to install it via pip.

    Returns
    -------
    bool
        True if the package is available (import succeeded), False otherwise.
    """
    try:
        importlib.import_module(pkg_name)
        return True
    except Exception:
        try:
            print(f"[Installing] {pkg_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.import_module(pkg_name)
            return True
        except Exception as e:
            print(f"[Warning] Could not install {pkg_name}: {e}")
            return False


# ----------------------------------------------------------------------------
# TDA-related packages (ripser, persim)
# ----------------------------------------------------------------------------
HAS_RIPSER = _ensure_package("ripser")
HAS_PERSIM = _ensure_package("persim")

if HAS_RIPSER:
    from ripser import ripser

if HAS_PERSIM:
    import persim
    from persim import wasserstein, plot_diagrams

# Global plotting style
plt.style.use("seaborn-v0_8-darkgrid")


# ============================================================================
# EXPERIMENT 1: Version 3 (SMARTGRID_TDA_AE_ANALYSIS_v3)
# ============================================================================

def run_smartgrid_version3():
    """
    This function encapsulates the functionality of:

        SmartGrid-TDA-AE-Topological-Pipeline-FULL-COMPARISON_v3.py

    It:

    - Generates synthetic smart-grid consumption data for a small set of clients.
    - Injects realistic anomalies (fraud, failure, sudden peaks).
    - Produces point plots, mosaics and a 3D delay embedding visualization.
    - Builds a global delay embedding cloud and normalizes it.
    - Trains two dense autoencoders (1D and 2D latent spaces).
    - Computes reconstruction MSE and histograms.
    - Computes persistent homology diagrams, barcodes, and Betti curves.
    - Computes Wasserstein distances between original and reconstructed diagrams.
    - Exports all numerical results to an Excel file.
    - Prints an explicit comparison between numeric (MSE) and topological (Wasserstein) metrics.
    """

    # ------------------------------------------------------------------------
    # 0. Basic configuration
    # ------------------------------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Global parameters
    N_DAYS = 30
    POINTS_PER_DAY = 96
    N_CLIENTS = 8
    TOTAL_POINTS = N_DAYS * POINTS_PER_DAY

    EMBED_DIM = 3
    DELAY = 2

    LATENT_1D = 1
    LATENT_2D = 2

    EPOCHS = 80
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    MAXDIM = 2
    N_SUBSAMPLE = 1500

    OUTPUT_EXCEL = "SMARTGRID_TDA_AE_ANALYSIS_v3.xlsx"
    OUTPUT_FIG_DIR = "fig_smartgrid_v3"

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Synthetic smart-grid data generation
    # ------------------------------------------------------------------------
    def generate_daily_profile(num_points=POINTS_PER_DAY,
                               base_load=1.0,
                               peak_amplitude=0.5,
                               noise_std=0.05,
                               random_phase=True):
        """
        Generate a synthetic daily consumption profile with:
        - A base component.
        - A sinusoidal day/night component.
        - An evening peak.
        - Gaussian noise.
        """
        t = np.linspace(0, 24, num_points, endpoint=False)
        base = base_load * np.ones_like(t)

        phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
        sinus = 0.3 * np.sin(2 * np.pi * t / 24 + phase)

        peak_center = 19.0
        peak_width = 2.0
        peak = peak_amplitude * np.exp(-0.5 * ((t - peak_center) / peak_width) ** 2)

        noise = np.random.normal(0, noise_std, size=num_points)

        profile = base + sinus + peak + noise
        profile = np.clip(profile, 0, None)
        return t, profile

    def generate_monthly_profile(n_days=N_DAYS,
                                 points_per_day=POINTS_PER_DAY,
                                 base_load=1.0,
                                 peak_amplitude=0.5,
                                 noise_std=0.05):
        """
        Concatenate slightly perturbed daily profiles to get a monthly profile.
        """
        series = []
        for _ in range(n_days):
            base_load_d = base_load * np.random.uniform(0.9, 1.1)
            peak_amp_d = peak_amplitude * np.random.uniform(0.8, 1.2)
            _, prof = generate_daily_profile(points_per_day,
                                             base_load=base_load_d,
                                             peak_amplitude=peak_amp_d,
                                             noise_std=noise_std)
            series.append(prof)
        series = np.concatenate(series)
        return series

    def inject_anomalies(series,
                         anomaly_type="fraud",
                         intensity=0.5,
                         duration_points=50):
        """
        Inject local anomalies in a time series:
        - fraud: consumption reduction
        - failure: near-zero drop
        - peak: sudden surge
        """
        series = series.copy()
        n = len(series)
        start = np.random.randint(0, n - duration_points)
        end = start + duration_points

        if anomaly_type == "fraud":
            factor = np.random.uniform(0.1, 0.5)
            series[start:end] *= factor
        elif anomaly_type == "failure":
            series[start:end] *= np.random.uniform(0.0, 0.1)
        elif anomaly_type == "peak":
            bump = np.random.uniform(1.0, 2.0)
            series[start:end] += bump

        series = np.clip(series, 0, None)
        return series

    def simulate_clients(n_clients=N_CLIENTS,
                         n_days=N_DAYS,
                         points_per_day=POINTS_PER_DAY):
        """
        Simulate n_clients series; half are normal, half have injected anomalies.
        """
        all_series = []
        labels = []
        for i in range(n_clients):
            s = generate_monthly_profile(n_days=n_days,
                                         points_per_day=points_per_day,
                                         base_load=np.random.uniform(0.8, 1.3),
                                         peak_amplitude=np.random.uniform(0.4, 0.8),
                                         noise_std=0.06)
            if i >= n_clients // 2:
                anomaly_type = np.random.choice(["fraud", "failure", "peak"])
                s = inject_anomalies(s,
                                     anomaly_type=anomaly_type,
                                     intensity=0.6,
                                     duration_points=np.random.randint(30, 120))
                labels.append(f"anomalous_{anomaly_type}")
            else:
                labels.append("normal")
            all_series.append(s)

        all_series = np.array(all_series)
        return all_series, labels

    all_series, client_labels = simulate_clients()

    print("[INFO v3] Synthetic smart-grid data generated.")
    print("          all_series shape:", all_series.shape)
    print("          labels:", client_labels)

    # ------------------------------------------------------------------------
    # 2. Exploratory plots (point plots, comparison, mosaics)
    # ------------------------------------------------------------------------
    def plot_example_series_points(series, title, filename):
        """
        Plot a 1D time series as a scatter (point cloud).
        """
        plt.figure(figsize=(10, 4))
        plt.scatter(np.arange(len(series)), series, s=8, alpha=0.8)
        plt.xlabel("Time (sample index)")
        plt.ylabel("Consumption (a.u.)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    idx_normal = 0
    idx_anom = N_CLIENTS - 1
    series_normal = all_series[idx_normal]
    series_anom = all_series[idx_anom]

    plot_example_series_points(series_normal,
                               "Normal client - monthly consumption (points)",
                               "v3_client_normal_points.png")

    plot_example_series_points(series_anom,
                               f"Anomalous client ({client_labels[idx_anom]}) - monthly consumption (points)",
                               "v3_client_anomalous_points.png")

    def plot_compare_two_series(series1, series2, labels, filename):
        """
        Compare two series as scatter plots.
        """
        t = np.arange(len(series1))
        plt.figure(figsize=(10, 4))
        plt.scatter(t, series1, s=8, alpha=0.8, label=labels[0])
        plt.scatter(t, series2, s=8, alpha=0.8, label=labels[1])
        plt.xlabel("Time (sample index)")
        plt.ylabel("Consumption (a.u.)")
        plt.title("Comparison of two clients (points)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_compare_two_series(series_normal,
                            series_anom,
                            ["Normal", f"Anomalous ({client_labels[idx_anom]})"],
                            "v3_comparison_normal_vs_anomalous_points.png")

    def plot_mosaic_clients(all_series, labels,
                            n_rows=2, n_cols=4,
                            filename="v3_clients_mosaic.png"):
        """
        Mosaic of all clients, each one with its monthly series as points.
        """
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(16, 6),
                                 sharex=True, sharey=True)
        axes = axes.flatten()
        t = np.arange(all_series.shape[1])

        for i, ax in enumerate(axes):
            if i < all_series.shape[0]:
                ax.scatter(t, all_series[i], s=5, alpha=0.8)
                ax.set_title(f"Client {i} ({labels[i]})", fontsize=8)
            ax.tick_params(labelsize=6)

        fig.suptitle("Clients mosaic - monthly consumption (points)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_mosaic_clients(all_series, client_labels)

    # ------------------------------------------------------------------------
    # 3. Delay embedding and 3D view
    # ------------------------------------------------------------------------
    def delay_embedding(series, dim=EMBED_DIM, delay=DELAY):
        """
        Build the delay embedding:
        X_t = [s_t, s_{t+delay}, ..., s_{t+(dim-1)*delay}]
        """
        n = len(series)
        m = n - (dim - 1) * delay
        if m <= 0:
            raise ValueError("Not enough points for the embedding.")
        emb = np.zeros((m, dim))
        for i in range(dim):
            emb[:, i] = series[i * delay: i * delay + m]
        return emb

    embedding_normal = delay_embedding(series_normal, dim=EMBED_DIM, delay=DELAY)
    print("[INFO v3] Delay embedding for normal client:", embedding_normal.shape)

    def plot_embedding_3d(embedding, title, filename):
        """
        3D point cloud of the delay embedding.
        """
        if embedding.shape[1] < 3:
            raise ValueError("At least 3 dimensions are required for 3D plotting.")
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                   s=4, alpha=0.6)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_embedding_3d(embedding_normal,
                      "3D delay embedding - normal client",
                      "v3_embedding3d_normal_client.png")

    # ------------------------------------------------------------------------
    # 4. Global set of embeddings and normalization
    # ------------------------------------------------------------------------
    all_embeddings = []
    for i in range(N_CLIENTS):
        emb = delay_embedding(all_series[i], dim=EMBED_DIM, delay=DELAY)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)
    print("[INFO v3] Global point cloud:", all_embeddings.shape)

    mean_emb = np.mean(all_embeddings, axis=0)
    std_emb = np.std(all_embeddings, axis=0) + 1e-8
    all_embeddings_norm = (all_embeddings - mean_emb) / std_emb

    # ------------------------------------------------------------------------
    # 5. Autoencoders definition and training
    # ------------------------------------------------------------------------
    def build_autoencoder(input_dim, latent_dim):
        """
        Simple dense autoencoder with a configurable latent dimension.
        """
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(32, activation="relu")(inputs)
        x = layers.Dense(16, activation="relu")(x)
        latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)
        x = layers.Dense(16, activation="relu")(latent)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(input_dim, activation="linear")(x)
        model = keras.Model(inputs, outputs, name=f"AE_latent_{latent_dim}")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss="mse")
        return model

    ae1d = build_autoencoder(input_dim=EMBED_DIM, latent_dim=LATENT_1D)
    history_ae1d = ae1d.fit(all_embeddings_norm,
                            all_embeddings_norm,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=0,
                            validation_split=0.1)
    print("[INFO v3] AE1D trained.")

    ae2d = build_autoencoder(input_dim=EMBED_DIM, latent_dim=LATENT_2D)
    history_ae2d = ae2d.fit(all_embeddings_norm,
                            all_embeddings_norm,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=0,
                            validation_split=0.1)
    print("[INFO v3] AE2D trained.")

    def plot_training_history(history, title, filename):
        """
        Training and validation loss curves.
        """
        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_training_history(history_ae1d,
                          "Training curve - AE1D",
                          "v3_training_AE1D.png")

    plot_training_history(history_ae2d,
                          "Training curve - AE2D",
                          "v3_training_AE2D.png")

    # ------------------------------------------------------------------------
    # 6. Reconstruction and pointwise MSE
    # ------------------------------------------------------------------------
    recon_ae1d = ae1d.predict(all_embeddings_norm, verbose=0)
    recon_ae2d = ae2d.predict(all_embeddings_norm, verbose=0)

    def mse_pointwise(original, reconstructed):
        """
        Pointwise MSE between two matrices.
        """
        return np.mean((original - reconstructed) ** 2, axis=1)

    mse_ae1d = mse_pointwise(all_embeddings_norm, recon_ae1d)
    mse_ae2d = mse_pointwise(all_embeddings_norm, recon_ae2d)

    print("[INFO v3] MSE computed for AE1D and AE2D.")

    def plot_hist_mse(mse_values, title, filename):
        plt.figure(figsize=(6, 4))
        plt.hist(mse_values, bins=40, alpha=0.7)
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_hist_mse(mse_ae1d,
                  "MSE histogram - AE1D",
                  "v3_hist_mse_AE1D.png")

    plot_hist_mse(mse_ae2d,
                  "MSE histogram - AE2D",
                  "v3_hist_mse_AE2D.png")

    mse_stats = {
        "AE": ["AE1D", "AE2D"],
        "mse_mean": [np.mean(mse_ae1d), np.mean(mse_ae2d)],
        "mse_median": [np.median(mse_ae1d), np.median(mse_ae2d)],
        "mse_std": [np.std(mse_ae1d), np.std(mse_ae2d)]
    }
    df_mse_stats = pd.DataFrame(mse_stats)
    print("[INFO v3] Global MSE statistics:")
    print(df_mse_stats)

    # ------------------------------------------------------------------------
    # 7. Topological analysis (TDA) with ripser
    # ------------------------------------------------------------------------
    def subsample_points(X, n_subsample=N_SUBSAMPLE, seed=SEED):
        """
        Subsample points for TDA.
        """
        n_points = X.shape[0]
        if n_points <= n_subsample:
            return X
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_points, size=n_subsample, replace=False)
        return X[idx]

    if not HAS_RIPSER or not HAS_PERSIM:
        print("[INFO v3] TDA part skipped because ripser/persim are not available.")
        # However, we still export all non-TDA results.
        X_original_sub = all_embeddings_norm
        X_ae1d_sub = recon_ae1d
        X_ae2d_sub = recon_ae2d
        dgms_original = dgms_ae1d = dgms_ae2d = None
        df_wass = pd.DataFrame(columns=["case", "dimension", "wasserstein"])
    else:
        X_original_sub = subsample_points(all_embeddings_norm, N_SUBSAMPLE)
        X_ae1d_sub = subsample_points(recon_ae1d, N_SUBSAMPLE)
        X_ae2d_sub = subsample_points(recon_ae2d, N_SUBSAMPLE)

        print("[INFO v3] Subsampling for TDA:")
        print("          X_original_sub:", X_original_sub.shape)
        print("          X_ae1d_sub:", X_ae1d_sub.shape)
        print("          X_ae2d_sub:", X_ae2d_sub.shape)

        def compute_persistence_diagrams(X, maxdim=MAXDIM):
            """
            Compute persistence diagrams with ripser.
            """
            result = ripser(X, maxdim=maxdim)
            return result["dgms"]

        dgms_original = compute_persistence_diagrams(X_original_sub)
        dgms_ae1d = compute_persistence_diagrams(X_ae1d_sub)
        dgms_ae2d = compute_persistence_diagrams(X_ae2d_sub)

        print("[INFO v3] Persistence diagrams computed.")

        def plot_persistence_diagrams_three(dgms1, dgms2, dgms3, labels, filename):
            """
            Plot three persistence diagrams in a single figure.
            """
            plt.figure(figsize=(12, 4))
            for i, (dg, lab) in enumerate(zip([dgms1, dgms2, dgms3], labels), start=1):
                plt.subplot(1, 3, i)
                persim.plot_diagrams(dg, show=False)
                plt.title(lab)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_persistence_diagrams_three(
            dgms_original, dgms_ae1d, dgms_ae2d,
            labels=["Original", "AE1D", "AE2D"],
            filename="v3_persistence_diagrams_3cases.png"
        )

        def plot_barcodes_three(dgms1, dgms2, dgms3, labels, filename):
            """
            Plot three barcodes (lifetime=True) in a single figure.
            """
            plt.figure(figsize=(12, 4))
            for i, (dg, lab) in enumerate(zip([dgms1, dgms2, dgms3], labels), start=1):
                plt.subplot(1, 3, i)
                persim.plot_diagrams(dg, lifetime=True, show=False)
                plt.title(f"Barcodes - {lab}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_barcodes_three(
            dgms_original, dgms_ae1d, dgms_ae2d,
            labels=["Original", "AE1D", "AE2D"],
            filename="v3_barcodes_3cases.png"
        )

        def betti_curve(dgms, t_values):
            """
            Betti curves for each dimension.
            """
            betti_dict = {}
            for dim, dg in enumerate(dgms):
                curve = []
                for t in t_values:
                    count = np.sum((dg[:, 0] <= t) & (dg[:, 1] > t))
                    curve.append(count)
                betti_dict[dim] = np.array(curve)
            return betti_dict

        t_min = 0.0
        t_max = max([
            np.max(dgms_original[0][:, 1]),
            np.max(dgms_ae1d[0][:, 1]),
            np.max(dgms_ae2d[0][:, 1])
        ])
        t_values = np.linspace(t_min, t_max, 100)

        betti_original = betti_curve(dgms_original, t_values)
        betti_ae1d = betti_curve(dgms_ae1d, t_values)
        betti_ae2d = betti_curve(dgms_ae2d, t_values)

        def plot_betti_curves(betti_dicts, labels,
                              t_values, max_dim=2,
                              filename="v3_betti_curves_3cases.png"):
            """
            Betti curves for Original, AE1D and AE2D.
            """
            plt.figure(figsize=(12, 4))
            for dim in range(max_dim + 1):
                plt.subplot(1, max_dim + 1, dim + 1)
                for betti, lab in zip(betti_dicts, labels):
                    if dim in betti:
                        plt.plot(t_values, betti[dim], label=lab)
                plt.title(f"Betti curve H{dim}")
                plt.xlabel("t")
                plt.ylabel("Betti")
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_betti_curves(
            [betti_original, betti_ae1d, betti_ae2d],
            ["Original", "AE1D", "AE2D"],
            t_values,
            max_dim=2,
            filename="v3_betti_curves_3cases.png"
        )

        # --------------------------------------------------------------------
        # 8. Wasserstein distances
        # --------------------------------------------------------------------
        def wasserstein_distances_v3(dgms_ref, dgms_list, labels, q=2):
            """
            Wasserstein distances between a reference diagram and a list of diagrams.
            """
            rows = []
            for case_name, dgms in zip(labels, dgms_list):
                for dim in range(min(len(dgms_ref), len(dgms))):
                    d = persim.wasserstein(dgms_ref[dim], dgms[dim],
                                           matching=False, order=q)
                    rows.append({
                        "case": case_name,
                        "dimension": f"H{dim}",
                        "wasserstein": d
                    })
            return pd.DataFrame(rows)

        df_wass = wasserstein_distances_v3(
            dgms_original,
            [dgms_ae1d, dgms_ae2d],
            labels=["AE1D", "AE2D"],
            q=2
        )

        print("[INFO v3] Wasserstein distances:")
        print(df_wass)

    # ------------------------------------------------------------------------
    # 9. Export to Excel
    # ------------------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_loss_ae1d = pd.DataFrame({
            "epoch": np.arange(1, EPOCHS + 1),
            "loss": history_ae1d.history["loss"],
            "val_loss": history_ae1d.history.get("val_loss", [np.nan] * EPOCHS)
        })
        df_loss_ae1d.to_excel(writer, sheet_name="losses_ae1d", index=False)

        df_loss_ae2d = pd.DataFrame({
            "epoch": np.arange(1, EPOCHS + 1),
            "loss": history_ae2d.history["loss"],
            "val_loss": history_ae2d.history.get("val_loss", [np.nan] * EPOCHS)
        })
        df_loss_ae2d.to_excel(writer, sheet_name="losses_ae2d", index=False)

        df_mse_point = pd.DataFrame({
            "mse_ae1d": mse_ae1d,
            "mse_ae2d": mse_ae2d
        })
        df_mse_point.to_excel(writer, sheet_name="mse_pointwise", index=False)

        df_mse_stats.to_excel(writer, sheet_name="mse_stats", index=False)

        if HAS_RIPSER and HAS_PERSIM and dgms_original is not None:
            df_wass.to_excel(writer, sheet_name="wasserstein", index=False)
        else:
            # Empty sheet or omitted if TDA not available
            pd.DataFrame(columns=["case", "dimension", "wasserstein"]).to_excel(
                writer, sheet_name="wasserstein", index=False
            )

        df_meta = pd.DataFrame([{
            "N_DAYS": N_DAYS,
            "POINTS_PER_DAY": POINTS_PER_DAY,
            "N_CLIENTS": N_CLIENTS,
            "EMBED_DIM": EMBED_DIM,
            "DELAY": DELAY,
            "LATENT_1D": LATENT_1D,
            "LATENT_2D": LATENT_2D,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "MAXDIM_TDA": MAXDIM,
            "N_SUBSAMPLE_TDA": N_SUBSAMPLE,
            "SEED": SEED
        }])
        df_meta.to_excel(writer, sheet_name="metadata", index=False)

    print(f"[INFO v3] Results exported to Excel in: {OUTPUT_EXCEL}")

    # ------------------------------------------------------------------------
    # 10. Explicit comparison: no topology vs with topology (version 3)
    # ------------------------------------------------------------------------
    mse_mean_ae1d = df_mse_stats.loc[df_mse_stats["AE"] == "AE1D", "mse_mean"].values[0]
    mse_mean_ae2d = df_mse_stats.loc[df_mse_stats["AE"] == "AE2D", "mse_mean"].values[0]

    mse_median_ae1d = df_mse_stats.loc[df_mse_stats["AE"] == "AE1D", "mse_median"].values[0]
    mse_median_ae2d = df_mse_stats.loc[df_mse_stats["AE"] == "AE2D", "mse_median"].values[0]

    if HAS_RIPSER and HAS_PERSIM and not df_wass.empty:
        # H0 and H1 are guaranteed; H2 may or may not appear
        def _get_wass(df, case, dimension):
            vals = df[(df["case"] == case) & (df["dimension"] == dimension)]["wasserstein"].values
            return vals[0] if len(vals) > 0 else np.nan

        wass_ae1d_H0 = _get_wass(df_wass, "AE1D", "H0")
        wass_ae2d_H0 = _get_wass(df_wass, "AE2D", "H0")

        wass_ae1d_H1 = _get_wass(df_wass, "AE1D", "H1")
        wass_ae2d_H1 = _get_wass(df_wass, "AE2D", "H1")

        wass_ae1d_H2 = _get_wass(df_wass, "AE1D", "H2")
        wass_ae2d_H2 = _get_wass(df_wass, "AE2D", "H2")
    else:
        wass_ae1d_H0 = wass_ae1d_H1 = wass_ae1d_H2 = np.nan
        wass_ae2d_H0 = wass_ae2d_H1 = wass_ae2d_H2 = np.nan

    print("\n" + "=" * 70)
    print("COMPARISON WITHOUT TOPOLOGY vs WITH TOPOLOGY (VERSION 3)")
    print("=" * 70)
    print("1) MSE (without topology) - numerical reconstruction measure:")
    print(f"   AE1D -> mean MSE:   {mse_mean_ae1d:.6f}, median: {mse_median_ae1d:.6f}")
    print(f"   AE2D -> mean MSE:   {mse_mean_ae2d:.6f}, median: {mse_median_ae2d:.6f}")

    print("\n2) Wasserstein distances (with topology):")
    print("   (between persistence diagrams of the original point cloud and the reconstruction)")
    print(f"   AE1D -> W_2(H0): {wass_ae1d_H0:.6f}, W_2(H1): {wass_ae1d_H1:.6f}, W_2(H2): {wass_ae1d_H2:.6f}")
    print(f"   AE2D -> W_2(H0): {wass_ae2d_H0:.6f}, W_2(H1): {wass_ae2d_H1:.6f}, W_2(H2): {wass_ae2d_H2:.6f}")

    print("\nInterpretation:")
    print(" - A small MSE indicates good reconstruction in the Euclidean sense.")
    print(" - Small Wasserstein distances indicate preservation of topological structure")
    print("   (connected components, loops, cavities).")
    print(" - A desirable model combines low MSE with moderate Wasserstein distances,")
    print("   signaling both numerical and structural fidelity.")
    print("=" * 70 + "\n")


# ============================================================================
# EXPERIMENT 2: Full Comparison Version (experiment_outputs + more clients)
# ============================================================================

def run_smartgrid_full_comparison():
    """
    This function encapsulates the functionality of:

        SmartGrid-TDA-AE-Topological-Pipeline-FULL-COMPARISON.py

    It:

    - Generates a larger synthetic smart-grid dataset (many clients, 14 days).
    - Injects anomalies (fraud, failure, unexpected peaks).
    - Produces point plots, client mosaics, and 3D delay embedding views.
    - Builds delay embeddings for all clients and stacks them.
    - Trains 1D and 2D autoencoders with train/test split.
    - Computes MSE distributions, latent-space plots, and training curves.
    - Applies TDA (persistence diagrams, barcodes, Betti curves, Wasserstein).
    - Performs an explicit comparison between non-topological and topological metrics.
    - Exports all results to an Excel file in ./experiment_outputs.
    """

    # ------------------------------------------------------------------------
    # 1) General configuration
    # ------------------------------------------------------------------------
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    N_CLIENTES = 400            # can be increased if GPU is available
    HORIZONTE_HORAS = 24 * 14   # 14 days, hourly sampling
    TEST_SIZE = 0.25
    EPOCHS = 120
    BATCH_SIZE = 64

    # Embedding
    EMBED_DIM = 3
    EMBED_DELAY = 2

    # TDA
    TDA_SUBSAMPLE = 700
    ALLOW_H2 = True
    TDA_MAXDIM = 2 if ALLOW_H2 else 1

    # Directories
    OUT_DIR = Path("experiment_outputs")
    IMG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_PATH = OUT_DIR / "smartgrid_topo_results.xlsx"

    # Plot style
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.grid": True
    })

    # ------------------------------------------------------------------------
    # 2) Smart-grid synthetic data generation
    # ------------------------------------------------------------------------
    def generar_curva_base_diaria(num_puntos=24,
                                  pico_maniana=1.3,
                                  pico_noche=1.5,
                                  nivel_base=0.6):
        """
        Base daily load curve with morning and evening peaks and small sinusoidal
        oscillation.
        """
        h = np.arange(num_puntos)
        carga = np.ones_like(h, dtype=np.float32) * nivel_base

        # morning peak
        carga += pico_maniana * np.exp(-0.5 * ((h - 8) / 2.5) ** 2)
        # night peak
        carga += pico_noche * np.exp(-0.5 * ((h - 20) / 2.5) ** 2)
        # small oscillation
        carga += 0.15 * np.sin(2 * np.pi * h / 24.0)
        return carga.astype(np.float32)

    def generar_serie_semana(num_dias=14):
        """
        Build a multi-day series by concatenating daily profiles with:
        - lower load during weekends
        - small day-to-day variation
        """
        serie = []
        for d in range(num_dias):
            base = generar_curva_base_diaria()
            # weekends (Saturday=5, Sunday=6) slightly lower
            if d % 7 in (5, 6):
                base = base * 0.8
            # mild day variation
            base = base * (1.0 + 0.05 * np.random.randn())
            serie.append(base)
        serie = np.concatenate(serie, axis=0).astype(np.float32)
        return serie

    def inyectar_anomalias(serie,
                           prob_fraude=0.05,
                           prob_fallo=0.03,
                           prob_pico=0.04):
        """
        Inject typical anomalies in the electrical domain:
        - fraud: reduced consumption for a period
        - technical failure: sudden drop to almost constant low level
        - unexpected peak: isolated large spike
        """
        serie = serie.copy()
        L = len(serie)

        # fraud
        if np.random.rand() < prob_fraude:
            ini = np.random.randint(0, L - 6)
            fin = ini + np.random.randint(3, 8)
            factor = np.random.uniform(0.3, 0.6)
            serie[ini:fin] = serie[ini:fin] * factor

        # technical failure
        if np.random.rand() < prob_fallo:
            ini = np.random.randint(0, L - 10)
            fin = ini + np.random.randint(4, 12)
            level = np.mean(serie[max(0, ini - 3):ini + 1])
            serie[ini:fin] = level + 0.01 * np.random.randn(fin - ini)

        # unexpected peak
        if np.random.rand() < prob_pico:
            idx = np.random.randint(0, L)
            serie[idx] = serie[idx] * np.random.uniform(1.8, 2.5)

        return serie

    def generar_dataset_smartgrid(n_clientes, horizonte_horas):
        """
        Generate the full smart-grid dataset:
        - each client has a 14-day hourly series
        - some clients have anomalies injected
        - renewable-like slow noise and small Gaussian noise are added
        """
        data = []
        etiquetas_anomalia = []
        for _ in range(n_clientes):
            s = generar_serie_semana(num_dias=horizonte_horas // 24)
            tuvo_anomalia = False
            if np.random.rand() < 0.35:
                s = inyectar_anomalias(s)
                tuvo_anomalia = True
            # slow renewable-like noise
            slow_noise = 0.05 * np.sin(2 * np.pi * np.arange(len(s)) / (24 * 3))
            s = s + slow_noise.astype(np.float32)
            # small Gaussian noise
            s = s + 0.02 * np.random.randn(*s.shape).astype(np.float32)

            data.append(s.astype(np.float32))
            etiquetas_anomalia.append(1 if tuvo_anomalia else 0)
        data = np.vstack(data).astype(np.float32)
        etiquetas_anomalia = np.array(etiquetas_anomalia, dtype=np.int32)
        return data, etiquetas_anomalia

    # Generate dataset
    X_series, y_anom = generar_dataset_smartgrid(N_CLIENTES, HORIZONTE_HORAS)

    # ------------------------------------------------------------------------
    # 2.1) Exploratory plots (points, mosaic, comparison)
    # ------------------------------------------------------------------------
    t = np.arange(HORIZONTE_HORAS)

    # 1) One client series
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(t, X_series[0], s=10, label="Client 0 (simulated)")
    plt.plot(t, X_series[0], alpha=0.4)
    plt.title("Simulated consumption series (Client 0)")
    plt.xlabel("Hour")
    plt.ylabel("Relative consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_client0_points.png")
    plt.close(fig)

    # 2) Healthy vs anomalous comparison
    serie_sana = generar_serie_semana(num_dias=HORIZONTE_HORAS // 24)
    serie_anom = inyectar_anomalias(serie_sana)
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(t, serie_sana, s=10, label="Healthy series", alpha=0.7)
    plt.scatter(t, serie_anom, s=10, label="Series with anomaly", alpha=0.7)
    plt.title("Comparison: healthy vs anomalous series")
    plt.xlabel("Hour")
    plt.ylabel("Relative consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_healthy_vs_anomalous.png")
    plt.close(fig)

    # 3) Mosaic of several clients
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    clientes_demo = [0, 1, 2, 3]
    for ax, c in zip(axes.flat, clientes_demo):
        tt = np.arange(X_series.shape[1])
        ax.scatter(tt, X_series[c], s=6)
        ax.plot(tt, X_series[c], alpha=0.4)
        ax.set_title(f"Client {c} (anom={y_anom[c]})")
    axes[1, 0].set_xlabel("Hour")
    axes[1, 1].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Consumption")
    axes[1, 0].set_ylabel("Consumption")
    plt.suptitle("Mosaic of simulated grid data")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(IMG_DIR / "sg_clients_mosaic.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 3) Delay embedding
    # ------------------------------------------------------------------------
    def delay_embedding_1serie(serie, m=3, delay=2):
        """
        Delay embedding for a single series.
        """
        L = len(serie)
        eff_len = L - (m - 1) * delay
        emb = np.zeros((eff_len, m), dtype=np.float32)
        for i in range(eff_len):
            for j in range(m):
                emb[i, j] = serie[i + j * delay]
        return emb

    def build_embedded_dataset(X, m=3, delay=2):
        """
        Build delay embeddings for all clients and stack them.
        """
        X_emb_list = []
        idx_slices = []
        all_points = []
        start = 0
        for i in range(X.shape[0]):
            emb_i = delay_embedding_1serie(X[i], m=m, delay=delay)
            X_emb_list.append(emb_i)
            end = start + emb_i.shape[0]
            idx_slices.append((start, end))
            all_points.append(emb_i)
            start = end
        X_emb_stack = np.vstack(all_points).astype(np.float32)
        return X_emb_list, X_emb_stack, idx_slices

    X_emb_list, X_emb_stack, idx_slices = build_embedded_dataset(
        X_series,
        m=EMBED_DIM,
        delay=EMBED_DELAY
    )

    # 3D embedding of one client
    emb_demo = X_emb_list[0]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb_demo[:, 0], emb_demo[:, 1], emb_demo[:, 2], s=6)
    ax.set_title("Delay embedding (Client 0)")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t-τ)")
    ax.set_zlabel("x(t-2τ)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_embedding_3d_client0.png")
    plt.close(fig)

    input_dim = EMBED_DIM

    # split train/test for autoencoders
    X_train, X_test = train_test_split(
        X_emb_stack,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # ------------------------------------------------------------------------
    # 4) Autoencoders (1D and 2D)
    # ------------------------------------------------------------------------
    # AE 1D
    latent_dim_1d = 1
    inp_1d = layers.Input(shape=(input_dim,), dtype="float32")
    h_1d = layers.Dense(64, activation="relu")(inp_1d)
    h_1d = layers.Dense(64, activation="relu")(h_1d)
    z_1d = layers.Dense(latent_dim_1d, name="latent_1d")(h_1d)

    zin_1d = layers.Input(shape=(latent_dim_1d,), dtype="float32")
    u_1d = layers.Dense(64, activation="relu")(zin_1d)
    u_1d = layers.Dense(64, activation="relu")(u_1d)
    out_1d = layers.Dense(input_dim)(u_1d)

    encoder_1d = Model(inp_1d, z_1d, name="encoder_smartgrid_1d")
    decoder_1d = Model(zin_1d, out_1d, name="decoder_smartgrid_1d")
    xhat_1d = decoder_1d(encoder_1d(inp_1d))
    ae_1d = Model(inp_1d, xhat_1d, name="AE_smartgrid_1d")
    ae_1d.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    hist_1d = ae_1d.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print(f"[AE 1D] final train loss: {hist_1d.history['loss'][-1]:.6f} | "
          f"final val loss: {hist_1d.history['val_loss'][-1]:.6f}")

    Z_1d = encoder_1d.predict(X_emb_stack, verbose=1).astype(np.float32)
    Xrec_1d = ae_1d.predict(X_emb_stack, verbose=1).astype(np.float32)

    # free a bit of memory
    K.clear_session()
    gc.collect()

    # AE 2D
    latent_dim_2d = 2
    inp_2d = layers.Input(shape=(input_dim,), dtype="float32")
    h_2d = layers.Dense(64, activation="relu")(inp_2d)
    h_2d = layers.Dense(64, activation="relu")(h_2d)
    z_2d = layers.Dense(latent_dim_2d, name="latent_2d")(h_2d)

    zin_2d = layers.Input(shape=(latent_dim_2d,), dtype="float32")
    u_2d = layers.Dense(64, activation="relu")(zin_2d)
    u_2d = layers.Dense(64, activation="relu")(u_2d)
    out_2d = layers.Dense(input_dim)(u_2d)

    encoder_2d = Model(inp_2d, z_2d, name="encoder_smartgrid_2d")
    decoder_2d = Model(zin_2d, out_2d, name="decoder_smartgrid_2d")
    xhat_2d = decoder_2d(encoder_2d(inp_2d))
    ae_2d = Model(inp_2d, xhat_2d, name="AE_smartgrid_2d")
    ae_2d.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    hist_2d = ae_2d.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print(f"[AE 2D] final train loss: {hist_2d.history['loss'][-1]:.6f} | "
          f"final val loss: {hist_2d.history['val_loss'][-1]:.6f}")

    Z_2d = encoder_2d.predict(X_emb_stack, verbose=0).astype(np.float32)
    Xrec_2d = ae_2d.predict(X_emb_stack, verbose=0).astype(np.float32)

    gc.collect()

    # ------------------------------------------------------------------------
    # 5) Classical metrics and training plots
    # ------------------------------------------------------------------------
    mse_pt_1d = np.mean((X_emb_stack - Xrec_1d) ** 2, axis=1).astype(np.float32)
    mse_pt_2d = np.mean((X_emb_stack - Xrec_2d) ** 2, axis=1).astype(np.float32)

    # Latent spaces
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    Z1d_flat = np.ravel(Z_1d)
    plt.scatter(Z1d_flat, np.zeros_like(Z1d_flat), s=4)
    plt.title("Latent AE 1D")
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=4)
    plt.title("Latent AE 2D")
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plt.scatter(Z_2d[:, 0], Z_2d[:, 1], s=4, label="AE 2D")
    plt.scatter(Z1d_flat, np.zeros_like(Z1d_flat), s=4, alpha=0.4, label="AE 1D (y=0)")
    plt.title("Latent comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_latent_spaces.png")
    plt.close(fig)

    # MSE histogram
    fig = plt.figure(figsize=(8, 4))
    plt.hist(mse_pt_1d, bins=40, alpha=0.7, label="AE 1D")
    plt.hist(mse_pt_2d, bins=40, alpha=0.7, label="AE 2D")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.title("Reconstruction error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_mse_hist.png")
    plt.close(fig)

    # Training curves
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_1d.history["loss"], label="train 1D")
    plt.plot(hist_1d.history["val_loss"], label="val 1D")
    plt.title("Training AE 1D")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_2d.history["loss"], label="train 2D")
    plt.plot(hist_2d.history["val_loss"], label="val 2D")
    plt.title("Training AE 2D")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.tight_layout()
    plt.savefig(IMG_DIR / "sg_training_curves.png")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 6) TDA: diagrams, barcodes, Betti, Wasserstein
    # ------------------------------------------------------------------------
    def _clean_dgm(D):
        if D is None or (isinstance(D, np.ndarray) and D.size == 0):
            return np.zeros((0, 2), dtype=np.float32)
        D = np.asarray(D, dtype=np.float32)
        mask = np.isfinite(D).all(axis=1)
        D = D[mask]
        if D.ndim != 2 or D.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.float32)
        return D

    if HAS_RIPSER and HAS_PERSIM:
        def compute_persistence_subsample(X_data, maxdim=1, n_perm=500):
            if X_data.shape[0] > n_perm:
                rng = np.random.default_rng(7)
                sel = rng.choice(X_data.shape[0], size=n_perm, replace=False)
                X_use = X_data[sel].astype(np.float32, copy=False)
            else:
                X_use = X_data.astype(np.float32, copy=False)
            res = ripser(X_use, maxdim=maxdim, n_perm=X_use.shape[0])
            dgms = res["dgms"]
            dgms_clean = [_clean_dgm(d) for d in dgms]
            return dgms_clean

        def plot_persistence_diagrams_full(dgms, title, savepath):
            fig = plt.figure(figsize=(6, 5))
            plot_diagrams(dgms, show=False)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close(fig)

        def plot_barcodes_full(dgms, title, savepath, max_bars=60):
            dims = len(dgms)
            fig, axes = plt.subplots(1, dims, figsize=(4 * dims, 4), sharey=True)
            if dims == 1:
                axes = [axes]
            for dim, ax in enumerate(axes):
                D = dgms[dim]
                ax.set_title(f"Barcodes H{dim}")
                ax.set_xlabel("ε")
                ax.set_yticks([])
                if D.size == 0:
                    continue
                deads = D[:, 1][np.isfinite(D[:, 1])]
                cutoff = np.percentile(deads, 99) if deads.size > 0 else (
                    D[:, 0].max() if D[:, 0].size > 0 else 1.0
                )
                Kbars = min(len(D), max_bars)
                sel = np.argsort(-(D[:, 1] - D[:, 0]))[:Kbars]
                Dsel = D[sel]
                for i in range(Kbars):
                    b, d = Dsel[i]
                    d_plot = min(d, cutoff)
                    ax.hlines(y=Kbars - 1 - i, xmin=b, xmax=d_plot,
                              color="black", linewidth=2)
            fig.suptitle(title, y=0.95)
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close(fig)

        def betti_curve_from_dgm(dgm, eps_grid):
            if dgm is None or dgm.size == 0:
                return np.zeros_like(eps_grid, dtype=np.int32)
            births = dgm[:, 0].astype(np.float32)
            deaths = dgm[:, 1].astype(np.float32)
            big = eps_grid.max() + (eps_grid[1] - eps_grid[0] if eps_grid.size > 1 else 1.0)
            deaths = np.where(np.isfinite(deaths), deaths, big).astype(np.float32)
            betti = np.zeros_like(eps_grid, dtype=np.int32)
            for i, eps in enumerate(eps_grid):
                betti[i] = int(np.sum((births <= eps) & (eps < deaths)))
            return betti

        def plot_betti_curves_full(dgms_list, labels, dims, eps_grid, title, savepath):
            fig = plt.figure(figsize=(6, 4.5))
            for dgms, lab in zip(dgms_list, labels):
                for dim in dims:
                    if dim < len(dgms):
                        bc = betti_curve_from_dgm(dgms[dim], eps_grid)
                        plt.plot(eps_grid, bc, label=f"{lab} (H{dim})")
            plt.title(title)
            plt.xlabel("ε")
            plt.ylabel("β")
            plt.legend()
            plt.tight_layout()
            plt.savefig(savepath)
            plt.close(fig)

        dgms_orig = compute_persistence_subsample(X_emb_stack,
                                                  maxdim=TDA_MAXDIM,
                                                  n_perm=TDA_SUBSAMPLE)
        dgms_ae1d = compute_persistence_subsample(Xrec_1d,
                                                  maxdim=TDA_MAXDIM,
                                                  n_perm=TDA_SUBSAMPLE)
        dgms_ae2d = compute_persistence_subsample(Xrec_2d,
                                                  maxdim=TDA_MAXDIM,
                                                  n_perm=TDA_SUBSAMPLE)

        dims_for_plots = [0, 1] if TDA_MAXDIM == 1 else [0, 1, 2]

        plot_persistence_diagrams_full(dgms_orig,
                                       "Persistence diagrams - Original",
                                       IMG_DIR / "sg_pd_original.png")
        plot_persistence_diagrams_full(dgms_ae1d,
                                       "Persistence diagrams - AE 1D",
                                       IMG_DIR / "sg_pd_ae1d.png")
        plot_persistence_diagrams_full(dgms_ae2d,
                                       "Persistence diagrams - AE 2D",
                                       IMG_DIR / "sg_pd_ae2d.png")

        plot_barcodes_full(dgms_orig,
                           "Barcodes - Original",
                           IMG_DIR / "sg_barcodes_original.png")
        plot_barcodes_full(dgms_ae1d,
                           "Barcodes - AE 1D",
                           IMG_DIR / "sg_barcodes_ae1d.png")
        plot_barcodes_full(dgms_ae2d,
                           "Barcodes - AE 2D",
                           IMG_DIR / "sg_barcodes_ae2d.png")

        eps_grid = np.linspace(0.0, 1.0, 200).astype(np.float32)
        plot_betti_curves_full(
            [dgms_orig, dgms_ae1d, dgms_ae2d],
            ["Original", "AE 1D", "AE 2D"],
            dims_for_plots,
            eps_grid,
            "Betti curves (Smart Grid)",
            IMG_DIR / "sg_betti_curves.png"
        )

        def wasserstein_per_dim(dgms_ref, dgms_hat, dims=(0, 1)):
            dists = {}
            for d in dims:
                if (dgms_ref is None) or (dgms_hat is None) or d >= len(dgms_ref) or d >= len(dgms_hat):
                    dists[f"H{d}"] = np.nan
                    continue
                A = _clean_dgm(dgms_ref[d])
                B = _clean_dgm(dgms_hat[d])
                dists[f"H{d}"] = (float(wasserstein(A, B)) if (A.size > 0 and B.size > 0) else np.nan)
            return dists

        wasserstein_ae1d = wasserstein_per_dim(dgms_orig, dgms_ae1d,
                                               dims=tuple(dims_for_plots))
        wasserstein_ae2d = wasserstein_per_dim(dgms_orig, dgms_ae2d,
                                               dims=tuple(dims_for_plots))

        print("[Wasserstein] Original vs AE 1D:", wasserstein_ae1d)
        print("[Wasserstein] Original vs AE 2D:", wasserstein_ae2d)
    else:
        print("TDA skipped: ripser/persim not available.")
        wasserstein_ae1d, wasserstein_ae2d = {}, {}
        dims_for_plots = [0, 1] if TDA_MAXDIM == 1 else [0, 1, 2]

    gc.collect()

    # ------------------------------------------------------------------------
    # 6.5) Explicit comparison: WITHOUT topological vs WITH topological metrics
    # ------------------------------------------------------------------------
    mse_mean_1d = float(mse_pt_1d.mean())
    mse_med_1d = float(np.median(mse_pt_1d))
    mse_mean_2d = float(mse_pt_2d.mean())
    mse_med_2d = float(np.median(mse_pt_2d))

    if HAS_RIPSER and HAS_PERSIM and (len(wasserstein_ae1d) > 0 or len(wasserstein_ae2d) > 0):
        print("\n=== EXPLICIT COMPARISON: WITHOUT TOPOLOGY vs WITH TOPOLOGY ===")
        print(">> WITHOUT TOPOLOGY (only MSE):")
        print(f"   AE 1D -> mse_mean={mse_mean_1d:.6f}, mse_median={mse_med_1d:.6f}")
        print(f"   AE 2D -> mse_mean={mse_mean_2d:.6f}, mse_median={mse_med_2d:.6f}")
        print(">> WITH TOPOLOGY (Wasserstein vs the original):")
        for dim_key, dist_val in wasserstein_ae1d.items():
            print(f"   dim {dim_key}: Original vs AE 1D = {dist_val}")
        for dim_key, dist_val in wasserstein_ae2d.items():
            print(f"   dim {dim_key}: Original vs AE 2D = {dist_val}")
        print("\nAutomatic interpretation:")
        print("- If AE 1D and AE 2D have similar MSE but the Wasserstein distance of AE 2D")
        print("  is smaller, then AE 2D preserves the cyclic structure of the grid better.")
        print("- If an AE has low MSE BUT a large Wasserstein distance, that AE has 'flattened'")
        print("  the shape; for electrical networks this is considered a loss of relevant structure.")
    else:
        print("\n[INFO] Topological comparison not fully available because ripser/persim")
        print("or Wasserstein data are missing. MSE metrics are still computed.")

    # ------------------------------------------------------------------------
    # 7) Export to Excel
    # ------------------------------------------------------------------------
    def _pick_excel_engine():
        try:
            importlib.import_module("openpyxl")
            return "openpyxl"
        except Exception:
            pass
        try:
            importlib.import_module("xlsxwriter")
            return "xlsxwriter"
        except Exception:
            pass

        if _ensure_package("openpyxl"):
            return "openpyxl"
        if _ensure_package("xlsxwriter"):
            return "xlsxwriter"

        raise RuntimeError("Could not find or install openpyxl or xlsxwriter.")

    EXCEL_ENGINE = _pick_excel_engine()
    print(f"[Excel] Engine used: {EXCEL_ENGINE}")

    def export_to_excel(
        excel_path,
        engine,
        hist_1d, hist_2d,
        mse_pt_1d, mse_pt_2d,
        wasserstein_ae1d, wasserstein_ae2d,
        dims_for_plots,
        mse_mean_1d, mse_med_1d, mse_mean_2d, mse_med_2d
    ):
        with pd.ExcelWriter(excel_path, engine=engine) as writer:
            # AE 1D losses
            df_loss_1d = pd.DataFrame({
                "epoch": np.arange(1, len(hist_1d.history["loss"]) + 1, dtype=np.int32),
                "loss_train": np.asarray(hist_1d.history["loss"], dtype=np.float32),
                "loss_val": np.asarray(hist_1d.history["val_loss"], dtype=np.float32)
            })
            df_loss_1d.to_excel(writer, sheet_name="losses_ae1d", index=False)

            # AE 2D losses
            df_loss_2d = pd.DataFrame({
                "epoch": np.arange(1, len(hist_2d.history["loss"]) + 1, dtype=np.int32),
                "loss_train": np.asarray(hist_2d.history["loss"], dtype=np.float32),
                "loss_val": np.asarray(hist_2d.history["val_loss"], dtype=np.float32)
            })
            df_loss_2d.to_excel(writer, sheet_name="losses_ae2d", index=False)

            # Classical MSE statistics
            df_stats = pd.DataFrame({
                "model": ["AE_1D", "AE_2D"],
                "mse_mean": [mse_mean_1d, mse_mean_2d],
                "mse_median": [mse_med_1d, mse_med_2d],
                "mse_std": [float(mse_pt_1d.std(ddof=1)), float(mse_pt_2d.std(ddof=1))]
            })
            df_stats.to_excel(writer, sheet_name="mse_stats", index=False)

            # Pointwise MSE
            pd.DataFrame({"mse_point": mse_pt_1d.astype(np.float32)}).to_excel(
                writer, sheet_name="mse_point_ae1d", index=False
            )
            pd.DataFrame({"mse_point": mse_pt_2d.astype(np.float32)}).to_excel(
                writer, sheet_name="mse_point_ae2d", index=False
            )

            # Topological distances (if any)
            if len(wasserstein_ae1d) > 0 or len(wasserstein_ae2d) > 0:
                dims_txt = [f"H{d}" for d in dims_for_plots]
                df_wass = pd.DataFrame({
                    "dimension": dims_txt,
                    "wasserstein_vs_AE1D": [wasserstein_ae1d.get(k, np.nan) for k in dims_txt],
                    "wasserstein_vs_AE2D": [wasserstein_ae2d.get(k, np.nan) for k in dims_txt],
                })
                df_wass.to_excel(writer, sheet_name="wasserstein", index=False)

            # Metadata and explicit comparison
            df_meta = pd.DataFrame({
                "parameter": [
                    "N_CLIENTES", "HORIZONTE_HORAS", "EMBED_DIM", "EMBED_DELAY",
                    "TEST_SIZE", "EPOCHS", "BATCH_SIZE",
                    "TDA_SUBSAMPLE", "TDA_MAXDIM", "ALLOW_H2", "EXCEL_ENGINE",
                    "mse_mean_ae1d", "mse_median_ae1d",
                    "mse_mean_ae2d", "mse_median_ae2d"
                ],
                "value": [
                    N_CLIENTES, HORIZONTE_HORAS, EMBED_DIM, EMBED_DELAY,
                    TEST_SIZE, EPOCHS, BATCH_SIZE,
                    TDA_SUBSAMPLE, TDA_MAXDIM, ALLOW_H2, engine,
                    mse_mean_1d, mse_med_1d,
                    mse_mean_2d, mse_med_2d
                ]
            })
            df_meta.to_excel(writer, sheet_name="metadata", index=False)

    export_to_excel(
        EXCEL_PATH,
        EXCEL_ENGINE,
        hist_1d, hist_2d,
        mse_pt_1d, mse_pt_2d,
        wasserstein_ae1d, wasserstein_ae2d,
        dims_for_plots,
        mse_mean_1d, mse_med_1d, mse_mean_2d, mse_med_2d
    )

    print("\n=== PROCESS COMPLETED (SmartGrid + AE + TDA + Explicit Comparison) ===")
    print(f"- Figures saved in: {IMG_DIR.resolve()}")
    print(f"- Excel file saved in: {EXCEL_PATH.resolve()}")


# ============================================================================
# MAIN: run one or both experiments
# ============================================================================

if __name__ == "__main__":
    # Run Version 3 experiment (small-scale, Excel in current directory)
    # run_smartgrid_version3()

    # Run full comparison experiment (larger-scale, results in experiment_outputs/)
    run_smartgrid_full_comparison()



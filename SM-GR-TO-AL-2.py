# -*- coding: utf-8 -*-
"""
SMART-GRIDS + AUTOENCODERS + TOPOLOGICAL DATA ANALYSIS (TDA)
FUSED VERSION OF 03.py AND 03D.py, RAM-FRIENDLY

===========================================================================
GOAL OF THIS FILE
===========================================================================

This script is a **unified and extended version** that merges, in a single
Python file, the functionality and outputs originally distributed across:

    1) SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-03.py
    2) SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-03D.py

The present code is designed so that:

    * It preserves (and in some points extends) the visualizations, tables,
      and Excel reports produced by both original files.

    * It includes:
        - Synthetic smart-grid data generation (normal and anomalous clients).
        - Point plots and mosaics of clients.
        - Delay embeddings and 3D visualizations.
        - Two autoencoders (1D latent and 2D latent).
        - Training curves and MSE histograms.
        - Topological Data Analysis (TDA) using ripser / persim:
          persistence diagrams, barcodes, Betti curves, and Wasserstein
          distances between original and reconstructed data.
        - A “version 3” experiment (small/medium scale) that imitates the
          03.py behavior, including the Excel file
          SMARTGRID_TDA_AE_ANALYSIS_v3.xlsx and a folder fig_smartgrid_v3.
        - A “full comparison” experiment, conceptually aligned with the
          extended behavior of 03D.py, writing results under the folder
          experiment_outputs/.

    * It is explicitly engineered to be **RAM friendly**, by:
        - Using moderate numbers of clients in the full experiment.
        - Applying random subsampling for TDA (e.g., <= 1000 points).
        - Avoiding unnecessary copies of large arrays.
        - Calling gc.collect() after heavy blocks.
        - Using the non-interactive Matplotlib backend "Agg".

    * All comments and docstrings are written in **English**, as requested.

===========================================================================
IMPORTANT ORGANIZATION
===========================================================================

The script is structured in the following sections:

    0. Imports, optional package installer, and global helpers.
    1. Experiment V3 (small/medium scale, reproduces 03.py style outputs).
    2. Experiment FULL (extended comparison, in the spirit of 03D.py,
       but with conservative RAM usage).
    3. Main entry point, where the user can choose which experiments
       are executed by default.

The user can run this file as:

    python SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-FUSED.py

and then enable/disable experiments in the __main__ block at the end.
"""

# ============================================================================
# 0. IMPORTS, OPTIONAL INSTALLATION, AND GLOBAL HELPERS
# ============================================================================

import os
import sys
import gc
import subprocess
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-interactive backend to avoid GUI issues and reduce overhead.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# ---------------------------------------------------------------------------
# Optional installer: try to ensure that some packages are available
# ---------------------------------------------------------------------------

def ensure_package(pkg_name: str) -> bool:
    """
    Try to import a package; if not found, attempt to install it via pip.

    Parameters
    ----------
    pkg_name : str
        Name of the package to import/install.

    Returns
    -------
    bool
        True if the package is successfully imported (before or after
        installation), False otherwise.

    Notes
    -----
    This helper is useful to keep the script self-contained in environments
    such as Google Colab or a fresh local installation, where ripser or
    persim might not be pre-installed.
    """
    try:
        importlib.import_module(pkg_name)
        return True
    except Exception:
        try:
            print(f"[INFO] Installing package: {pkg_name} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg_name]
            )
            importlib.import_module(pkg_name)
            return True
        except Exception as e:
            print(f"[WARNING] Could not install '{pkg_name}': {e}")
            return False


# Try to ensure TDA packages are present
HAS_RIPSER = ensure_package("ripser")
HAS_PERSIM = ensure_package("persim")

if HAS_RIPSER:
    from ripser import ripser

if HAS_PERSIM:
    import persim
    from persim import plot_diagrams  # used for persistence diagrams

# Ensure TensorFlow / Keras for autoencoders
ensure_package("tensorflow")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Consistent visual style for all plots
plt.style.use("seaborn-v0_8-darkgrid")


# ============================================================================
# 0.1 GENERIC UTILITY FUNCTIONS
# ============================================================================

def safe_mkdir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Path to the directory to be created.

    Notes
    -----
    The function is idempotent: calling it multiple times for the same
    directory is safe and does not raise an error.
    """
    os.makedirs(path, exist_ok=True)


def describe_array(name: str, arr: np.ndarray) -> None:
    """
    Print a brief description (shape and dtype) of a NumPy array.

    Parameters
    ----------
    name : str
        Label for the array to be printed.
    arr : np.ndarray
        Array to be described.

    This utility is purely diagnostic and helps to quickly understand the
    scale of the data being handled by the pipeline.
    """
    print(f"[DEBUG] {name}: shape={arr.shape}, dtype={arr.dtype}")


def choose_excel_engine() -> str:
    """
    Try to select a reasonable Excel writer engine (e.g., openpyxl).

    Returns
    -------
    str
        Name of the engine, or None if no specific engine is found.

    By default, openpyxl is the most common engine for .xlsx files.
    """
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        return None


def delay_embedding_1d_series(series: np.ndarray,
                              dimension: int,
                              delay: int) -> np.ndarray:
    """
    Build the delay embedding of a 1D time series.

    Parameters
    ----------
    series : np.ndarray
        1D array of shape (T,), representing a time series.
    dimension : int
        Embedding dimension (number of delayed coordinates).
    delay : int
        Delay between coordinates.

    Returns
    -------
    np.ndarray
        2D array of shape (T_eff, dimension), where T_eff is the number
        of points that can be embedded given the chosen dimension and delay.

    Notes
    -----
    The constructed vector at time t is:

        X_t = [x_t, x_{t+delay}, ..., x_{t+(dimension-1)*delay}]

    This function is used in both experiments (v3 and full comparison).
    """
    series = np.asarray(series, dtype=np.float32)
    L = len(series)
    T_eff = L - (dimension - 1) * delay
    if T_eff <= 0:
        raise ValueError(
            "Not enough points for delay embedding with "
            f"dimension={dimension}, delay={delay}, length={L}"
        )
    emb = np.zeros((T_eff, dimension), dtype=np.float32)
    for j in range(dimension):
        emb[:, j] = series[j * delay : j * delay + T_eff]
    return emb


def subsample_points(X: np.ndarray,
                     n_samples: int,
                     seed: int = 42) -> np.ndarray:
    """
    Subsample rows of a point cloud to limit memory and runtime for TDA.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (N, d), a point cloud in R^d.
    n_samples : int
        Maximum number of points to keep.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Subsampled point cloud.

    Notes
    -----
    If the array has fewer than n_samples rows, the function returns
    the original array unchanged.
    """
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    if N <= n_samples:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=n_samples, replace=False)
    return X[idx]


def compute_persistence_diagrams(X: np.ndarray,
                                 maxdim: int = 2) -> list:
    """
    Compute persistence diagrams using ripser for a point cloud.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (N, d) representing a point cloud.
    maxdim : int
        Maximum homology dimension to compute (0, 1, 2, ...).

    Returns
    -------
    list
        List of persistence diagrams; dgms[k] contains the diagram
        for H^k.

    Requirements
    ------------
    This function requires ripser to be installed. If ripser is not
    available, the user should ensure HAS_RIPSER is True before calling.
    """
    if not HAS_RIPSER:
        raise RuntimeError(
            "ripser is not available. Install it or disable TDA."
        )
    X = np.asarray(X, dtype=np.float32)
    result = ripser(X, maxdim=maxdim)
    return result["dgms"]


def betti_curve_from_diagram(diagram: np.ndarray,
                             t_values: np.ndarray) -> np.ndarray:
    """
    Build a Betti curve from a single persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram of shape (n_points, 2), where each row
        is [birth, death].
    t_values : np.ndarray
        1D array of evaluation points for the Betti curve.

    Returns
    -------
    np.ndarray
        1D array of length len(t_values) with Betti number at each t.

    Notes
    -----
    Points with infinite death time are truncated slightly beyond
    the maximum of t_values to avoid numerical issues.
    """
    if diagram is None or len(diagram) == 0:
        return np.zeros_like(t_values, dtype=int)

    dgm = np.asarray(diagram, dtype=np.float32)
    births = dgm[:, 0]
    deaths = dgm[:, 1]

    t_max = float(np.max(t_values))
    deaths = np.where(
        np.isfinite(deaths),
        deaths,
        t_max + 0.05 * t_max + 1e-6
    )

    betti = np.zeros_like(t_values, dtype=int)
    for i, t in enumerate(t_values):
        betti[i] = np.sum((births <= t) & (t < deaths))

    return betti


def wasserstein_distance_simple(diag_ref: np.ndarray,
                                diag_other: np.ndarray) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams
    using persim.wasserstein with default parameters.

    Parameters
    ----------
    diag_ref : np.ndarray
        Reference persistence diagram.
    diag_other : np.ndarray
        Comparison persistence diagram.

    Returns
    -------
    float
        Wasserstein distance value.

    Notes
    -----
    The function calls persim.wasserstein(diag_ref, diag_other)
    without extra keyword arguments (to avoid signature mismatches
    with different persim versions).
    """
    if not HAS_PERSIM:
        return np.nan
    return float(persim.wasserstein(diag_ref, diag_other))


# ============================================================================
# 1. EXPERIMENT V3 (SMALL/MEDIUM SCALE, LIKE 03.py)
# ============================================================================

def run_smartgrid_version3():
    """
    Execute a small/medium scale smart-grid experiment that mirrors
    the logic of SMART-GRIDS-TOPOLOGIA-ALGEBRAICA-03.py:

        - Generate synthetic monthly consumption for several clients.
        - Inject anomalies into half of them.
        - Produce exploratory point plots and mosaics.
        - Build 3D delay embeddings.
        - Train 1D and 2D autoencoders on the global point cloud.
        - Compute reconstruction MSE and histograms.
        - Apply TDA to original vs reconstructed embeddings.
        - Compute Betti curves and Wasserstein distances.
        - Export all relevant metrics to an Excel file:
              SMARTGRID_TDA_AE_ANALYSIS_v3.xlsx
        - Save figures under:
              fig_smartgrid_v3/

    The configuration in this function is conservative with respect to RAM.
    """

    # -----------------------------------------------------------------------
    # 1.1 Configuration for version 3
    # -----------------------------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    N_DAYS = 30
    POINTS_PER_DAY = 96
    N_CLIENTS = 8
    TOTAL_POINTS = N_DAYS * POINTS_PER_DAY

    EMBED_DIM = 3
    EMBED_DELAY = 2

    LATENT_1D = 1
    LATENT_2D = 2

    EPOCHS = 80
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    MAX_TDA_DIM = 2
    TDA_SUBSAMPLE = 1000

    OUTPUT_FIG_DIR = "fig_smartgrid_v3"
    safe_mkdir(OUTPUT_FIG_DIR)
    OUTPUT_EXCEL_V3 = "SMARTGRID_TDA_AE_ANALYSIS_v3.xlsx"

    print("\n" + "=" * 75)
    print("RUNNING SMART-GRID EXPERIMENT V3 (SMALL/MEDIUM SCALE)")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # 1.2 Synthetic monthly data generator
    # -----------------------------------------------------------------------
    def generate_daily_profile(num_points=POINTS_PER_DAY,
                               base_load=1.0,
                               peak_amplitude=0.5,
                               noise_std=0.05,
                               random_phase=True):
        """
        Generate a single daily profile with:
            - A base level of consumption.
            - A sinusoidal component (day/night pattern).
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
        return profile

    def generate_monthly_profile(n_days=N_DAYS,
                                 points_per_day=POINTS_PER_DAY,
                                 base_load=1.0,
                                 peak_amplitude=0.5,
                                 noise_std=0.05):
        """
        Generate one month of consumption by concatenating daily profiles,
        with slight variations per day to mimic real fluctuations.
        """
        series = []
        for _ in range(n_days):
            base_d = base_load * np.random.uniform(0.9, 1.1)
            peak_d = peak_amplitude * np.random.uniform(0.8, 1.2)
            day_profile = generate_daily_profile(
                num_points=points_per_day,
                base_load=base_d,
                peak_amplitude=peak_d,
                noise_std=noise_std
            )
            series.append(day_profile)
        return np.concatenate(series)

    def inject_anomaly(series: np.ndarray,
                       anomaly_type: str = "fraud",
                       duration_points: int = 80):
        """
        Inject a simple anomaly into a consumption time series.

        anomaly_type can be one of:
            - "fraud": artificially reduced consumption.
            - "failure": drastic drop to near-zero.
            - "spike": short peak of abnormally high consumption.
        """
        series = series.copy()
        L = len(series)
        if duration_points >= L:
            return series
        start = np.random.randint(0, L - duration_points)
        end = start + duration_points

        if anomaly_type == "fraud":
            factor = np.random.uniform(0.1, 0.5)
            series[start:end] *= factor
        elif anomaly_type == "failure":
            series[start:end] *= np.random.uniform(0.0, 0.1)
        elif anomaly_type == "spike":
            bump = np.random.uniform(1.0, 2.0)
            series[start:end] += bump

        series = np.clip(series, 0, None)
        return series

    def simulate_clients(n_clients=N_CLIENTS,
                         n_days=N_DAYS,
                         points_per_day=POINTS_PER_DAY):
        """
        Simulate several client series; the first half are normal,
        the second half contain anomalies of random type.
        """
        all_series = []
        labels = []
        for i in range(n_clients):
            base_load = np.random.uniform(0.8, 1.3)
            peak_amp = np.random.uniform(0.4, 0.8)
            s = generate_monthly_profile(
                n_days=n_days,
                points_per_day=points_per_day,
                base_load=base_load,
                peak_amplitude=peak_amp,
                noise_std=0.06
            )
            if i < n_clients // 2:
                labels.append("normal")
            else:
                anomaly_type = np.random.choice(["fraud", "failure", "spike"])
                s = inject_anomaly(
                    s,
                    anomaly_type=anomaly_type,
                    duration_points=np.random.randint(40, 160)
                )
                labels.append(f"anomalous_{anomaly_type}")
            all_series.append(s)
        return np.array(all_series, dtype=np.float32), labels

    all_series_v3, labels_v3 = simulate_clients()
    describe_array("all_series_v3", all_series_v3)
    print("[INFO v3] Client labels:", labels_v3)

    # -----------------------------------------------------------------------
    # 1.3 Exploratory plots (points and mosaics)
    # -----------------------------------------------------------------------
    def plot_series_points(series: np.ndarray,
                           title: str,
                           filename: str):
        """
        Plot a single time series as a scatter plot of (time index, value).
        """
        t = np.arange(len(series))
        plt.figure(figsize=(10, 4))
        plt.scatter(t, series, s=8, alpha=0.8)
        plt.xlabel("Time index")
        plt.ylabel("Consumption (arbitrary units)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    idx_normal = 0
    idx_anom = N_CLIENTS - 1
    series_normal = all_series_v3[idx_normal]
    series_anom = all_series_v3[idx_anom]

    plot_series_points(
        series_normal,
        "Monthly consumption – normal client (points)",
        "v3_client_normal_points.png"
    )
    plot_series_points(
        series_anom,
        f"Monthly consumption – anomalous client ({labels_v3[idx_anom]}) (points)",
        "v3_client_anomalous_points.png"
    )

    def plot_two_series_comparison(series1: np.ndarray,
                                   series2: np.ndarray,
                                   labels: list,
                                   filename: str):
        """
        Plot two time series on the same scatter figure for comparison.
        """
        t = np.arange(len(series1))
        plt.figure(figsize=(10, 4))
        plt.scatter(t, series1, s=8, alpha=0.8, label=labels[0])
        plt.scatter(t, series2, s=8, alpha=0.8, label=labels[1])
        plt.xlabel("Time index")
        plt.ylabel("Consumption")
        plt.title("Comparison of two clients (points)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_two_series_comparison(
        series_normal,
        series_anom,
        ["Normal", f"Anomalous ({labels_v3[idx_anom]})"],
        "v3_comparison_normal_vs_anomalous_points.png"
    )

    def plot_clients_mosaic(all_series: np.ndarray,
                            labels: list,
                            n_rows: int = 2,
                            n_cols: int = 4,
                            filename: str = "v3_clients_mosaic.png"):
        """
        Create a mosaic showing multiple client series as scatter plots.

        The mosaic is particularly useful to illustrate the variety of
        normal vs anomalous patterns across the population.
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
        fig.suptitle("Clients mosaic – monthly consumption (points)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_clients_mosaic(all_series_v3, labels_v3)
    gc.collect()

    # -----------------------------------------------------------------------
    # 1.4 Delay embedding (3D) and global point cloud
    # -----------------------------------------------------------------------
    embedding_normal = delay_embedding_1d_series(
        series_normal,
        dimension=EMBED_DIM,
        delay=EMBED_DELAY
    )

    describe_array("embedding_normal", embedding_normal)

    def plot_3d_embedding(embedding: np.ndarray,
                          title: str,
                          filename: str):
        """
        Visualize a 3D delay embedding as a scatter plot in R^3.
        """
        if embedding.shape[1] < 3:
            raise ValueError("Embedding must have at least 3 dimensions for 3D plot.")
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embedding[:, 0],
                   embedding[:, 1],
                   embedding[:, 2],
                   s=4,
                   alpha=0.6)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+τ)")
        ax.set_zlabel("x(t+2τ)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_3d_embedding(
        embedding_normal,
        "3D delay embedding – normal client",
        "v3_embedding3d_normal_client.png"
    )

    # Build a global point cloud by concatenating embeddings for all clients
    embeddings_list = []
    for i in range(N_CLIENTS):
        emb_i = delay_embedding_1d_series(
            all_series_v3[i],
            dimension=EMBED_DIM,
            delay=EMBED_DELAY
        )
        embeddings_list.append(emb_i)

    global_embeddings = np.vstack(embeddings_list).astype(np.float32)
    describe_array("global_embeddings (raw)", global_embeddings)

    # Normalize the global point cloud (zero mean, unit variance per dimension)
    mean_emb = np.mean(global_embeddings, axis=0)
    std_emb = np.std(global_embeddings, axis=0) + 1e-8
    global_embeddings_norm = (global_embeddings - mean_emb) / std_emb
    describe_array("global_embeddings_norm", global_embeddings_norm)

    del global_embeddings  # free memory
    gc.collect()

    # -----------------------------------------------------------------------
    # 1.5 Autoencoder models (1D and 2D latent)
    # -----------------------------------------------------------------------
    def build_autoencoder(input_dim: int,
                          latent_dim: int,
                          learning_rate: float = LEARNING_RATE) -> Model:
        """
        Build a simple feedforward autoencoder with a given latent dimension.

        Architecture:
            - Dense(32, relu)
            - Dense(16, relu)
            - Dense(latent_dim, linear)  -> latent space
            - Dense(16, relu)
            - Dense(32, relu)
            - Dense(input_dim, linear)   -> reconstruction
        """
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(32, activation="relu")(inputs)
        x = layers.Dense(16, activation="relu")(x)
        latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)
        x = layers.Dense(16, activation="relu")(latent)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(input_dim, activation="linear")(x)
        model = keras.Model(inputs, outputs, name=f"AE_latent_{latent_dim}")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse"
        )
        return model

    ae1d = build_autoencoder(input_dim=EMBED_DIM, latent_dim=LATENT_1D)
    history_ae1d = ae1d.fit(
        global_embeddings_norm,
        global_embeddings_norm,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.1
    )
    print("[INFO v3] AE1D training finished.")

    ae2d = build_autoencoder(input_dim=EMBED_DIM, latent_dim=LATENT_2D)
    history_ae2d = ae2d.fit(
        global_embeddings_norm,
        global_embeddings_norm,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_split=0.1
    )
    print("[INFO v3] AE2D training finished.")

    def plot_training_history(history,
                              title: str,
                              filename: str):
        """
        Plot training and validation loss over epochs.
        """
        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_training_history(
        history_ae1d,
        "Training curve – AE1D",
        "v3_training_AE1D.png"
    )
    plot_training_history(
        history_ae2d,
        "Training curve – AE2D",
        "v3_training_AE2D.png"
    )

    # -----------------------------------------------------------------------
    # 1.6 Reconstruction and MSE analysis
    # -----------------------------------------------------------------------
    recon_ae1d = ae1d.predict(global_embeddings_norm, verbose=0)
    recon_ae2d = ae2d.predict(global_embeddings_norm, verbose=0)

    def mse_pointwise(original: np.ndarray,
                      reconstructed: np.ndarray) -> np.ndarray:
        """
        Compute MSE per point for two arrays with identical shape.
        """
        original = np.asarray(original, dtype=np.float32)
        reconstructed = np.asarray(reconstructed, dtype=np.float32)
        return np.mean((original - reconstructed) ** 2, axis=1)

    mse_ae1d = mse_pointwise(global_embeddings_norm, recon_ae1d)
    mse_ae2d = mse_pointwise(global_embeddings_norm, recon_ae2d)

    def plot_mse_histogram(mse_values: np.ndarray,
                           title: str,
                           filename: str):
        """
        Plot a histogram for MSE values.
        """
        plt.figure(figsize=(6, 4))
        plt.hist(mse_values, bins=40, alpha=0.7)
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
        plt.close()

    plot_mse_histogram(
        mse_ae1d,
        "Histogram of pointwise MSE – AE1D",
        "v3_hist_mse_AE1D.png"
    )
    plot_mse_histogram(
        mse_ae2d,
        "Histogram of pointwise MSE – AE2D",
        "v3_hist_mse_AE2D.png"
    )

    df_mse_stats = pd.DataFrame({
        "AE": ["AE1D", "AE2D"],
        "mse_mean": [np.mean(mse_ae1d), np.mean(mse_ae2d)],
        "mse_median": [np.median(mse_ae1d), np.median(mse_ae2d)],
        "mse_std": [np.std(mse_ae1d), np.std(mse_ae2d)]
    })

    print("[INFO v3] Global MSE statistics:")
    print(df_mse_stats)

    # -----------------------------------------------------------------------
    # 1.7 Topological Data Analysis (TDA) for version 3
    # -----------------------------------------------------------------------
    if HAS_RIPSER and HAS_PERSIM:
        # Subsample point clouds to keep TDA computations under control
        X_orig_sub = subsample_points(global_embeddings_norm,
                                      n_samples=TDA_SUBSAMPLE,
                                      seed=SEED)
        X_ae1d_sub = subsample_points(recon_ae1d,
                                      n_samples=TDA_SUBSAMPLE,
                                      seed=SEED + 1)
        X_ae2d_sub = subsample_points(recon_ae2d,
                                      n_samples=TDA_SUBSAMPLE,
                                      seed=SEED + 2)

        describe_array("X_orig_sub (v3)", X_orig_sub)
        describe_array("X_ae1d_sub (v3)", X_ae1d_sub)
        describe_array("X_ae2d_sub (v3)", X_ae2d_sub)

        dgms_original = compute_persistence_diagrams(
            X_orig_sub,
            maxdim=MAX_TDA_DIM
        )
        dgms_ae1d = compute_persistence_diagrams(
            X_ae1d_sub,
            maxdim=MAX_TDA_DIM
        )
        dgms_ae2d = compute_persistence_diagrams(
            X_ae2d_sub,
            maxdim=MAX_TDA_DIM
        )

        print("[INFO v3] TDA diagrams computed for original/AE1D/AE2D.")

        # Persistence diagrams for the three cases
        def plot_persistence_diagrams_three(dgms1,
                                            dgms2,
                                            dgms3,
                                            labels,
                                            filename):
            """
            Plot persistence diagrams for three different systems
            (e.g., Original, AE1D, AE2D) on a single figure.
            """
            plt.figure(figsize=(12, 4))
            for i, (dg, lab) in enumerate(zip([dgms1, dgms2, dgms3], labels), start=1):
                plt.subplot(1, 3, i)
                plot_diagrams(dg, show=False)
                plt.title(lab)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_persistence_diagrams_three(
            dgms_original,
            dgms_ae1d,
            dgms_ae2d,
            labels=["Original", "AE1D", "AE2D"],
            filename="v3_persistence_diagrams_3cases.png"
        )

        # Barcode plots for the three cases
        def plot_barcodes_three(dgms1,
                                dgms2,
                                dgms3,
                                labels,
                                filename):
            """
            Plot barcodes (lifetimes) for three different systems.
            """
            plt.figure(figsize=(12, 4))
            for i, (dg, lab) in enumerate(zip([dgms1, dgms2, dgms3], labels), start=1):
                plt.subplot(1, 3, i)
                plot_diagrams(dg, lifetime=True, show=False)
                plt.title(f"Barcodes – {lab}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_barcodes_three(
            dgms_original,
            dgms_ae1d,
            dgms_ae2d,
            labels=["Original", "AE1D", "AE2D"],
            filename="v3_barcodes_3cases.png"
        )

        # Betti curves for dimensions 0, 1, and 2
        births_all = []
        deaths_all = []
        for dgms in [dgms_original, dgms_ae1d, dgms_ae2d]:
            for dgm in dgms:
                if dgm is None or len(dgm) == 0:
                    continue
                dgm_arr = np.asarray(dgm, dtype=np.float32)
                births_all.append(np.min(dgm_arr[:, 0]))
                finite_deaths = dgm_arr[:, 1][np.isfinite(dgm_arr[:, 1])]
                if finite_deaths.size > 0:
                    deaths_all.append(np.max(finite_deaths))

        if len(deaths_all) == 0:
            t_values = np.linspace(0.0, 1.0, 100)
        else:
            t_min = float(min(births_all))
            t_max = float(max(deaths_all))
            if not np.isfinite(t_min):
                t_min = 0.0
            if not np.isfinite(t_max) or t_max <= t_min:
                t_max = t_min + 1.0
            t_values = np.linspace(t_min, t_max, 200)

        betti_orig = {}
        betti_ae1d = {}
        betti_ae2d = {}
        for dim in range(MAX_TDA_DIM + 1):
            if dim < len(dgms_original):
                betti_orig[dim] = betti_curve_from_diagram(
                    dgms_original[dim],
                    t_values
                )
            if dim < len(dgms_ae1d):
                betti_ae1d[dim] = betti_curve_from_diagram(
                    dgms_ae1d[dim],
                    t_values
                )
            if dim < len(dgms_ae2d):
                betti_ae2d[dim] = betti_curve_from_diagram(
                    dgms_ae2d[dim],
                    t_values
                )

        def plot_betti_curves_three(betti_dicts,
                                    labels,
                                    t_values,
                                    max_dim,
                                    filename):
            """
            Plot Betti curves for multiple systems across dimensions 0..max_dim.
            """
            plt.figure(figsize=(12, 4))
            for dim in range(max_dim + 1):
                plt.subplot(1, max_dim + 1, dim + 1)
                for betti, lab in zip(betti_dicts, labels):
                    if dim in betti:
                        plt.plot(t_values, betti[dim], label=lab)
                plt.title(f"Betti curve H{dim}")
                plt.xlabel("Scale parameter t")
                plt.ylabel("Betti number")
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FIG_DIR, filename), dpi=150)
            plt.close()

        plot_betti_curves_three(
            [betti_orig, betti_ae1d, betti_ae2d],
            ["Original", "AE1D", "AE2D"],
            t_values,
            max_dim=2,
            filename="v3_betti_curves_3cases.png"
        )

        # Wasserstein distances for H0, H1, H2 (when available)
        rows = []
        for dim in range(min(len(dgms_original),
                             len(dgms_ae1d),
                             len(dgms_ae2d))):
            d_ae1d = wasserstein_distance_simple(
                dgms_original[dim],
                dgms_ae1d[dim]
            )
            d_ae2d = wasserstein_distance_simple(
                dgms_original[dim],
                dgms_ae2d[dim]
            )
            rows.append({"AE": "AE1D",
                         "dimension": f"H{dim}",
                         "wasserstein": d_ae1d})
            rows.append({"AE": "AE2D",
                         "dimension": f"H{dim}",
                         "wasserstein": d_ae2d})
        df_wass_v3 = pd.DataFrame(rows)

        print("[INFO v3] Wasserstein distances:")
        print(df_wass_v3)
    else:
        print("[INFO v3] TDA is skipped because ripser or persim is not available.")
        df_wass_v3 = pd.DataFrame(
            columns=["AE", "dimension", "wasserstein"]
        )

    # -----------------------------------------------------------------------
    # 1.8 Export v3 results to Excel
    # -----------------------------------------------------------------------
    engine = choose_excel_engine()
    print(f"[INFO v3] Using Excel engine: {engine}")

    with pd.ExcelWriter(OUTPUT_EXCEL_V3, engine=engine) as writer:
        df_loss_ae1d = pd.DataFrame({
            "epoch": np.arange(1, EPOCHS + 1),
            "loss": history_ae1d.history["loss"],
            "val_loss": history_ae1d.history.get(
                "val_loss", [np.nan] * EPOCHS
            )
        })
        df_loss_ae1d.to_excel(
            writer,
            sheet_name="losses_ae1d",
            index=False
        )

        df_loss_ae2d = pd.DataFrame({
            "epoch": np.arange(1, EPOCHS + 1),
            "loss": history_ae2d.history["loss"],
            "val_loss": history_ae2d.history.get(
                "val_loss", [np.nan] * EPOCHS
            )
        })
        df_loss_ae2d.to_excel(
            writer,
            sheet_name="losses_ae2d",
            index=False
        )

        df_mse_point = pd.DataFrame({
            "mse_ae1d": mse_ae1d,
            "mse_ae2d": mse_ae2d
        })
        df_mse_point.to_excel(
            writer,
            sheet_name="mse_pointwise",
            index=False
        )

        df_mse_stats.to_excel(
            writer,
            sheet_name="mse_stats",
            index=False
        )

        df_wass_v3.to_excel(
            writer,
            sheet_name="wasserstein",
            index=False
        )

        df_meta = pd.DataFrame([{
            "N_DAYS": N_DAYS,
            "POINTS_PER_DAY": POINTS_PER_DAY,
            "N_CLIENTS": N_CLIENTS,
            "TOTAL_POINTS": TOTAL_POINTS,
            "EMBED_DIM": EMBED_DIM,
            "EMBED_DELAY": EMBED_DELAY,
            "LATENT_1D": LATENT_1D,
            "LATENT_2D": LATENT_2D,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "MAX_TDA_DIM": MAX_TDA_DIM,
            "TDA_SUBSAMPLE": TDA_SUBSAMPLE,
            "SEED": SEED
        }])
        df_meta.to_excel(
            writer,
            sheet_name="metadata",
            index=False
        )

    print(f"[INFO v3] Excel file saved as: {OUTPUT_EXCEL_V3}")
    print("[INFO v3] Figures are in folder:", OUTPUT_FIG_DIR)
    print("=" * 75 + "\n")


# ============================================================================
# 2. FULL COMPARISON EXPERIMENT (EXTENDED VERSION OF 03D.py, RAM-FRIENDLY)
# ============================================================================

def run_smartgrid_full_comparison_ram_friendly():
    """
    Run a full comparison experiment inspired by the extended 03D.py,
    but with explicit constraints to avoid RAM exhaustion.

    Main features:
        - Larger number of clients than v3, but still moderate.
        - 14 days of hourly data per client (24 * 14 points).
        - Anomaly injection (fraud, failure, peaks).
        - Exploratory point plots and mosaics.
        - Delay embeddings for all clients (dimension 3, delay 2).
        - Two autoencoders (1D and 2D latent) with train/test split.
        - MSE statistics and pointwise distributions.
        - TDA on subsampled embeddings:
              * Persistence diagrams
              * Barcodes
              * Wasserstein distances
        - Export of all results to:
              experiment_outputs/smartgrid_topo_results.xlsx
        - Figures under:
              experiment_outputs/figures/

    This experiment integrates and extends what 03D.py was doing, but
    with parameters that are significantly safer with respect to memory.
    """

    print("\n" + "=" * 75)
    print("RUNNING FULL COMPARISON EXPERIMENT (RAM-FRIENDLY)")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # 2.1 Configuration (carefully chosen for RAM)
    # -----------------------------------------------------------------------
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    N_CLIENTS = 150             # smaller than very large setups, to protect RAM
    HOURS_PER_DAY = 24
    N_DAYS = 14
    HORIZON_HOURS = HOURS_PER_DAY * N_DAYS

    TEST_SIZE = 0.25
    EPOCHS = 80
    BATCH_SIZE = 64

    EMBED_DIM = 3
    EMBED_DELAY = 2

    TDA_SUBSAMPLE = 1000
    TDA_MAXDIM = 2

    BASE_OUTPUT_DIR = Path("experiment_outputs")
    FIG_DIR = BASE_OUTPUT_DIR / "figures"
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_PATH = BASE_OUTPUT_DIR / "smartgrid_topo_results.xlsx"

    print(f"[FULL] Output directory: {BASE_OUTPUT_DIR.resolve()}")
    print(f"[FULL] Figures directory: {FIG_DIR.resolve()}")

    # -----------------------------------------------------------------------
    # 2.2 Data generation for the extended experiment
    # -----------------------------------------------------------------------
    def daily_base_curve(num_points: int = 24,
                         morning_peak: float = 1.3,
                         evening_peak: float = 1.5,
                         base_level: float = 0.6) -> np.ndarray:
        """
        Generate a smooth daily load curve with morning and evening peaks,
        plus a slight sinusoidal variation.

        This base curve is used as a building block for multi-day series.
        """
        h = np.arange(num_points, dtype=np.float32)
        load = np.ones_like(h) * base_level
        load += morning_peak * np.exp(-0.5 * ((h - 8) / 2.5) ** 2)
        load += evening_peak * np.exp(-0.5 * ((h - 20) / 2.5) ** 2)
        load += 0.15 * np.sin(2 * np.pi * h / 24.0)
        return load.astype(np.float32)

    def weekly_series(num_days: int = 14) -> np.ndarray:
        """
        Construct a multi-day series by concatenating the daily base curve
        with slight random amplitude shifts and weekend modulation.
        """
        series = []
        for d in range(num_days):
            base = daily_base_curve()
            # Weekend dampening
            if d % 7 in (5, 6):
                base = base * 0.8
            # Small day-to-day variability
            scale = 1.0 + 0.05 * np.random.randn()
            day_curve = base * scale
            series.append(day_curve)
        out = np.concatenate(series).astype(np.float32)
        return out

    def inject_smartgrid_anomalies(series: np.ndarray,
                                   p_fraud: float = 0.06,
                                   p_failure: float = 0.04,
                                   p_spike: float = 0.06) -> (np.ndarray, bool):
        """
        Inject typical smart-grid anomalies into a series:

            - fraud: artificially reduced consumption over a short window.
            - failure: sudden drop to a near-constant low level.
            - spike: sharp, localized overconsumption.

        Returns
        -------
        new_series : np.ndarray
            Possibly modified series with anomalies.
        had_anomaly : bool
            True if any anomaly was actually injected.
        """
        series = series.copy()
        L = len(series)
        had_anomaly = False

        # Fraud anomaly
        if np.random.rand() < p_fraud:
            start = np.random.randint(0, L - 6)
            end = start + np.random.randint(3, 8)
            factor = np.random.uniform(0.3, 0.6)
            series[start:end] = series[start:end] * factor
            had_anomaly = True

        # Failure anomaly
        if np.random.rand() < p_failure:
            start = np.random.randint(0, L - 10)
            end = start + np.random.randint(4, 12)
            baseline = np.mean(series[max(0, start - 3):start + 1])
            series[start:end] = baseline + 0.01 * np.random.randn(end - start)
            had_anomaly = True

        # Spike anomaly
        if np.random.rand() < p_spike:
            idx = np.random.randint(0, L)
            series[idx] = series[idx] * np.random.uniform(1.8, 2.5)
            had_anomaly = True

        return np.clip(series, 0, None).astype(np.float32), had_anomaly

    def build_smartgrid_dataset(n_clients: int,
                                horizon_hours: int) -> (np.ndarray, np.ndarray):
        """
        Build a dataset consisting of n_clients smart-grid series, each
        with horizon_hours time points, and a binary anomaly label for
        each client.
        """
        all_series = []
        anomaly_labels = []
        for _ in range(n_clients):
            s = weekly_series(num_days=horizon_hours // 24)
            s, had_anomaly = inject_smartgrid_anomalies(s)
            # Add slow oscillation and a small noise term
            slow_noise = 0.05 * np.sin(
                2 * np.pi * np.arange(len(s)) / (24 * 3)
            )
            s = s + slow_noise.astype(np.float32)
            s = s + 0.02 * np.random.randn(*s.shape).astype(np.float32)
            all_series.append(s.astype(np.float32))
            anomaly_labels.append(1 if had_anomaly else 0)
        return np.vstack(all_series), np.array(anomaly_labels, dtype=np.int32)

    X_series, y_anom = build_smartgrid_dataset(N_CLIENTS, HORIZON_HOURS)
    describe_array("X_series (FULL)", X_series)
    print("[FULL] Number of clients:", N_CLIENTS)
    print("[FULL] Horizon (hours):", HORIZON_HOURS)
    print("[FULL] Binary anomaly labels (first 20):", y_anom[:20])

    # -----------------------------------------------------------------------
    # 2.3 Exploratory plots for full experiment
    # -----------------------------------------------------------------------
    hours_axis = np.arange(HORIZON_HOURS)

    # Single client example
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(hours_axis, X_series[0], s=10, label="Client 0 (simulated)")
    plt.plot(hours_axis, X_series[0], alpha=0.4)
    plt.title("Simulated consumption series – Client 0")
    plt.xlabel("Hour")
    plt.ylabel("Relative consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "full_client0_points.png")
    plt.close(fig)

    # Healthy vs anomalous example
    s_healthy = weekly_series(num_days=HORIZON_HOURS // 24)
    s_anom, _ = inject_smartgrid_anomalies(s_healthy)
    fig = plt.figure(figsize=(10, 4))
    plt.scatter(hours_axis, s_healthy, s=10, label="Healthy series", alpha=0.7)
    plt.scatter(hours_axis, s_anom, s=10, label="Series with anomaly", alpha=0.7)
    plt.title("Comparison: healthy vs anomalous series")
    plt.xlabel("Hour")
    plt.ylabel("Relative consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "full_healthy_vs_anomalous.png")
    plt.close(fig)

    # Mosaic of four example clients
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    clients_demo = [0, 1, 2, 3]
    for ax, c_idx in zip(axes.flat, clients_demo):
        ax.scatter(hours_axis, X_series[c_idx], s=6)
        ax.plot(hours_axis, X_series[c_idx], alpha=0.4)
        ax.set_title(f"Client {c_idx} (anom={y_anom[c_idx]})")
    axes[1, 0].set_xlabel("Hour")
    axes[1, 1].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Consumption")
    axes[1, 0].set_ylabel("Consumption")
    plt.suptitle("Mosaic of simulated grid data (4 example clients)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG_DIR / "full_clients_mosaic.png")
    plt.close(fig)

    gc.collect()

    # -----------------------------------------------------------------------
    # 2.4 Delay embeddings for all clients
    # -----------------------------------------------------------------------
    def build_embedded_dataset(X: np.ndarray,
                               dim: int,
                               delay: int):
        """
        Build delay embeddings for all client series and stack them
        into a single point cloud.

        Returns
        -------
        X_emb_list : list of np.ndarray
            List of embeddings, one per client.
        X_emb_stack : np.ndarray
            Single array stacking all client embeddings.
        index_slices : list of (start, end)
            For each client i, the range of rows [start, end) in X_emb_stack
            that correspond to that client.
        """
        X_emb_list = []
        index_slices = []
        all_points = []
        start = 0
        for i in range(X.shape[0]):
            emb_i = delay_embedding_1d_series(X[i], dimension=dim, delay=delay)
            X_emb_list.append(emb_i)
            end = start + emb_i.shape[0]
            index_slices.append((start, end))
            all_points.append(emb_i)
            start = end
        X_emb_stack = np.vstack(all_points).astype(np.float32)
        return X_emb_list, X_emb_stack, index_slices

    X_emb_list, X_emb_stack, index_slices = build_embedded_dataset(
        X_series,
        dim=EMBED_DIM,
        delay=EMBED_DELAY
    )
    describe_array("X_emb_stack (FULL)", X_emb_stack)

    # 3D embedding visualization for client 0
    emb_demo = X_emb_list[0]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(emb_demo[:, 0], emb_demo[:, 1], emb_demo[:, 2], s=6)
    ax.set_title("Delay embedding (Client 0)")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t+τ)")
    ax.set_zlabel("x(t+2τ)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "full_embedding_3d_client0.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 2.5 Autoencoders for the full experiment
    # -----------------------------------------------------------------------
    input_dim = EMBED_DIM

    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(
        X_emb_stack,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

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
    ae_1d.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    hist_1d = ae_1d.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print(f"[FULL] AE1D final train loss: {hist_1d.history['loss'][-1]:.6f} | "
          f"final val loss: {hist_1d.history['val_loss'][-1]:.6f}")

    Z_1d = encoder_1d.predict(X_emb_stack, verbose=0).astype(np.float32)
    Xrec_1d = ae_1d.predict(X_emb_stack, verbose=0).astype(np.float32)
    describe_array("Z_1d (FULL)", Z_1d)
    describe_array("Xrec_1d (FULL)", Xrec_1d)

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
    ae_2d.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    hist_2d = ae_2d.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print(f"[FULL] AE2D final train loss: {hist_2d.history['loss'][-1]:.6f} | "
          f"final val loss: {hist_2d.history['val_loss'][-1]:.6f}")

    Z_2d = encoder_2d.predict(X_emb_stack, verbose=0).astype(np.float32)
    Xrec_2d = ae_2d.predict(X_emb_stack, verbose=0).astype(np.float32)
    describe_array("Z_2d (FULL)", Z_2d)
    describe_array("Xrec_2d (FULL)", Xrec_2d)

    # -----------------------------------------------------------------------
    # 2.6 MSE calculations and basic statistics
    # -----------------------------------------------------------------------
    mse_pt_1d = np.mean((X_emb_stack - Xrec_1d) ** 2, axis=1).astype(np.float32)
    mse_pt_2d = np.mean((X_emb_stack - Xrec_2d) ** 2, axis=1).astype(np.float32)

    mse_mean_1d = float(mse_pt_1d.mean())
    mse_med_1d = float(np.median(mse_pt_1d))
    mse_mean_2d = float(mse_pt_2d.mean())
    mse_med_2d = float(np.median(mse_pt_2d))

    print("[FULL] MSE AE1D (mean, median):", mse_mean_1d, mse_med_1d)
    print("[FULL] MSE AE2D (mean, median):", mse_mean_2d, mse_med_2d)

    # -----------------------------------------------------------------------
    # 2.7 TDA with subsampling
    # -----------------------------------------------------------------------
    wasserstein_ae1d = {}
    wasserstein_ae2d = {}
    dims_for_plots = []

    if HAS_RIPSER and HAS_PERSIM:
        Xorig_sub = subsample_points(
            X_emb_stack,
            n_samples=TDA_SUBSAMPLE,
            seed=RANDOM_SEED
        )
        Xrec1_sub = subsample_points(
            Xrec_1d,
            n_samples=TDA_SUBSAMPLE,
            seed=RANDOM_SEED + 1
        )
        Xrec2_sub = subsample_points(
            Xrec_2d,
            n_samples=TDA_SUBSAMPLE,
            seed=RANDOM_SEED + 2
        )

        describe_array("Xorig_sub (FULL)", Xorig_sub)
        describe_array("Xrec1_sub (FULL)", Xrec1_sub)
        describe_array("Xrec2_sub (FULL)", Xrec2_sub)

        dgms_orig = compute_persistence_diagrams(
            Xorig_sub,
            maxdim=TDA_MAXDIM
        )
        dgms_1d = compute_persistence_diagrams(
            Xrec1_sub,
            maxdim=TDA_MAXDIM
        )
        dgms_2d = compute_persistence_diagrams(
            Xrec2_sub,
            maxdim=TDA_MAXDIM
        )

        max_common_dim = min(len(dgms_orig),
                             len(dgms_1d),
                             len(dgms_2d)) - 1

        for dim in range(max_common_dim + 1):
            d1 = wasserstein_distance_simple(dgms_orig[dim], dgms_1d[dim])
            d2 = wasserstein_distance_simple(dgms_orig[dim], dgms_2d[dim])
            key = f"H{dim}"
            wasserstein_ae1d[key] = float(d1)
            wasserstein_ae2d[key] = float(d2)
            dims_for_plots.append(dim)

        print("[FULL] Wasserstein vs AE1D:", wasserstein_ae1d)
        print("[FULL] Wasserstein vs AE2D:", wasserstein_ae2d)

        # Simple diagrams figure for full experiment
        plt.figure(figsize=(12, 4))
        for i, (dg, label) in enumerate(
            zip([dgms_orig, dgms_1d, dgms_2d],
                ["Original", "AE1D", "AE2D"]),
            start=1
        ):
            plt.subplot(1, 3, i)
            plot_diagrams(dg, show=False)
            plt.title(f"Diagrams – {label}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "full_persistence_diagrams_3cases.png")
        plt.close()

    else:
        print("[FULL] TDA skipped: ripser or persim not available.")

    # -----------------------------------------------------------------------
    # 2.8 Export extended experiment results to Excel
    # -----------------------------------------------------------------------
    engine = choose_excel_engine()
    print(f"[FULL] Excel engine used: {engine}")

    with pd.ExcelWriter(EXCEL_PATH, engine=engine) as writer:
        df_loss_1d = pd.DataFrame({
            "epoch": np.arange(1, len(hist_1d.history["loss"]) + 1, dtype=np.int32),
            "loss_train": np.asarray(hist_1d.history["loss"], dtype=np.float32),
            "loss_val": np.asarray(hist_1d.history["val_loss"], dtype=np.float32)
        })
        df_loss_1d.to_excel(writer, sheet_name="losses_ae1d", index=False)

        df_loss_2d = pd.DataFrame({
            "epoch": np.arange(1, len(hist_2d.history["loss"]) + 1, dtype=np.int32),
            "loss_train": np.asarray(hist_2d.history["loss"], dtype=np.float32),
            "loss_val": np.asarray(hist_2d.history["val_loss"], dtype=np.float32)
        })
        df_loss_2d.to_excel(writer, sheet_name="losses_ae2d", index=False)

        df_stats = pd.DataFrame({
            "model": ["AE_1D", "AE_2D"],
            "mse_mean": [mse_mean_1d, mse_mean_2d],
            "mse_median": [mse_med_1d, mse_med_2d],
            "mse_std": [
                float(mse_pt_1d.std(ddof=1)),
                float(mse_pt_2d.std(ddof=1))
            ]
        })
        df_stats.to_excel(writer, sheet_name="mse_stats", index=False)

        pd.DataFrame({"mse_point": mse_pt_1d.astype(np.float32)}).to_excel(
            writer,
            sheet_name="mse_point_ae1d",
            index=False
        )
        pd.DataFrame({"mse_point": mse_pt_2d.astype(np.float32)}).to_excel(
            writer,
            sheet_name="mse_point_ae2d",
            index=False
        )

        if len(wasserstein_ae1d) > 0 or len(wasserstein_ae2d) > 0:
            dims_txt = [f"H{d}" for d in dims_for_plots]
            df_wass = pd.DataFrame({
                "dimension": dims_txt,
                "wasserstein_vs_AE1D": [
                    wasserstein_ae1d.get(k, np.nan) for k in dims_txt
                ],
                "wasserstein_vs_AE2D": [
                    wasserstein_ae2d.get(k, np.nan) for k in dims_txt
                ]
            })
        else:
            df_wass = pd.DataFrame(
                columns=[
                    "dimension",
                    "wasserstein_vs_AE1D",
                    "wasserstein_vs_AE2D"
                ]
            )
        df_wass.to_excel(writer, sheet_name="wasserstein", index=False)

        df_meta = pd.DataFrame({
            "parameter": [
                "N_CLIENTS", "HORIZON_HOURS", "EMBED_DIM", "EMBED_DELAY",
                "TEST_SIZE", "EPOCHS", "BATCH_SIZE",
                "TDA_SUBSAMPLE", "TDA_MAXDIM", "EXCEL_ENGINE",
                "mse_mean_ae1d", "mse_median_ae1d",
                "mse_mean_ae2d", "mse_median_ae2d"
            ],
            "value": [
                N_CLIENTS, HORIZON_HOURS, EMBED_DIM, EMBED_DELAY,
                TEST_SIZE, EPOCHS, BATCH_SIZE,
                TDA_SUBSAMPLE, TDA_MAXDIM, engine,
                mse_mean_1d, mse_med_1d,
                mse_mean_2d, mse_med_2d
            ]
        })
        df_meta.to_excel(writer, sheet_name="metadata", index=False)

    print("\n=== FULL COMPARISON EXPERIMENT COMPLETED (RAM-FRIENDLY) ===")
    print(f"- Figures saved in: {FIG_DIR.resolve()}")
    print(f"- Excel file saved in: {EXCEL_PATH.resolve()}")
    print("=" * 75 + "\n")


# ============================================================================
# 3. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run Version 3 (small/medium scale) – reproduces and extends 03.py
    run_smartgrid_version3()

    # Run full comparison (extended experiment) – integrated 03D.py behavior
    # but with RAM-friendly configuration. If you only want the small
    # experiment, you can comment out the next line.
    run_smartgrid_full_comparison_ram_friendly()



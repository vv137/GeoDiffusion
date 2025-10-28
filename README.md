# GeoDiffusion: A Toy Experiment for Loss Function Validation

This repository tests the hypothesis that a composite loss function, similar to that in AlphaFold 3 (incorporating both MSE and a local structure term like smooth LDDT), can more effectively learn both global and local geometric features compared to a standard MSE-based diffusion loss (like in EDM).

**Problem**: Standard MSE loss in diffusion models might excel at capturing the global structure but struggle with fine-grained local geometry (e.g., precise bond angles or local motifs). Composite losses, like AF3's, aim to address this by explicitly penalizing deviations in local structure.

**Toy Task**: We model a 2D polymer chain composed of rigid V-shaped "monomers." The vertices of these V-shapes lie on a sine wave (global structure), while the internal angle of each V-shape is fixed (local structure).

* **Global Task**: Learn the sine wave shape. Measured by MSE on vertex positions.
* **Local Task**: Learn the fixed internal angle (e.g., 45 degrees). Measured by Mean Absolute Error (MAE) on the predicted internal angles.

## Experiments

We compare four conditions:

1. **EDM (MSE Only)**: Uses EDM-style weighting on a standard L2 (MSE) loss between predicted and true clean structures. (`--loss_type edm --smooth_lddt_weight 0.0`)
2. **EDM + SmoothLDDT**: Uses EDM-style MSE weighting plus the AF3-inspired smooth\_lddt\_loss. (`--loss_type edm --smooth_lddt_weight 1.0`)
3. **AF3 Weight (MSE Only)**: Uses AF3-style weighting on the L2 (MSE) loss. (`--loss_type af3 --smooth_lddt_weight 0.0`)
4. **AF3 + SmoothLDDT**: Uses AF3-style MSE weighting plus the smooth\_lddt\_loss. This mimics the AF3 diffusion loss structure. (`--loss_type af3 --smooth_lddt_weight 1.0`)

## Files

* `generate_dataset.py`: Generates the toy dataset of 2D polymers.
* `train.py`: The main training script. It can handle different loss functions and weighting strategies.
* `model.py`: Defines the GeoDiffusionTransformer, a conditional transformer model for the denoising task.
* `plot_results.py`: A script to plot the results from the training logs.
* `sbatch.sh`, `run_experiment_array.sh`: Slurm submission scripts for running experiments on a cluster.
* `logs_transformer/`: Directory containing the training logs for each experiment.
* `transformer_experiment_comparison.png`: A plot comparing the results of the four experiments.

## How to Run

1. **Setup Environment**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install torch numpy pandas matplotlib
    ```

2. **Generate Dataset**

    ```bash
    python generate_dataset.py
    ```

3. **Run Experiments**

    You can run the experiments sequentially using `sbatch.sh`:

    ```bash
    bash sbatch.sh
    ```

    Or, if you are using a Slurm cluster, you can use the experiment array script:

    ```bash
    sbatch run_experiment_array.sh
    ```

4. **Plot Results**

    ```bash
    bash plot_all_results.sh
    ```

## Results

The results of the experiments are summarized in the plot below:

From the "Global Accuracy" plot, we can see that all four models achieve a low Mean Squared Error on the vertex positions, indicating that they are all able to learn the global sine wave structure. The AF3-style weighting schemes appear to converge slightly faster.

The "Local Accuracy" plot shows a more distinct difference between the models. The models that include the SmoothLDDT loss term ("EDM + SmoothLDDT" and "AF3 + SmoothLDDT") achieve a significantly lower Mean Absolute Error on the internal angles. This suggests that the composite loss function is indeed more effective at learning the local geometric features of the polymer chain.

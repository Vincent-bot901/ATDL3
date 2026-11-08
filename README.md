# MeanFlow 2D Conditioning Benchmark



Code for training a 2D MeanFlow model and benchmarking inference conditioning methods (Oracle, Local Gaussian Conditioning, Vanilla DPS, and Hybrid pushforward) on a constrained Gaussian mixture.



## Requirements



Install the dependencies:



```bash

pip install torch numpy matplotlib

```



Or install via `requirements.txt`:



```bash

pip install -r requirements.txt

```



## Quickstart



Run the full pipeline (model training followed by both experiment suites):



```bash

python 2d_final.py

```



## Pipeline Overview



The script executes sequentially from `2d_final.py`:



| Routine | Description | Output Files / Logs |

| :--- | :--- | :--- |

| `train_meanflow()` | Trains the 2D MeanFlow MLP on GMM data using forward-mode JVP velocity targets (3,000 epochs). | Trained model instance |

| `run_and_plot_flow_geometry(model)` | Evaluates a 25-particle grid across 4 methods: Oracle (50 steps), LGC Flow (50 steps), Hybrid ($t_s=0.5, K=5$), and 1-Step + $JJ^\top$. | `fig_E4_flow_geometry.png`<br>`fig_E4_flow_geometry.pdf` |

| `run_and_plot_meanflow_experiment(model)` | Runs step-budget ablations ($N \in \{1, 2, 4, 8, 16, 32, 64, 128\}$) comparing Oracle, LGC (Full Covariance), and Vanilla DPS. | Console log (MSE, LogP)<br>`meanflow_ablation_N_hires.png`<br>`meanflow_ablation_N_vector.pdf` |



## Running Experiments Individually



You can toggle individual evaluation suites at the bottom of the script:



```python

if __name__ == '__main__':

    # Train model

    mf_model = train_meanflow(epochs=3000, batch_size=512, lr=6e-4)



    # 4-panel flow geometry figure (Oracle vs LGC vs Hybrid vs 1-Step)

    run_and_plot_flow_geometry(mf_model)



    # Step budget ablation across N = 1 to 128

    run_and_plot_meanflow_experiment(mf_model, steps_list=[1, 2, 4, 8, 16, 32, 64, 128])

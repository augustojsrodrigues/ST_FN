# Modified-opportunistic inspection policy

This repository contains a Streamlit app to evaluate a modified-opportunistic inspection policy with false negative errors.

The app evaluates a fixed pair `(S, T)`. It does not optimize the policy and does not run differential evolution.

## Policy variables

- `S`: opportunity acceptance threshold
- `T`: scheduled inspection interval

The app requires `S < T`. If `S >= T`, the interface stops and warns the user.

## Main inputs

- `mu_x`: mean time to defect arrival
- `mu_h`: mean delay time from defect to failure
- `mu_z`: mean time between opportunities
- `CF`: corrective replacement cost
- `CP`: preventive replacement cost
- `CI`: scheduled inspection cost
- `CO`: opportunistic inspection cost
- `beta_s`: false negative probability in scheduled inspections
- `beta_o`: false negative probability in opportunistic inspections
- `n_cycles`: number of simulated decision steps
- `seed`: random seed

## Outputs

- Cost rate
- MTBOF
- PFRBO
- LOM
- Probability of each event case

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the `assets` folder.
3. Open Streamlit Community Cloud.
4. Select the repository and set `app.py` as the main file.

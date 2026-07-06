Eye Gaze Interaction Project

This repository contains scripts and notes for an independent analysis of an eye-gaze interaction dataset. The project focuses on using information-theoretic methods, especially transfer entropy and partial information decomposition, to study how head and eye movement relate to task performance under degraded gaze-interaction conditions.

Project Motivation

The original dataset involves participants completing target-selection trials under different layouts and tracking modes. Some tracking modes degrade the interaction signal through bias and jitter. This creates a useful setting for asking whether head and eye movement contain meaningful information about task performance, or whether apparent relationships are partly driven by artifacts such as sample length, noise, and time dependence.

The main goal of this project is to replicate and extend transfer entropy-style results using the IDTxl Python package.

Research Questions

The project is currently organized around the following questions:

Do head and eye movement show directional temporal coupling during gaze-interaction tasks?
Are transfer entropy measures associated with task performance under degraded interaction conditions?
Do these relationships remain after accounting for sample length and shuffled-baseline corrections?
Can partial information decomposition help separate unique, redundant, and synergistic contributions from head and eye movement?
Methods

The project uses information-theoretic tools including:

Transfer entropy
Corrected transfer entropy using shuffled baselines
Partial information decomposition
Block-level aggregation by participant, layout, and tracking mode
Correlation and exploratory modelling against task performance

The main Python package used for information-theoretic analysis is:

IDTxl
Dataset Structure

The original dataset is expected to include frame-level and trial-level data. The relevant analysis unit is generally:

participant × layout × tracking mode

Within each condition block, participants completed target-selection trials. Frame-level analyses focus on periods where the trial state is active.

The data are not included directly in this repository unless otherwise stated.

Current Status

This repository is currently in the setup and replication stage. The immediate goals are:

Install and test IDTxl.
Load the frame-level gaze-interaction data.
Reconstruct trial-level head and eye time series.
Estimate transfer entropy in both directions:
eye → head
head → eye
Aggregate results to the block level.
Compare replicated results against the original analysis.
Extend the analysis using partial information decomposition.
Environment Setup

This project is developed on Arch Linux using a Python virtual environment.

Example setup:

uv venv --python 3.11
source .venv/bin/activate.fish
python -m pip install --upgrade pip setuptools wheel
python -m pip install Cython numpy scipy matplotlib h5py networkx pandas jpype1 statsmodels

Install IDTxl from source:

git clone https://github.com/pwollstadt/IDTxl.git
cd IDTxl
python -m pip install -e . --no-build-isolation

Test the installation:

python -c "from idtxl.data import Data; from idtxl.bivariate_te import BivariateTE; from idtxl.bivariate_pid import BivariatePID; print('IDTxl import works')"
Planned Repository Organization
.
├── data/                  # Local data files, ignored if large or private
├── notebooks/             # Exploratory notebooks
├── scripts/               # Analysis scripts
├── results/               # Generated tables and figures
├── docs/                  # Notes and documentation
├── README.md
└── .gitignore
Notes

This project is exploratory. Transfer entropy results should be interpreted carefully, especially when raw TE relationships weaken after shuffled-baseline correction or sample-length controls. The goal is not only to estimate information flow, but also to understand which parts of the signal reflect meaningful behavioural structure and which parts may reflect statistical artifacts.
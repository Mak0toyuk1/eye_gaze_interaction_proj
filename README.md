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


This project is exploratory. Transfer entropy results should be interpreted carefully, especially when raw TE relationships weaken after shuffled-baseline correction or sample-length controls. The goal is not only to estimate information flow, but also to understand which parts of the signal reflect meaningful behavioural structure and which parts may reflect statistical artifacts.
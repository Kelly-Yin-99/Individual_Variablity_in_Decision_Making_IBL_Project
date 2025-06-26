Neural Dynamics Analysis

This repository contains some analysis pipeline for investigating neural dynamics, behavioral variability, and brain state transitions using 
electrophysiological data collected across cortical regions. The analysis includes querying raw data, computing spiking and trajectory metrics, 
performing Hidden Markov Modeling (HMM), spectral analysis, and Bayesian timescale estimation.

Folder Structure and Descriptions:

Query Ephys/
This folder contains scripts for querying and organizing session and insertion information across cortical regions.

-find_phys.py: 

Searches for electrophysiological sessions and probe insertions based on predefined session IDs that match the following task protocols:task_protocols = [
    '_iblrig_tasks_ephysChoiceWorld6.*', 
    '_iblrig_tasks_ephysChoiceWorld7.*'
]

-find_behavior.py: Extracts behavioral information (e.g., reaction time, accuracy) for each session across regions.

-find_impulsivity.py: Computes the anticipatory index (impulsivity measure) for each session.



Distance Analysis/
This folder focuses on trajectory-based neural metrics.

-rho_region.py: Calculates the total trajectory distance normalized by absolute displacement (ρ), as well as trajectory deviation.

-plot_rho.py: Visualizes ρ and trajectory deviation across cortical regions and correlates them with behavior.

-plot_trajectory.py: Plots neural trajectories (left vs. right first movements) for a selected session.



Spiking Metrics/
This folder analyzes spiking variability and firing statistics.

-FR_CV_Fano.py: Computes Firing rate, Coefficient of variation (CV) of inter-spike intervals (ISI), Fano factor
for both task and spontaneous activity (SA) periods.

-Relative_Change.py: Measures the mean and median relative change in spiking activity between task and SA periods.




Hidden Markov Analysis/
This folder contains tools for uncovering latent brain states via HMM.

-HMM analysis for a single region (ephys).py: Estimates the optimal number of hidden states and computes metrics like state stickiness.

-Transition Matrix and State Vector Evolution.py: Calculates the transition matrix and tracks the evolution of state vectors over time.

-Plot Raster.ipynb: Jupyter Notebook for visualizing spike raster plots aligned to state transitions.



Fourier Analysis/
This folder contains scripts for frequency-domain analyses of population activity.

-Fourier_Transform_of_Inter_Task_Interval.py: Applies Fourier transform to inter-trial intervals (ITIs).

-Fourier_Transform_of_SA_and_Task.py: Performs Fourier analysis on task and spontaneous activity periods.

-Subfolders: Include scripts for visualizing power spectra and exploring their correlations with behavioral features at session and subject levels.




Bayesian_Methods/
This folder includes simulation-based approaches for timescale estimation using Bayesian inference.

-Implements both Bayesian estimation and trust-region methods, incorporating unbiased estimators for population mean autocorrelation functions.

-Simulates both single- and multi-timescale decay processes for synthetic neural activity data.









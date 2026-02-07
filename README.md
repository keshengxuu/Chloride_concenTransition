## Chloride_concenTransition model

The MATLAB implementation of the chloride concentration–dependent model of seizure transitions was derived and modified from the model proposed by Jyun-you Liou et al. (eLife, 2020, 9:e50927). In addition, all simulation results were independently reproduced using Python code.

The neural field model was implemented in both MATLAB and Python on a two-dimensional $N \times N$ nodes ($N=100$), masked by a circular domain of normalized radius 0.5, with activity outside the domain set to zero. Synaptic coupling was realized by convolving the firing rate with a Gaussian kernel using two-dimensional convolution with zero padding, followed by centering and cropping to the original grid size. The system was integrated using the exponential Euler scheme over 85 s with a time step $dt = 0.001$ s. The system was initialized at its equilibrium values: $V = -58$ mV, $\text{Cl}_{in} = 6$ mM,$\phi = -45$ and $g_K = 0$ nS.




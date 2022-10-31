"""
utility functions
"""

# imports
import numpy as np

def sim_ou_process(n_seconds, fs, tau, mu=100., sigma=10.):
    ''' 
    Simulate an Ornstein-Uhlenbeck process.
    
    
    Parameters
    ----------
    n_seconds : float
        Simulation time (s)
    fs : float
        Sampling rate (Hz)
    tau : float
        Timescale of signal (s)
    mu : float, optional, default: 100.
        Mean of signal
    sigma : float, optional, default: 10.
        Standard deviation signal

    Returns
    signal : 1d array
        Simulated Ornstein-Uhlenbeck process
    time : 1d array
        time vector for signal

    References
    ----------
    https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
    
    '''

    # initialize signal and set first value equal to the mean
    signal = np.zeros(int(np.ceil(n_seconds * fs)))
    signal[0] = mu
    
    # define constants in OU equation (to speed computation) 
    dt = 1 / fs
    sqrtdt = np.sqrt(dt)
    rand = np.random.randn(len(signal))
    
    # simulate OU
    for ii in range(len(signal)-1):
        signal[ii + 1] = signal[ii] + \
                        dt * (-(signal[ii] - mu) / tau) + \
                        sigma * np.sqrt(2/tau) * sqrtdt * rand[ii]
    
    # define time vector
    time = np.linspace(0, n_seconds, len(signal))
    
    return signal, time


def convolve_psps(spikes, fs, tau_r=0., tau_d=0.01, t_ker=None):
    """Adapted from neurodsp.sim.aperiodic.sim_synaptic_current
    
    Convolve spike train and synaptic kernel.

    Parameters
    ----------
    spikes : 1D array, int 
        spike train 
    tau_r : float, optional, default: 0.
        Rise time of synaptic kernel, in seconds.
    tau_d : float, optional, default: 0.01
        Decay time of synaptic kernel, in seconds.
    t_ker : float, optional
        Length of time of the simulated synaptic kernel, in seconds.

    Returns
    -------
    sig : 1d array
        Simulated synaptic current.
    time : 1d array
        associated time-vector (sig is trimmed  during convolution).

    """
    from neurodsp.sim.transients import sim_synaptic_kernel
    from neurodsp.utils import create_times
    
    # If not provided, compute t_ker as a function of decay time constant
    if t_ker is None:
        t_ker = 5. * tau_d

    # Simulate
    ker = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    sig = np.convolve(spikes, ker, 'valid')

    return sig

def spiketimes_to_spiketrains(spike_times, fs, t_stop, return_matrix=False):
    """
    convert list of spike times (or list of lists) to Neo SpikeTrain object
    and binaraized spike matrix.

    Parameters
    ----------
    spike_times : float, list
        list of spike times (or list of lists).
    fs : float
        sampling frequency.
    t_stop : float
        stop time .
    return_matrix : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    spike_trains : obj
        Neo SpikeTrain object.
    spikes : int, array
        binarized spike matrix.        

    """
    # imports
    from elephant.conversion import binarize
    from neo.core import SpikeTrain
    import quantities as pq
    
    # initialize
    spike_trains = []
    n_units = len(spike_times)
    if return_matrix:
        temp = binarize(SpikeTrain(spike_times[0]*pq.s, t_stop=t_stop), 
                        sampling_rate=fs*pq.Hz)
        spikes = np.zeros([n_units,len(temp)])

    # convert spike times to spike train
    for i_unit in range(n_units):
        if np.ndim(spike_times[i_unit]) == 0: continue
        if len(spike_times[i_unit]) == 0: continue
        sp = SpikeTrain(spike_times[i_unit]*pq.s, t_stop=t_stop)
        spike_trains.append(sp)
        
        # generate binaraized spike matrix
        if return_matrix:
            spikes[i_unit] = binarize(sp, sampling_rate=fs*pq.Hz)

    # return
    if return_matrix:
        return spike_trains, spikes
    else:
        return spike_trains



def spikes_to_lfp(spikes, fs):
    """
    simulate an LFP signal from population spiking activity. Spikes are 
    convolved with a kernel representing an excitatory post-synaptic potential.

    Parameters
    ----------
    spikes : int
        matrix of 0's and 1's, where each row is a neuron (or unit) and each 
        column is a time bin.
    fs : float
        sampling frequency.

    Returns
    -------
    lfp : float, 1D array
        simulated LFP signal.

    """

    # compute population rate
    pop_spikes = np.sum(spikes, axis=0)

    # convolve spikes with synaptic kernel
    lfp = convolve_psps(pop_spikes, fs)
    
    return lfp


def comp_exp(lfp, fs, peak_width_limits=[2,20], ap_mode='knee',
             f_range=[2,200], n_jobs=None):
    """
    compute aperiodic exponet of (LFP) signal.

    Parameters
    ----------
    lfp : float
        signal.
    fs : float
        sampling frequnecy.
    peak_width_limits : float, 1x2 array or list, optional
        peak width limits for spectral parameterization (Hz). 
        The default is [2,20].
    ap_mode : str, optional
        aperiodic mode for spectral parameterization ('fixed' pr 'knee').
        The default is 'knee'.
    f_range : float, 1x2 array or list, optional
        frequency range for spectral parameterization. The default is [2,200].
    n_jobs : int, optional
        number of jobs for parallel processing. The default is None.

    Returns
    -------
    exp : float
        aperiodic exponent.

    """
    # imports
    from neurodsp.spectral import compute_spectrum
    from fooof import FOOOF, FOOOFGroup
    
    # compute psd
    freq, spectra = compute_spectrum(lfp, fs)

    # initialize
    if np.ndim(spectra) == 1:
        sp = FOOOF(peak_width_limits=peak_width_limits, aperiodic_mode=ap_mode)
    elif np.ndim(spectra) > 1:
        sp = FOOOFGroup(peak_width_limits=peak_width_limits, 
                        aperiodic_mode=ap_mode)
    else:
        print("Check dimensions of lfp argument")
    
    # parameterize
    if n_jobs is None:
        sp.fit(freq, spectra, freq_range=f_range)
    else:
        sp.fit(freq, spectra, freq_range=f_range, n_jobs=n_jobs)
        
    # get exponent
    exp = sp.get_params('aperiodic', 'exponent')
    
    return exp

def comp_spike_cov(spikes):
    """
    compute coefficient of variation of spike train

    Parameters
    ----------
    spikes : int, array
        spiking activity (array of 0's and 1's).

    Returns
    -------
    cov : float
        coefficient of variation.

    """
    isi = np.diff(spikes)
    cov = np.std(isi) / np.mean(isi)
    
    return cov


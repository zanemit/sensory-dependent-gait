import numpy as np
import pandas as pd

def derivative(x):
    """
    takes the derivative of an n-dimensional array x
    where for n > 1, function is applied along columns (axis = 0)
    PARAMETERS:
        x (n-dim numpy array) : array to be differentiated
    RETURNS:
        (n-dim numpy array) : derivative of the input array with an initial zero
    """
    if len(x.shape) == 1:
        return np.concatenate([[0], np.diff(x)])
    else:
        d = np.zeros_like(x)
        d[1:, :] = np.diff(x, axis = 0)
        return d

def circular_mean(linear_phase): # linear_phase is a number in [0,1]
    linear_phase = linear_phase[~np.isnan(linear_phase)] # drop nans
    radians = np.asarray(linear_phase) * 2 * np.pi
    mean_cos = np.mean(np.cos(radians))
    mean_sin = np.mean(np.sin(radians))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    if mean_rad < 0:
        mean_rad = np.radians(360 + np.degrees(mean_rad))
    mean_linear = mean_rad / (2*np.pi)
    return (mean_rad, mean_linear)

def circular_stats(linear_phase, stat = 'median'): # linear_phase is a number in [-0.5,0.5]
    import math    
    linear_phase = linear_phase[~np.isnan(linear_phase)] # drop nans
    radians = np.asarray(linear_phase) * 2 * np.pi
    cosines = np.cos(radians)
    sines = np.sin(radians)
    if stat == 'median':
        computed_phase = math.atan2(np.median(sines),np.median(cosines))
    elif stat == 'mean':
        computed_phase = math.atan2(np.mean(sines),np.mean(cosines))
    elif stat == 'mode':
        from scipy import stats
        computed_phase = math.atan2(stats.mode(sines)[0], stats.mode(cosines)[0])
    else:
        raise ValueError("Invalid 'stat' parameter! Only 'mean' and 'median' allowed!")
    return computed_phase /(2*np.pi)

def cohen_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx = len(x)
    ny = len(y)
    dof = nx + ny -2
    pooled_std = np.sqrt(((nx-1)*np.var(x,ddof=1) + (ny-1) * np.var(y,ddof=1)) / dof)
    return (np.mean(x)-np.mean(y))/pooled_std
    
def temporal_angle(x, y):
    '''
    compute the angular changes in the position of a given bodypart over frames
    if x2 and y2 provided, computes angles relative to x axis across frames
    PARAMETERS:
        x (1d array) : array of cleaned DLC data, e.g. data[bp]['x']
        y (1d array) : array of cleaned DLC data, e.g. data[bp]['y']
    RETURNS
        (T x 1 numpy array) : angular velocity of a given bodypart
    '''
    bpX = np.diff(x) # change in X position
    bpY = np.diff(y) # change in Y position
    bpXn = bpX / np.linalg.norm((bpX, bpY), axis = 0)
    bpYn = bpY / np.linalg.norm((bpX, bpY), axis = 0)
    angles_and_signs = np.zeros_like(np.vstack((x,y)).T)
    angles_and_signs[:,0] = np.concatenate([[0], np.degrees(np.arccos(bpXn))])
    angles_and_signs[:,1] = np.concatenate([[0], np.sign(np.arccos(bpYn))])
    # replace those nans with zeros that arise due to unchanging coordinate (bpX and bpY == zero)
    angles_and_signs[1:,:][(bpX==0) & (bpY==0)] = 0
    return np.prod(angles_and_signs, axis = 1)

def get_vector(ax, ay, bx, by):
    if type(ax) == int or type(ax) == float:
        return np.vstack((bx-ax, by-ay)).T
    elif type(ax) == list:
        ax = np.asarray(ax); ay = np.asarray(ay)
        bx = np.asarray(bx); by = np.asarray(by)
    if type(ax) == np.ndarray or isinstance(ax, pd.DataFrame) or isinstance(ax, pd.Series):
        dimx = bx-ax
        dimy = by-ay
        return np.stack((dimx, dimy), axis = ax.ndim)
    else:
        raise ValueError("Data type not supported!")
    

def get_vector_length(ax, ay, bx, by):
    vec = get_vector(ax, ay, bx, by)
    if type(ax) == int or type(ax) == float:
        return np.sqrt(vec[0]**2 + vec[1]**2)
    elif type(ax) == list or type(ax) == np.ndarray or isinstance(ax, pd.DataFrame): 
        return np.sqrt(vec[...,0]**2 + vec[...,1]**2)
    else:
        raise ValueError("Data type not supported!")
        
    
def angle_with_x(x1, y1, x2, y2):
    bpX = x2-x1
    bpY = y2-y1
    bpX /= np.linalg.norm((bpX, bpY), axis = 0)
    bpY /= np.linalg.norm((bpX, bpY), axis = 0)
    angles_and_signs = np.zeros_like(np.vstack((x1,y1)).T)
    angles_and_signs[:,0] = np.concatenate([np.degrees(np.arccos(bpX))])
    angles_and_signs[:,1] = np.concatenate([np.sign(np.arccos(bpY))])
    return np.prod(angles_and_signs, axis = 1)
    
def angle_between_vectors_2d(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y):
    # works on (np.asarray([np.nan,0,0]),np.asarray([np.nan,0,0]),np.asarray([1,4,3]),np.asarray([1,0,0]),np.asarray([np.nan,0,0]),np.asarray([np.nan,0,0]),np.asarray([1,4,3]),np.asarray([1,3,4]))
    vector1 = get_vector(a1x, a1y, b1x, b1y)
    vector2 = get_vector(a2x, a2y, b2x, b2y)
    vector1norm = (vector1[~np.isnan(vector1).any(axis=1)].T / np.linalg.norm(vector1[~np.isnan(vector1).any(axis=1)], axis = 1)).T
    vector2norm = (vector2[~np.isnan(vector2).any(axis=1)].T / np.linalg.norm(vector2[~np.isnan(vector2).any(axis=1)], axis = 1)).T
    vector1norm = np.concatenate((vector1[np.isnan(vector1).any(axis=1)], vector1norm))
    vector2norm = np.concatenate((vector2[np.isnan(vector2).any(axis=1)], vector2norm))
    dot_product = np.diagonal(vector1norm @ vector2norm.T)
    # return np.degrees(np.clip(np.arccos(dot_product), -1, 1))
    return np.degrees(np.arccos(dot_product))

def angle_between_vectors_polygon(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y):
    """ 
    computes angle between two edges of a polygon if vertices are numbered counterclockwise
    """
    vector1 = get_vector(a1x, a1y, b1x, b1y)
    vector2 = get_vector(a2x, a2y, b2x, b2y)
    cross_product = np.cross(vector1, vector2) # radians
    cross_product_sign = np.sign(cross_product) # positive = left turn (angle < 180 deg), negative = right turn (angle > 180 deg)
    dot_product = angle_between_vectors_2d(a1x, a1y, b1x, b1y, a2x, a2y, b2x, b2y) # radians
    
    # dot product is computed by shifting vector 2 so that its origin aligns with that of vector 1
    # to get the angle between vectors without shifting vec 2 (end of vec1 and origin of vec2 aligned)
    angle = 180 - dot_product
    
    # the following applies only if points are ordered counterclockwise
    if len(angle) > 1:
        for i in range(len(angle)):
            if cross_product_sign[i] < 0:
                angle[i] = 360 - angle[i] # obtuse angle!
    else:
        angle = 360 - angle # obtuse angle!

    return angle

def hpd_circular(trace, mass_frac, low = -np.pi, high = np.pi) :
    """
    Based on: http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
        defined for boundaries at which sin(x) = 0, e.g. [-pi,pi] or [0,2*pi]
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
    low : the lower boundary of desired output range
    high : the upper boundary of desired output range
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # compute and store supplied values, sines, cosines in an array
    trace_arr = np.empty((n,8))
    trace_arr[:] = np.nan
    trace_arr[:,0] = trace
    trace_arr[:,1] = np.sin(trace)
    trace_arr[:,2] = np.cos(trace)
    trace_arr[:,3] = np.arctan2(trace_arr[:,1], trace_arr[:,2]) # value definitely in [-pi,pi] range
    
    # sort data by quadrant
    trace_arr[:,4][np.where((trace_arr[:,2]>0)&(trace_arr[:,1]>=0))[0]] = 1
    trace_arr[:,4][np.where((trace_arr[:,2]<=0)&(trace_arr[:,1]>0))[0]] = 2
    trace_arr[:,4][np.where((trace_arr[:,2]<0)&(trace_arr[:,1]<=0))[0]] = 3
    trace_arr[:,4][np.where((trace_arr[:,2]>=0)&(trace_arr[:,1]<0))[0]] = 4
    
    # order data anti-clockwise, store in trace[:,4]
    sorted_count = 0
    for q in (np.arange(4)+1)[::-1]:
        quadrant_row_ids = trace_arr[:,4] == q
        if q == 4 or q == 1: # anticlockwise sine is decreasing, so [::-1] for sorting
            sorted_quadrant = np.argsort(np.argsort(-trace_arr[quadrant_row_ids,1]))
        else:
            sorted_quadrant = np.argsort(np.argsort(trace_arr[quadrant_row_ids,1]))
        sorted_quadrant = sorted_quadrant + sorted_count # shift sorted values (rather than start from zero again) 
        trace_arr[quadrant_row_ids,5] = sorted_quadrant
        
        sorted_count = sorted_count + len(sorted_quadrant) # update sorted count
    
    # find interval widths, sotre in trace_arr[:,7]
    for start in np.arange(1, n+1):
        trace_arr[:,6] = (trace_arr[:,5] + start) % n # shift sorted order values around the circle clockwise
        min_point = trace_arr[trace_arr[:,6] == 0]
        max_point = trace_arr[trace_arr[:,6] == n_samples]
        
        # compute the arc length encompassing 95% of the data
        if min_point[:,4] == 3: # quadrant 3
            if max_point[:,4] == 3:
                if np.sin(max_point[:,3]) >= np.sin(min_point[:,3]):
                    trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
                else:
                    trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 4 or max_point[:,4] == 2 or max_point[:,4] == 1:
                trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            else:
                print(f"Unexpected value for start {start}!")
        
        elif min_point[:,4] == 2: # quadrant 2
            if max_point[:,4] == 2:
                if np.sin(max_point[:,3]) >= np.sin(min_point[:,3]):
                    trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
                else:
                    trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 4 or max_point[:,4] == 3 or max_point[:,4] == 1:
                trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
            else:
                print(f"Unexpected value for start {start}!")   
                
        elif min_point[:,4] == 1: # quadrant 1
            if max_point[:,4] == 1: 
                if np.sin(max_point[:,3]) <= np.sin(min_point[:,3]): 
                    trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
                else:
                    trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 2:
                trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 3 or max_point[:,4] == 4:
                trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
            else:
                print(f"Unexpected value for start {start}!")  
                
        elif min_point[:,4] == 4: # quadrant 4
            if max_point[:,4] == 4: 
                if np.sin(max_point[:,3]) <= np.sin(min_point[:,3]): 
                    trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
                else:
                    trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 3:
                trace_arr[trace_arr[:,6]==0,7] = np.abs(max_point[:,3]-min_point[:,3])
            elif max_point[:,4] == 1 or max_point[:,4] == 2:
                trace_arr[trace_arr[:,6]==0,7] = 2*np.pi - np.abs(max_point[:,3]-min_point[:,3])
            else:
                print(f"Unexpected value for start {start}!")
            
    upper_hpd_bound = trace_arr[np.argmin(trace_arr[:,7] ), 3]   
    upper_counter = trace_arr[np.argmin(trace_arr[:,7] ), 5] 
    # trace_arr[:,6] = (trace_arr[:,5] + counter) %n
    # lower_hpd_bound = trace_arr[trace_arr[:,6] == n_samples, 3]  [0]   
    lower_counter= (upper_counter + n_samples) %n
    lower_hpd_bound = trace_arr[trace_arr[:,5] == lower_counter, 3]  [0] 
    
    if upper_hpd_bound < lower_hpd_bound : # possible when lHPD is in quadrant 2 and hHPD - in quadrant 3
        upper_hpd_bound = upper_hpd_bound + 2*np.pi
        
    # Check that the supplied output range is 2pi
    if (high-low) != 2*np.pi:
        raise ValueError("Unsuitable boundaries supplied")
    
    # Compute range shift
    shift = 0
    import scipy.stats
    if scipy.stats.circmean(trace, high = high, low = low) > upper_hpd_bound:
        shift = 2*np.pi
    if scipy.stats.circmean(trace, high = high, low = low) < lower_hpd_bound:
        shift = -2*np.pi
    
    # Return interval
    return np.array([lower_hpd_bound, upper_hpd_bound]) + shift

def hpd_circular_simple(samples, mass_frac=0.95, low=-np.pi, high=np.pi):
    samples = np.asarray(samples)
    samples = np.mod(samples - low, 2*np.pi) + low  # map to [low, high]

    n = len(samples)
    n_samples = int(np.floor(mass_frac * n))

    # Sort the samples linearly
    sorted_samples = np.sort(samples)
    # For circular wraparound, append first samples shifted by 2pi
    sorted_aug = np.concatenate([sorted_samples, sorted_samples + 2*np.pi])

    # Sliding window of n_samples
    intervals = sorted_aug[n_samples:n + n_samples] - sorted_aug[:n]
    min_idx = np.argmin(intervals)

    lower = sorted_aug[min_idx]
    upper = sorted_aug[min_idx + n_samples - 1]

    # Wrap back to original interval
    lower = (lower - low) % (2*np.pi) + low
    upper = (upper - low) % (2*np.pi) + low

    return np.array([lower, upper])

def mean_resultant_length(phases):
    """
    computes the mean resultant length of 1D phase data
    
    mean resultant length is a measure of variability
    
    RETURNS : mean resultant length (float)

    """
    phases = phases[~np.isnan(phases)]

    sin = np.sum(np.sin(phases))
    cos = np.sum(np.cos(phases))
    resultant_length = np.sqrt(sin**2 + cos**2)/len(phases)
    return resultant_length
    
    
def hpd(trace, mass_frac) :
    """
    (c) http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html
    Returns highest probability density region given by
    a set of samples.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
    low : the lower boundary of desired output range
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

def von_mises_kde_exact(data, kappa, n_bins=100):
    from scipy.special import i0
    x = np.linspace(-np.pi, np.pi, n_bins)
    if len(data.shape) > 1:
        data = data.flatten()
    data = np.asarray(data[~np.isnan(data)])
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=x)
    return x, kde

def circular_optimal_transport(sample1, sample2, dataType = 'radians'):
    """
    inspired by https://stochastik.math.uni-goettingen.de/cot/
    """
    sample1 = np.asarray(sample1)[~np.isnan(np.asarray(sample1))]
    sample2 = np.asarray(sample2)[~np.isnan(np.asarray(sample2))]
    if dataType == "unit" :
        if not (np.min(np.concatenate((sample1,sample2))) > 0 and np.max(np.concatenate((sample1,sample2))) < 1):
            sample1 = (sample1-sample1.min())/(sample1.max()-sample1.min())
            sample2 = (sample2-sample2.min())/(sample2.max()-sample2.min())          
    elif dataType == "radians" :
        sample1 = (sample1 % (2*np.pi))/(2*np.pi)
        sample2 = (sample2 % (2*np.pi))/(2*np.pi)
    elif dataType =="angles" :
        sample1 = (sample1 % 360)/360
        sample2 = (sample2 % 360)/360
    else:
        raise ValueError("dataType must be 'radians', 'angles', or 'unit'! ")
  
    orderOfSamples = np.argsort(np.concatenate((sample1,sample2)))
    combinedSample = np.concatenate((np.concatenate((sample1,sample2))[orderOfSamples],[1]))
    k = len(combinedSample)-1
  
    diffCDFs = np.cumsum(np.concatenate((np.repeat(1/len(sample1),len(sample1)), np.repeat(-1/len(sample2),len(sample2))))[orderOfSamples])
    order_diffCDFs = np.argsort(diffCDFs)
    sorted_diffCDFs = diffCDFs[order_diffCDFs]
  
    weighting = combinedSample[1:] - combinedSample[:k]
    levMed = sorted_diffCDFs[np.where(np.cumsum(weighting[order_diffCDFs])>=0.5)[0][0]]
  
    return(np.sum(np.abs(diffCDFs - levMed)*weighting))

def WatsonU2_TwoTest(x, y):
    x = np.asarray(x[~np.isnan(x)])
    y = np.asarray(y[~np.isnan(y)])
    n1 = len(x)
    n2 = len(y)
    n = n1 + n2
    x = np.vstack((np.sort(x % (2 * np.pi)), np.repeat(1, n1))).T
    y = np.vstack((np.sort(y % (2 * np.pi)), np.repeat(2, n2))).T
    xx = np.vstack((x, y))
    rank = np.argsort(xx[:, 0])
    xx = np.hstack((xx[rank,  :], np.arange(n).reshape(-1,1)))
    a = np.arange(n)
    b = np.arange(n)
    for i in range(n):
       a[i] = np.sum(xx[:i, 1] == 1)
       b[i] = np.sum(xx[:i, 1] == 2)
    
    d = b/n2 - a/n1
    dbar = np.mean(d)
    u2 = (n1 * n2)/n**2 * np.sum((d - dbar)**2)
    
    if u2 > 0.385:
        p = "<0.001"
    elif u2 > 0.268:
        p = "<0.01"
    elif u2 > 0.187:
        p = "<0.05"
    elif u2 > 0.152:
        p = "<0.10"
    else:
        p = ">0.10"
    
    result = {"statistic": u2, "p": p, "n1": n1, "n2": n2}
    return(result)
  
    

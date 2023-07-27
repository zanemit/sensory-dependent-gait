import numpy as np
import pandas as pd
import copy
from scipy.signal import butter, filtfilt, medfilt  , lfilter
import warnings

def butter_filter(inputlist, filter_order = 2, filter_freq = 5, sampling_freq = 60):
    """
    applies a Butterworth filter
    PARAMETERS:
        inputlist (1d numpy array) : time-series to be filtered 
        filter_order (int) : filter order
        filter_freq (int) : low-pass filter frequency
        sampling_freq (int) : sampling frequency of the input data
        ignore_nan (bool) : CAREFUL! this removes nans even if they are in the middle, could shift time series!
    RETURNS:
        (1d numpy array) : filtered array
    """
    inputlist = np.asarray(inputlist)
    outputlist = np.zeros_like(inputlist)
    outputlist[:] = np.nan
    
    b,a = butter(filter_order, filter_freq, btype = 'low', fs = sampling_freq) # define the filter
    
    outputlist[~np.isnan(inputlist)] = filtfilt(b,a, inputlist[~np.isnan(inputlist)], axis =0) 
    return outputlist  

def butter_bandpass(lowcut, highcut, sampling_freq, filter_order=2):
    nyq = 0.5 * sampling_freq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(filter_order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(inputlist, lowcut, highcut, sampling_freq, filter_order=2):
    b, a = butter_bandpass(lowcut, highcut, sampling_freq, filter_order = filter_order)
    y = lfilter(b, a, inputlist)
    return y

def median_filter_1d(x, kernel=11):
    '''
    kernel must be odd
    if kernel = 11, there will be 11//2=5 pads,
    i.e. the 6th value will be repeated five more times
    PARAMETERS:
        x (1-dimensional array)
    RETURNS:
        (1-dimensional array) : filtered input array
    '''
    pad = kernel//2
    x_pad = np.pad(x, pad, "edge")
    return medfilt(x_pad, kernel_size=kernel)[pad:-pad]    

def find_ids_of_common_entries(search_array, master_array):
    """
    finds the ids where entries in array1 match those in array2
    PARAMETERS:
        array1 (1d numpy array) : array with entries of interest 
        array2 (1d numpy array) : array with IDs of interest (master)
    RETURNS:
        (numpy array) : ids of interest
    """
    matching_entries = np.intersect1d(search_array, master_array)
    return np.searchsorted(master_array, matching_entries)

def get_num_subplots(num_plots):
    """
    finds the optimal arrangement of subplots given the total number of subplots
    PARAMETERS:
        num_plots (int) : the total number of subplots
    RETURNS:
        (int) : number of rows
        (int) : number of columns
    """
    nrows = ncols = int(np.sqrt(num_plots))
    remainder = num_plots - (nrows * ncols)
    if (remainder > 0) & (remainder <= ncols):
        nrows += 1
    elif (remainder > ncols):
        nrows += 1; ncols += 1
    else:
        pass
    return nrows, ncols

def get_hist_probs_from_array(arr, num_bins):
    """
    returns histogram edge values
    and normalised histogram y values
    """
    h, e = np.histogram(arr, num_bins) # histogram and edge values
    p = h / arr.shape[0]
    return e,p

def compute_forelimb_phase(ref_phases, nonref_phases):
    phases = nonref_phases-ref_phases
    phases[phases<-0.5] = phases[phases<-0.5]+1
    phases[phases>0.5] = phases[phases>0.5]-1
    return phases

def support_intersection(p,q):
    sup_int = (list(filter(
        lambda x: (x[0]!=0) & (x[1]!=0), zip(p,q))))
    return sup_int

def get_probs_tuplelist(list_of_tuples):
    p = np.array([x[0] for x in list_of_tuples])
    q = np.array([x[1] for x in list_of_tuples])
    return p,q

def alternating(l):
    return all(cmp(a, b)*cmp(b, c) == -1 for a, b, c in zip(l, l[1:], l[2:]))

def cmp(a, b):
    return int(a > b) - int(a < b)

def get_range(x):
    '''
    PARAMETERS:
        x (n-dim array)
    RETURNS:
        an array of min, max values in every column of x
        obtain (min, max) of 0th column by calling tuple(output[0])
    '''
    minimum = np.nanmin(np.asarray(x))
    maximum = np.nanmax(np.asarray(x))
    return np.array([minimum, maximum]).T, maximum-minimum

def flatten_list(x):
    """
    flattens a list of arrays
    PARAMETERS:
        x (list of arrays)
    RETURNS:
        (flat list)
    """
    return [item for arr in x for item in arr]

def grab_data(arr, start, end):
    """
    takes a 1D input array and subsets it based on the supplied start and end values
    if either start or end is out of range, this function adds nan padding
    """
    if start >= 0 and end < arr.shape[0]:
        return np.asarray(arr[start:end])
    elif start < 0 and end > arr.shape[0]-1:
        return np.concatenate((np.repeat(np.nan, start*-1), np.asarray(arr), np.repeat(np.nan, end-arr.shape[0])))
    elif start < 0:
        return np.concatenate((np.repeat(np.nan, start*-1), np.asarray(arr[:end])))
    elif end > arr.shape[0]-1:
        return np.concatenate((np.asarray(arr[start:]), np.repeat(np.nan, end-arr.shape[0])))
    else:
        raise ValueError("Unknown problem with the array")
        
def preprocess_dlc_data(filepath, likelihood_thr = 0.95):
    """
    converts DLC output data into dict = {bp1,bp2,bp3,...}
    where each bp is a dataframe: | time | x | y |
    also interpolates xy values with unacceptable likelihoods
    
    PARAMETERS
    -------
    filepath (str) : full path to the DLC multilevel output file
    likelihood_thr (int) : minimum value of acceptable likelihoods
    
    RETURNS
    -------
    trackings (dict) : xy coords referenced by bodypart
    bodyparts (numpy array) : labeled bodyparts
    
    """
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    data = pd.read_hdf(filepath)
    dlc_ext = data.columns.get_level_values(0)[0]
    bodyparts = np.unique(data.columns.get_level_values(1))
    tracking = data.unstack()
    trackings = {}; trackings_interpol = {}; likelihoods = {}
    for bp in bodyparts:
        tr = {c:tracking.loc[dlc_ext, bp, c].values for c in ['x', 'y', 'likelihood']} #dict with three items
        trackings[bp] = pd.DataFrame(tr).drop('likelihood', axis=1) #for bp, store x,y,lkhd
        # trackings[bp] = pd.DataFrame(tr)
        trackings_interpol[bp] = copy.deepcopy(trackings[bp])
        likelihoods[bp] = pd.DataFrame(tr['likelihood'])
        if 0 <= likelihood_thr <= 1:
            trackings_interpol[bp][np.asarray(likelihoods[bp]<likelihood_thr).reshape(1,-1)[0]] = np.nan
            trackings[bp] = trackings_interpol[bp].interpolate(axis=0)
            # trackings_interpol[bp]['x'] = trackings[bp].interpolate(axis=0)
            # trackings_interpol[bp][np.isnan(trackings_interpol[bp])] = trackings[bp][np.isnan(trackings_interpol[bp])]
        else:
            raise ValueError('Likelihood threshold must be between 0 and 1')
    return trackings, bodyparts

def downsample_data(array, target_length):
    """
    downsamples 1d array to a target length
    by averaging over some numbers
    """
    # window = 50
    new_array = np.empty(target_length)
    num_to_average_over_array = np.empty(target_length)
    starting_length = len(array)
    lowest_num_to_average_over = int(starting_length / target_length)
    num_to_average_over_array[:] = lowest_num_to_average_over
    remainder = starting_length % target_length
    ids_to_increment = np.linspace(0, target_length-1, remainder).astype(int)
    num_to_average_over_array[ids_to_increment] = lowest_num_to_average_over + 1
    # num_to_average_over_array += window # SMOOTHING -  not much difference
    i = 0
    for i_n, num in enumerate(num_to_average_over_array.astype(int)):
        new_array[i_n] = array[i:(i+num)].mean()
        i += num  #-window
    return new_array

def remove_outliers(array):
    """
    removes outliers from a 1d array
    where an outlier is anything outside [Q1-1.5*IQR, Q3+1.5*IQR]
    """
    q1 = np.nanpercentile(array, 25, interpolation = 'midpoint')
    q3 = np.nanpercentile(array, 75, interpolation = 'midpoint')
    iqr = q3-q1
    outlier_thresh_low = q1 - (1.5*iqr)
    outlier_thresh_high = q3 + (1.5*iqr)
    return array[(array>=outlier_thresh_low) & (array<=outlier_thresh_high)]

def get_weight_age(metadata_df, mouseID, expDate):
    """
    returns the weight and age of mouse when supplied metadataPyRAT_processed, 
    mouseID, and expDate
    """
    metadata_sub = metadata_df.loc[metadata_df['mouseID']['mouseID'] == mouseID, :].iloc[:,2:]
    metadata_sub = metadata_sub.loc[:, ~np.isnan(metadata_sub.iloc[0,:])]
    dates = np.unique(metadata_sub.columns.get_level_values(1)).astype(int)
    if int(expDate) in dates:
        weight = float(metadata_sub.loc[:,('Weight', str(expDate))])
        age = int(metadata_sub.loc[:,('Age', str(expDate))])
    else:
        from datetime import datetime
        expDate_rec = f'20{str(expDate)[:2]}-{str(expDate)[2:4]}-{str(expDate)[4:]}'
        expDate_dt = datetime.strptime(expDate_rec, "%Y-%m-%d")
        nextDate_id = np.where(dates > int(expDate))[0]
        if len(nextDate_id) > 0:
            nextDate = dates[nextDate_id[0]]
            previousDate = dates[nextDate_id[0]-1]
            
            nextDate_rec = f'20{str(nextDate)[:2]}-{str(nextDate)[2:4]}-{str(nextDate)[4:]}'
            previousDate_rec = f'20{str(previousDate)[:2]}-{str(previousDate)[2:4]}-{str(previousDate)[4:]}'
            
            nextDate_dt = datetime.strptime(nextDate_rec, "%Y-%m-%d")
            previousDate_dt = datetime.strptime(previousDate_rec, "%Y-%m-%d")
            
            day_frac_since_previous = (expDate_dt-previousDate_dt).days/(nextDate_dt-previousDate_dt).days
            
            nextWeight = float(metadata_sub.loc[:,('Weight', str(nextDate))])
            previousWeight = float(metadata_sub.loc[:,('Weight', str(previousDate))])
            weight = round(previousWeight + (day_frac_since_previous*(nextWeight-previousWeight)),1)
        
        else:
            previousDate = dates[-1]
            previousDate_rec = f'20{str(previousDate)[:2]}-{str(previousDate)[2:4]}-{str(previousDate)[4:]}'
            previousDate_dt = datetime.strptime(previousDate_rec, "%Y-%m-%d")
            weight = float(metadata_sub.loc[:,('Weight', str(dates[-1]))])
            
        previousAge = int(metadata_sub.loc[:,('Age', str(previousDate))])
        age = previousAge + (expDate_dt-previousDate_dt).days
        
    return weight, age

def are_arrays_alternating(arr1, arr2):
    """
    arr1 (1d array) : stance onsets - "peaks"
    arr2 (1d array) : swing onsets - "troughs"
    returns arrays of equal length or with one extra trough
    trough[0] < peak [0]
    
    """
    # want the steps to start with the swing phase
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if arr1[0] < arr2[0] and len(arr1) > 1:
        print("Removing first peak...")
        arr1 = arr1[np.where(arr1 > arr2[0])[0][0]:] #find where arr1>arr2[0] and keep those
    arr1 = np.vstack((arr1, np.zeros_like(arr1)))
    arr2 = np.vstack((arr2, np.zeros_like(arr2)+1))
    arrs = (arr1, arr2)
    arr = np.concatenate(arrs, axis=1).T
    arr1 = arr1.T
    arr2 = arr2.T
    sorted_arr = arr[np.argsort(arr[:,0])]
    round_num = 0
    while not (alternating(sorted_arr[:,1]) and (arr1.shape[0] == arr2.shape[0] or arr1.shape[0] + 1 == arr2.shape[0])):
        # if alternating(sorted_arr[:,1]): # sorted_arr ends with an arr2 element (it also starts with arr2)
        #     print("Removing last trough...")    
        #     arr2 = arr2[:np.where(arr2 > arr1[-1])[0][0]]
        if alternating(sorted_arr[:-1,1]): #the last two elements are the same
            print("Removing last (repeated) peak/trough...")
            sorted_arr = sorted_arr[:-1]
            arr1 = sorted_arr[1::2, 0] #peaks
            arr2 = sorted_arr[::2, 0] #trough
        else:
            diff = np.diff(sorted_arr[:,1])
            nonalternations = np.where(diff == 0)[0]
            print(f"Found {nonalternations.shape[0]} non-alternations...")
            for n in nonalternations[::-1]:
                if sorted_arr[n,1] == 0: #peak
                    peaks_or_troughs = sorted_arr[np.where(sorted_arr[:,1] == 0)[0], 0]
                elif sorted_arr[n,1] == 1: #trough
                    peaks_or_troughs = sorted_arr[np.where(sorted_arr[:,1] == 1)[0], 0]
                ave = peaks_or_troughs.mean()
                try:
                    closest = np.argmin(np.abs(sorted_arr[n,:]-ave), np.abs(sorted_arr[n+1,:]-ave))
                except:
                    # print('here')
                    return False, False
                sorted_arr = np.delete(sorted_arr, n+closest, axis = 0)
            arr1 = sorted_arr[1::2, 0] #peaks
            arr2 = sorted_arr[::2, 0] #troughs 
        round_num += 1
        if round_num > 5:
            print("Could not correct in five rounds!")
            return False, False
    if np.ndim(arr1) > 1 and np.ndim(arr2) > 1:
        return arr1[:,0], arr2[:,0]   
    else:
        return arr1, arr2 

def split_by_percentile(data, group_num):
    """
    splits data from one dataframe column into equally sized groups 
    
    PARAMETERS:
        data (1d array, list, df col) : 1d data
        group_num (int) : number of groups to split data into
    
    """
    tile_size = 100/group_num
    
    data = np.asarray(data)
    data_split = np.empty(group_num+1)
    for i in range(group_num+1):
        pc = tile_size*i
        data_split[i] = np.nanpercentile(data, pc)
    return data_split

def invert_nested_dict(d):
    d_inv = {}
    for key, val in d.items():
        for subkey, subval in val.items():
            for subsubkey, subsubval in subval.items():
                if subsubkey in d_inv.keys():
                    if subkey in d_inv[subsubkey].keys():
                        d_inv[subsubkey][subkey][key] = d[key][subkey][subsubkey]
                    else:
                        d_inv[subsubkey][subkey] = {}
                        d_inv[subsubkey][subkey][key] = d[key][subkey][subsubkey]
                else:
                    d_inv[subsubkey] = {}
                    d_inv[subsubkey][subkey] = {}
                    d_inv[subsubkey][subkey][key] = d[key][subkey][subsubkey]
    return d_inv

def populate_nested_dict(targetDict, dataToAdd, metadataLVL):
    """
    populates a dictionary with 2, 3, 4, or 5 nested levels
    
    PARAMETERS
    -----------
    targetDict (dict) : dictionary to add data to (can be empty {})
    dataToAdd (anything) : array/int/float to be added at the deepest level
    metadataLVL (list) : list of items defining nested level keys
    
    RETURNS
    -----------
    targetDict (dict) : updated dictionary
    
    """
    # populating the nested optoTrigDict
    if len(metadataLVL) == 2:
        a, b = metadataLVL
    elif len(metadataLVL) == 3:
        a, b, c = metadataLVL 
    elif len(metadataLVL) == 4:
        a, b, c, d = metadataLVL 
    elif len(metadataLVL) == 5:
        a, b, c, d, e = metadataLVL 
    else: 
        raise ValueError ("metadataLVL length does not match the allowed ones!")
    if len(metadataLVL) > 1:
        if a in targetDict.keys():
            if len(metadataLVL) == 2:
                targetDict[a][b] = dataToAdd
            else: # len(metadataLVL) > 2
                if b in targetDict[a].keys():
                    if len(metadataLVL) == 3:
                        targetDict[a][b][c] = dataToAdd
                    else: # len(metadataLVL) > 3
                        if c in targetDict[a][b].keys():
                            if len(metadataLVL) == 4:
                                targetDict[a][b][c][d] = dataToAdd
                            else: # len(metadataLVL) > 4
                                if d in targetDict[a][b][c][d].keys():
                                    if len(metadataLVL) == 5:
                                        targetDict[a][b][c][d][e] = dataToAdd
                                    else: # len(metadataLVL) > 5
                                        raise ValueError("More than 5 metadata levels provided!")
                                else: # len(metadataLVL) > 4 and d not in keys
                                    if len(metadataLVL) == 5:
                                        targetDict[a][b][c][d] = {}
                                        targetDict[a][b][c][d][e] = dataToAdd
                                    else: # len(metadataLVL) > 5 and d not in keys
                                        raise ValueError("More than 5 metadata levels provided!")
                        else: # len(metadata) > 3 and c not in keys
                            targetDict[a][b][c] = {}
                            if len(metadataLVL) == 4:
                                targetDict[a][b][c][d] = dataToAdd
                            elif len(metadataLVL) == 5:
                                targetDict[a][b][c][d] = {}
                                targetDict[a][b][c][d][e] = dataToAdd
                            else:
                                raise ValueError("More than 5 metadata levels provided!")
                else: # len(metadata) > 2 and b not in keys
                    targetDict[a][b] = {}
                    if len(metadataLVL) == 3:
                        targetDict[a][b][c] = dataToAdd
                    elif len(metadataLVL) == 4:
                        targetDict[a][b][c] = {}
                        targetDict[a][b][c][d] = dataToAdd
                    elif len(metadataLVL) == 5:
                        targetDict[a][b][c] = {}
                        targetDict[a][b][c][d] = {}
                        targetDict[a][b][c][d][e] = dataToAdd
                    else:
                        raise ValueError("More than 5 metadata levels provided!")
        else:
            targetDict[a] = {}
            if len(metadataLVL) == 2:
                targetDict[a][b] = dataToAdd
            elif len(metadataLVL) == 3:
                targetDict[a][b] = {}
                targetDict[a][b][c] = dataToAdd
            elif len(metadataLVL) == 4:
                targetDict[a][b] = {}
                targetDict[a][b][c] = {}
                targetDict[a][b][c][d] = dataToAdd
            elif len(metadataLVL) == 5:
                targetDict[a][b] = {}
                targetDict[a][b][c] = {}
                targetDict[a][b][c][d] = {}
                targetDict[a][b][c][d][e] = dataToAdd
            else:
                raise ValueError("More than 5 metadata levels provided!")
    else: # len(metadataLVL) == 1
        targetDict[a] = dataToAdd
    
    return targetDict
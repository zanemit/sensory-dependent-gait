# import sys
# sys.path.append(r'C:\Users\zanem\OneDrive\Documents\Python Scripts\Analysis')
# sys.path.append(r'C:\Users\MurrayLab\Documents\Python Scripts')
import numpy as np
import pandas as pd
import copy
import os
import cv2

def get_trigger_times(trigger_array, threshold = 3.3):
    """
    finds the rising/falling edge IDs in a TTL array
    
    PARAMETERS
    ----------
    trigger_array (1-d numpy array) : one column of the analog input array
    threshold (float, optional) : voltage value considered to switch signal on/off
    
    RETURNS
    ----------
    rising_edge_ids (1d numpy array) : sample IDs where voltage crosses threshold
    falling_edge_ids (1d numpy array) : sample IDs where voltage drops below threshold
    trigger_num (int) : number of triggers
    
    """
    high_state = (trigger_array > threshold).astype(int)
    difference = np.concatenate(([0],np.diff(high_state)))
    trigger_num = np.sum(difference == 1)
    rising_edge_ids = np.where(difference==1)[0]
    falling_edge_ids = np.where(difference==-1)[0]
    return rising_edge_ids, falling_edge_ids, trigger_num

def check_skipped_frames(camera_array, printTrue = False, camera_path = None):
    """
    checks if frameIDs in the csv file are a continuous sequence

    PARAMETERS
    ----------
    camera_array (1d pandas series) : series containing frameIDs from a given camera
    printTrue (bool, optional) : if true, a message declaring a good outcome will be printed
    camera_path (str, optional) : if not None, full path to camera csv file

    RETURNS
    -------
    (bool) : true if data passes the test

    """
    frame_ID_test = np.all(np.diff(camera_array) == 1)
    if frame_ID_test:
        if printTrue:
            print('Check passed: No frames have been skipped by camera hardware!')
        return True
    else:
        if camera_path != None:
            filename = os.path.split(camera_path)
            print(f'Check failed: Some frames have been missed by camera hardware in {filename}!')
        return False
    
def check_skipped_triggers(camera_array, trigger_array, printTrue = False, bin_path = None, threshold = 3.3):
    """
    checks if frameIDs in the csv file matches the generated camera triggers

    PARAMETERS
    ----------
    camera_array (1d pandas series) : series containing frameIDs from a given camera
    trigger_array (1d numpy array) : array containing voltage values from camera trigger channel
    video_cap (cv2.VideoCapture) : loaded video file 
    printTrue (bool, optional) : if true, a message declaring a good outcome will be printed
    bin_path (str, optional) : if not None, full path to camera csv file
    threshold (float, optional) : voltage value considered to switch signal on/off

    RETURNS
    -------
    (bool) : true if data passes the test

    """
    trigger_number = get_trigger_times(trigger_array, threshold)[2]
    if camera_array.shape[0] == trigger_number:
        if printTrue:
            print('Check passed: The number of triggers matches the number of frame IDs!')
        return True
    else:
        if bin_path != None:
            filename = os.path.split(bin_path)
            print(f'Check failed: There are {trigger_number} triggers and {camera_array.shape[0]} frame IDs in {filename} and the corresponding csv files.')
        return False
    
def check_dlc_file(camera_array, dlc_df, printTrue = False, dlc_path = None):
    """
    checks if all frames have been processed by DLC
    (should be fine, but the tracking itself could be poor!)

    PARAMETERS
    ----------
    camera_array (1d pandas series) : series containing frameIDs from a given camera
    dlc_df (pandas multilevel dataframe) : DLC output dataframe
    printTrue (bool, optional) : if true, a message declaring a good outcome will be printed
    dlc_path (str, optional) : if not None, full path to the DLC h5 file

    RETURNS
    -------
    (bool) : true if data passes the test

    """
    if camera_array.shape[0] == dlc_df.shape[0]:
        if printTrue:
            print('Check passed: The number of frame IDs matches the number processed frames!')
        return True
    else:
        if dlc_path != None:
            filename = os.path.split(dlc_path)
            print(f'Check failed: There are {camera_array.shape[0]} frame IDs and {dlc_df.shape[0]} tracked frames in {filename}!')
        return False


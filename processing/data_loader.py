import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from preprocessing.data_config import Config

def load_processed_data(outputDir = Config.paths["passiveOpto_output_folder"],  
                        dataToLoad = 'passiveOptoData', 
                        yyyymmdd = None,
                        appdx = "",
                        limb = None,
                        param = None):
    """
    loads processed datafiles based on their shorthand names
    outputDir (str, optional) : path to output folder
    yyyymmdd (str or None, optional) : date of experiment
    appdx (str, optional) : "" (empty) or "_incline" for head height and incline trials respectively
    limb (str or None, optional) : reference limb
    
    dataToLoad options :
        passiveOptoData (default)
        optoTrigDict
        metadataProcessed (forceplate)
        metadataPyRAT (passiveOpto, forceplate)
        forceplateData
        forceplateData_groupedby
        forceplateAngles
        forceplateHeadHW
        forceplateAngleParams
        meanParamDF (forceplate)
        weightCalibration
        bodyAngles (treadmill)
        locomParams (treadmill)
        dlcPostureX (forceplate)
        dlcPostureY (forceplate)
        dlcPosts (forceplate)
        locomFrameDict (passiveOpto)
        bodyAngles (passiveOpto)
        strideParams (passiveOpto)
        beltSpeedData (passiveOpto)
        limbX (passiveOpto)
        limbY (passiveOpto)
        arrayIdentifier (passiveOpto)
        limbX_strides (passiveOpto)
        limbY_strides (passiveOpto)
        limbX_speed (passiveOpto)
        limbX_bodyAngles (passiveOpto)
        supportFractions (passiveOpto/mtTreadmill)
        modePhases (passiveOpto/mtTreadmill)
        mtLimbData (mtTreadmill)
        mtOtherData (mtTreadmill)
        mouseSpeed (mtTreadmill)
        movementDict (mtTreadmill)
              

    """
    # load existing data
    outputDir = Path(outputDir).resolve()
    if outputDir.exists():
        if dataToLoad == 'passiveOptoData':
            print("Loading passive-opto DLC summary dataframe...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'passiveOptoData{appdx}' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_passiveOptoData{appdx}.csv')
            data = pd.read_csv(path, header = [0,1,2,3,4,5], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'optoTrigDict':
            print("Loading optotrig dictionary...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.pkl') and f'optoTrigDict{appdx}' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_optoTrigDict{appdx}.pkl')
            data = pickle.load(open(path, "rb" ))
            return data, yyyymmdd
        elif dataToLoad == 'metadataProcessed':
            print('Loading processed PyRAT metadata...')
            path = outputDir / ('metadataPyRAT_processed.csv')
            data = pd.read_csv(path, header=[0, 1], index_col=0)
            return data
        elif dataToLoad == 'metadataPyRAT':
            print('Loading PyRAT metadata...')
            path = outputDir / ('metadataPyRAT.csv')
            data = pd.read_csv(path)
            return data
        elif dataToLoad == 'forceplateData':
            print("Loading forceplate dataframe...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateData' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_forceplateData.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'forceplateAngles':
            print("Loading forceplate angles...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateAngles' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_forceplateAngles.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'forceplateHeadHW':
            print("Loading forceplate HW...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateHeadHW' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_forceplateHeadHW.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'forceplateData_groupedby':
            print("Loading forceplate HW...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateHeadHW' in f]
                yyyymmdd = input(fileDates)
            if appdx == "":
                params = np.unique([f.split("_")[-1][:-4] for f in os.listdir(outputDir) if 'forceplateData_groupedby' in f and yyyymmdd in f])
                appdx = input(params)
            appdx = "_"+ appdx
            path = outputDir / (yyyymmdd + f'_forceplateData_groupedby{appdx}.csv')
            data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'forceplateData_CoMgrouped':
            print("Loading forceplate CoM grouped...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateData_CoMgrouped' in f]
                yyyymmdd = input(fileDates)
            if appdx == "":
                params = np.unique([f.split("_")[-1][:-4] for f in os.listdir(outputDir) if 'forceplateData_CoMgrouped' in f and yyyymmdd in f])
                appdx = input(params)
            appdx = "_"+ appdx
            path = outputDir / (yyyymmdd + f'_forceplateData_CoMgrouped{appdx}.csv')
            data = pd.read_csv(path, header=[0, 1, 2,3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'forceplateAngleParams':
            print("Loading forceplate angle params...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateAngleParamsR' in f]
                yyyymmdd = input(fileDates)
            if appdx == "":
                params = np.unique([f.split("_")[-1][:-4] for f in os.listdir(outputDir) if 'forceplateAngleParamsR' in f and yyyymmdd in f])
                appdx = "_"+input(params)
            path = outputDir / (yyyymmdd + f'_forceplateAngleParamsR{appdx}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd
        elif dataToLoad == 'meanParamDF':
            if yyyymmdd == None:
                fileDates = list(np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'meanParamDF' in f]))
                yyyymmdd = input(np.unique(fileDates))
            if appdx == None:
                params = list(np.unique([f.split('_')[-1][:-4] for f in os.listdir(outputDir) if f.endswith('.csv') and 'meanParamDF' in f and yyyymmdd in f]))
                appdx = input(params)
            path = outputDir / (yyyymmdd + f'_meanParamDF{appdx}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd
        elif dataToLoad == 'weightCalibration':
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'forceplateData' in f]
                yyyymmdd = input(fileDates)
            if int(yyyymmdd[:4]) >= 2022:
                yyyymmddCalib = '2022-05-25'
            else:
                yyyymmddCalib = '2020-06-22'
            path = outputDir / (yyyymmddCalib + '_load_cell_calibration_summary.csv')
            data = pd.read_csv(path)
            data = data.set_index('Unnamed: 0')
            return data, yyyymmdd
        elif dataToLoad == 'bodyAngles':
            print("Loading body angles array...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'bodyAngles{appdx}' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_bodyAngles{appdx}.csv')
            if yyyymmdd == '2022-05-06':
                data = pd.read_csv(path, header = [0,1,2,3,4,5], index_col = 0)
            else:
                data = pd.read_csv(path, header = [0,1,2,3,4], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'locomParams':
            print("Loading locomotor param df...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'locomParams{appdx}' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_locomParams{appdx}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd   
        elif dataToLoad == 'dlcPostureX':
            print("Loading dlcPostureX data...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'dlcPostureX' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_dlcPostureX.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'dlcPostureY':
            print("Loading dlcPostureY data...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'dlcPostureY' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_dlcPostureY.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'dlcPosts':
            print("Loading dlcPosts data...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'dlcPosts' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_dlcPosts.csv')
            data = pd.read_csv(path, header=[0, 1, 2, 3, 4], index_col=0)
            return data, yyyymmdd
        elif dataToLoad == 'locomFrameDict':
            print("Loading locomotor frame dictionary...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.pkl') and f'locomFrameDict{appdx}' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_locomFrameDict{appdx}.pkl')
            data = pickle.load(open(path, "rb" ))
            return data, yyyymmdd
        elif dataToLoad == 'bodyAngles':
            print("Loading body angles array...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'bodyAngles{appdx}' in f]
                yyyymmdd = input(yyyymmdd)
            path = outputDir / (yyyymmdd + f'_bodyAngles{appdx}.csv')
            data = pd.read_csv(path, header = [0,1,2,3,4], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'limbX':
            print("Loading array of limb x coordinates...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX{appdx}.' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_limbX{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd
        elif dataToLoad == 'strideParams':
            print("Loading stride param df...")
            if yyyymmdd ==None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'strideParams{appdx}' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                print("Pick a reference limb!")
                limbs = np.unique([f[-7:-4] for f in os.listdir(outputDir) if f.endswith('.csv') and f'strideParams{appdx}' in f and yyyymmdd in f])
                limb = input(limbs)
            path = outputDir / (yyyymmdd + f'_strideParams{appdx}_{limb}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd, limb
        elif dataToLoad =='locomParamsAcrossMice':
            print("Loading locomotor params across mice df...")
            if yyyymmdd ==None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'locomParamsAcrossMice{appdx}' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_locomParamsAcrossMice{appdx}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd
        elif dataToLoad == 'beltSpeedData':
            print('Loading belt speed summary dataframe...')
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'beltSpeedData{appdx}' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_beltSpeedData{appdx}.csv')
            data = pd.read_csv(path, header = [0,1,2,3], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'limbX':
            print("Loading array of limb x coordinates...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX{appdx}.' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_limbX{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd
        elif dataToLoad == 'limbY':
            print("Loading array of limb y coordinates...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbY{appdx}.' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_limbY{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd
        elif dataToLoad == 'arrayIdentifier':
            print("Loading the identifier of the reformatted limb data array...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'arrayIdentifier{appdx}' in f])
                yyyymmdd  = input(fileDates)
            path = outputDir / (yyyymmdd  + f'_arrayIdentifier{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd 
        elif dataToLoad == "limbX_strides":
            print("Loading array of limb X coordinates per stride...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX_bodyAngles{appdx}' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                print("Pick a reference limb!")
                limbs = np.unique([f[-7:-4] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX_strides{appdx}' in f and yyyymmdd in f])
                limb = input(limbs)
            path = outputDir / (yyyymmdd + f'_limbX_strides{appdx}_{limb}.npy')
            data = np.load(path)
            return data, yyyymmdd, limb
        elif dataToLoad == "limbY_strides":
            print("Loading array of limb Y coordinates per stride...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbY_bodyAngles{appdx}' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                print("Pick a reference limb!")
                limbs = np.unique([f[-7:-4] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbY_strides{appdx}' in f and yyyymmdd in f])
                limb = input(limbs)
            path = outputDir / (yyyymmdd + f'_limbY_strides{appdx}_{limb}.npy')
            data = np.load(path)
            return data, yyyymmdd, limb
        elif dataToLoad == 'limbX_speed':
            print("Loading array of limb x coordinates...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX_speed{appdx}.' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_limbX_speed{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd
        elif dataToLoad == 'limbX_bodyAngles':
            print("Loading array of limb x coordinates...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.npy') and f'limbX_bodyAngles{appdx}.' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + f'_limbX_bodyAngles{appdx}.npy')
            data = np.load(path)
            return data, yyyymmdd
        elif dataToLoad == 'limbKinematicsDict':
            print("Loading array of limb kinematics data...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.pkl') and 'limbKinematicsDict' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                 print("Pick a reference limb!")
                 limbs = np.unique([f.split('_')[3] for f in os.listdir(outputDir) if f.endswith('.pkl') and 'limbKinematicsDict' in f and yyyymmdd in f])
                 limb = input(limbs)
            print("Pick a param!")
            params = np.unique([f.split('_')[2] for f in os.listdir(outputDir) if f.endswith('.pkl') and 'limbKinematicsDict' in f and yyyymmdd in f and limb in f])
            param = input(params)
            path = outputDir / (yyyymmdd + f'_limbKinematicsDict_{param}_{limb}{appdx}.pkl')
            data = pickle.load(open(path, "rb" ))
            return data, yyyymmdd, limb, param
        elif dataToLoad == 'supportFractions':
            print("Loading array of support fractions...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and f'supportFractions{appdx}' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                print("Pick a reference limb!")
                limbs = np.unique([f[-7:-4] for f in os.listdir(outputDir) if f.endswith('.csv') and f'supportFractions{appdx}' in f and yyyymmdd in f])
                limb = input(limbs)
            path = outputDir / (yyyymmdd + f'_supportFractions{appdx}_{limb}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd, limb
        elif dataToLoad == 'modePhases':
            print("Loading array of support fractions...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'modePhases' in f])
                yyyymmdd = input(fileDates)
            if limb == None:
                print("Pick a reference limb!")
                limbs = np.unique([f[-7:-4] for f in os.listdir(outputDir) if f.endswith('.csv') and f'modePhases' in f and yyyymmdd in f])
                limb = input(limbs)
            if param == None:
                print("Pick a param!")
                params = ['snoutBodyAngle', 'trialType', 'headLVL', 'speed', 'other']
                param = input(params)
                if param == 'other':
                    print('Enter the param!')
                    param = input()
            path = outputDir / (yyyymmdd + f'_modePhases_limbs_param_{param}{appdx}_{limb}.csv')
            data = pd.read_csv(path)
            return data, yyyymmdd, limb, param
        elif dataToLoad == 'mtLimbData':
            print("Loading motorised treadmill limb dataframe...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'motoTreadmillLimbData' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_motoTreadmillLimbData.csv')
            data = pd.read_csv(path, header = [0,1,2,3,4,5,6], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'mtOtherData':
            print("Loading motorised treadmill other bodypart dataframe...")
            if yyyymmdd == None:
                fileDates = np.unique([f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'motoTreadmillOtherData' in f])
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_motoTreadmillOtherData.csv')
            data = pd.read_csv(path, header = [0,1,2,3,4,5,6], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'mouseSpeed':
            print("Loading body angles...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.csv') and 'mouseSpeed' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_mouseSpeed.csv')
            data = pd.read_csv(path, header = [0,1,2,3,4], index_col = 0)
            return data, yyyymmdd
        elif dataToLoad == 'movementDict':
            print("Loading treadmill and locomotion onset/offset dictionary...")
            if yyyymmdd == None:
                fileDates = [f[:10] for f in os.listdir(outputDir) if f.endswith('.pkl') and 'movementDict' in f]
                yyyymmdd = input(fileDates)
            path = outputDir / (yyyymmdd + '_movementDict.pkl')
            data = pickle.load(open(path, "rb" ))
            return data, yyyymmdd
        
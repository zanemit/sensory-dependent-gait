from preprocessing import data_loader
import os
import numpy as np
import pandas as pd

def reorganise_metadata(outputDir):
    """
    reorganised the metadata downloaded from PyRAT into an easily readable df
    """
    metadata, yyyymmdd = data_loader.load_processed_data(dataToLoad='metadataPyRAT')
    from datetime import datetime

    # reformat dates
    dates = []
    for col in metadata.columns:
        if 'Date' in col:
            for d in metadata[col]:
                try:
                    dd, mm, yyyy = str(d).split('/')
                    dates.append(int(yyyy[-2:]+mm+dd))
                except:
                    if str(d) == 'nan':
                        continue
                    else:
                        raise ValueError(f'Problem caused by {d}!')

    # prepare tuples for a multi-column array
    tuples = [('mouseID', 'mouseID'), ('Sex', 'Sex')]
    for d in np.unique(dates):  # this is a sorted list of dates
        tuples.append(('Weight', d))
        tuples.append(('Age', d))

    mice = np.unique(metadata['ID'])
    arr = np.empty((mice.shape[0], len(tuples)), dtype='object')
    for im, mouse in enumerate(mice):
        metadata_sub = metadata[metadata['ID'] == mouse]
        dob = datetime.strptime(np.asarray(metadata_sub['DOB'])[0], "%d/%m/%Y")
        row = [f"{mouse[:3]}{mouse[4:]}", np.asarray(metadata_sub['Sex'])[0]]
        print(row)
        for tup in tuples:
            d_reconstructed = f'{str(tup[1])[4:]}/{str(tup[1])[2:4]}/20{str(tup[1])[:2]}'
            if tup[0] == 'Weight':
                i = np.where(np.asarray(metadata_sub)[
                             0, :] == d_reconstructed)[0]
                if len(i) > 0:
                    # pick the first one (only surgery days have two anyway)
                    i = i[0]
                    weight = metadata_sub.iloc[0, i-1]
                    if type(weight) == str:
                        row.append(float(f'{weight[:2]}.{weight[-1]}'))
                    else:
                        row.append(weight)
                else:
                    row.append(np.nan)
            elif tup[0] == 'Age':
                if not np.isnan(row[-1]):
                    d_strp = datetime.strptime(d_reconstructed, "%d/%m/%Y")
                    row.append(abs((d_strp-dob).days))
                else:
                    row.append(np.nan)
        arr[im, :] = np.asarray(row)

    index = pd.MultiIndex.from_tuples(tuples, names=["metaType", "expDate"])
    arr_df = pd.DataFrame(arr, columns=index)

    # print mean/sd summaries
    fmean = np.nanmean(np.asarray(arr_df.loc[:, 'Weight']).astype(float)[arr_df['Sex']['Sex'] == 'f'], axis=1).mean()
    mmean = np.nanmean(np.asarray(arr_df.loc[:, 'Weight']).astype(float)[arr_df['Sex']['Sex'] == 'm'], axis=1).mean()
    fsd = np.nanstd(np.asarray(arr_df.loc[:, 'Weight']).astype(float)[arr_df['Sex']['Sex'] == 'f'], axis=1).mean()
    msd = np.nanstd(np.asarray(arr_df.loc[:, 'Weight']).astype( float)[arr_df['Sex']['Sex'] == 'm'], axis=1).mean()
    print(f'Average weight of female mice: {fmean:.1f} +- {fsd:.1f} g')
    print(f'Average weight of male mice: {mmean:.1f} +- {msd:.1f} g')

    arr_df.to_csv(os.path.join(outputDir, yyyymmdd + '_metadataPyRAT_processed.csv'))
    
def get_weight_age(metadata_df, mouseID, expDate):
    """
    returns the weight and age of mouse when supplied metadataPyRAT_processed, mouseID, and expDate
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
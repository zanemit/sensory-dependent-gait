class Config:
    paths = { 
        "passiveOpto_data_folder": r"D:\Zane\PassiveOptoTreadmill\passiveOptoTreadmill",
        "passiveOpto_output_folder": r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill", 
        "forceplate_data_folder": r"Z:\murray\Zane\ForceSensors",
        "forceplate_output_folder": r"C:\Users\MurrayLab\Documents\Forceplate" , 
        "mtTreadmill_data_folder": r"Z:\murray\Zane\MotorisedTreadmill",
        "mtTreadmill_output_folder": r"C:\Users\MurrayLab\Documents\MotorisedTreadmill"
        }
    
    passiveOpto_config = {
        "stim_dur_dict" : {'10': '40', '20': '20', '30': '13.2', '40': '9.999999', '50': '8'}, # stimulus frequency-duration relationship as outputted by bonsai
        "fps": 400, # camera frame rate
        "sample_rate": 30000, # DAQ sample rate
        "px_per_cm" : {'lH': 96.3, 'rH': 69.0, 'lF': 96.3, 'rF': 69.0,
                       'lH0': 96.3, 'rH0': 69.0, 'lF0': 96.3, 'rF0': 69.0,
                       'lH1': 96.3, 'lH2': 96.3, 'rH1': 69.0, 'rH2': 69.0,
                       'lF1': 96.3, 'lF2': 96.3, 'rF1': 69.0, 'rF2': 69.0},
        "mm_per_g" : 1.25, # for weight-adjusting of head height ("rl-3" for a 22g mouse what "rl-8" is for a 26g mouse)
        "mice" : ['FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842',
                  'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
                  'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'],
        "kde_bin_num" : 100,
        "stride_num_threshold" : 40,
        }
    
    forceplate_config = {
        "fps": 100, # camera frame rate
        "sample_rate": 30000, # DAQ sample rate
        "trial_duration": 5, # seconds
        "px_per_cm" : 76.2,
        "fore_hind_post_cm" : 3.5, 
        "mm_per_g" : 1.25, # for weight-adjusting of head height ("rl-3" for a 22g mouse what "rl-8" is for a 26g mouse)
        "head_height_exp": 210925,
        "incline_exp" : 220401,
        "corrected_labels" : {'LHL': 'rF', 'RHL': 'lF', 'LFL': 'rH', 'RFL': 'lH'},
        "corrected_inclines" : {'deg-40': '40 deg', 'deg-20': '20 deg', 'deg0': '0 deg', 'deg20': '-20 deg', 'deg40': '-40 deg'}
        }
    
    mtTreadmill_config = {
        "mice_incline" : ['FAA1034924', 'FAA1034925', 'FAA1034926', 'FAA1034927',
                          'FAA1034928', 'FAA1034929', 'FAA1034930', 'FAA1034931',
                          'FAA1034932', 'FAA1034933'],
        "mice_level" : ['FAA1034608', 'FAA1034609', 'FAA1034610', 'FAA1034612',
                        'FAA1034613', 'FAA1034614', 'FAA1034626', 'FAA1034627',
                        'FAA1034630', 'FAA1034662', 'FAA1034663', 'FAA1034664'],
        "stride_num_threshold" : 40,
        "px_per_cm" : {'lH': 38.5, 'rH': 40, 'lF': 38.5, 'rF': 40,
                       'lH1': 38.5, 'lH2': 38.5, 'rH1': 40, 'rH2': 40,
                       'lF1': 38.5, 'lF2': 38.5, 'rF1': 40, 'rF2': 40,
                       'lF0': 38.5, 'lH0': 38.5, 'rH0': 40, 'rF0': 40,
                       'snout': 40, 'body': 40, 'tailbase': 40},
        "fps" : 400,
        }
    
    injection_config = {
        "left_inj_imp": ["FAA1034840", "FAA1034867", "FAA1034870","FAA1034942", "FAA1034947",
                           "FAA1034944", "FAA1034945", "FAA1034948", "FAA1034949", "FAA1034950",
                           "FAA1035017", "FAA1035018", "FAA1035021", "FAA1035022", "FAA1035103",
                           "FAA1035067", "FAA1035065", "FAA1035066", 'FAA1035414', 'FAA1035415',
                           'FAA1035416', 'FAA1035420'], 
        "right_inj_imp": ['BAA1098955',  'BAA1099004', 'FAA1034468', 'FAA1034469', 'FAA1034471', 
                       'FAA1034472', 'FAA1034570', 'FAA1034572', 'FAA1034573', 'FAA1034575', 
                       'FAA1034576', 'FAA1034665', 'FAA1034821', 'FAA1034827', 'FAA1034836', 
                       'FAA1034839', 'FAA1034833', 'FAA1034825', 'BAA1098921', 'FAA1034823', 
                       'FAA1034824', 'BAA1098923', 'BAA1099003', 'FAA1034470', 'BAA1098923',
                       'BAA1099270', 'FAA1034509', 'FAA1034510', "FAA1034825", "FAA1034827",
                       "FAA1034835", "FAA1034821", "FAA1034823", "FAA1034824", "FAA1034833",
                       "FAA1034836", "FAA1034839", "FAA1034841", "FAA1034842", "FAA1034868",
                       "FAA1034869", "FAA1034858", "FAA1034861", "FAA1034906", "FAA1034907",
                       "BAA1101344", "BAA1101343", "FAA1034943", "FAA1034946", "FAA1035062",
                       "FAA1035063", "FAA1035064", "FAA1035107", "FAA1035108", 'FAA1035413',
                       'FAA1035417', 'FAA1035418', 'FAA1035419', 'FAA1035421'],
        "both_inj_left_imp": ["FAA1034840", "FAA1034867"]
        }
    
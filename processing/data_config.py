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
    
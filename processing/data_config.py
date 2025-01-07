class Config:
    paths = { 
        "passiveOpto_data_folder": r"D:\Zane\PassiveOptoTreadmill\passiveOptoTreadmill",
        "passiveOpto_output_folder": r"C:\Users\MurrayLab\Documents\PassiveOptoTreadmill", 
        "forceplate_data_folder": r"F:\Forceplate",
        "forceplate_video_folder": r"F:\Forceplate",
        "forceplate_output_folder": r"C:\Users\MurrayLab\Documents\Forceplate" , 
        "mtTreadmill_data_folder": r"Z:\murray\Zane\MotorisedTreadmill",
        "mtTreadmill_output_folder": r"C:\Users\MurrayLab\Documents\MotorisedTreadmill",
        "openField_data_folder": r"Z:\murray\Zane\OpenField",
        "openField_output_folder": r"C:\Users\MurrayLab\Documents\OpenField"
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
        "mice_pilot": ['BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
               'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576'],
        "dtx_ctrl_mice": ['FAA1035268','FAA1035312','FAA1035336','FAA1035337',
                           'FAA1035340','FAA1035413'],
        "egr3_mice": ['FAA1035501', 'FAA1035528', 'FAA1035572', 'FAA1035603',
                      'FAA1035607', 'FAA1035608', 'FAA1035609', 'FAA1035676', 
                      'FAA1035696', 'FAA1035700'],
        "egr3_ctrl_mice": ['FAA1035504', 'CAA1120310', 'FAA1035571', 'FAA1035563',
                           'FAA1035568', 'FAA1035599', 'FAA1035600', 'FAA1035601',
                           'FAA1035604', 'FAA1035612', 'FAA1035613', 'FAA1035671'],
        "dtx_mice": ['FAA1035269','FAA1035273','FAA1035289','FAA1035288',
            'FAA105291','FAA1035313','FAA1035286','FAA1035287','FAA1035338',
            'FAA1035354','FAA1035366','FAA1035367','FAA1035357',
            'FAA1035358','FAA1035414','FAA1035415','FAA1035418','FAA1035419',
            'FAA1035420','FAA1035421','FAA1035423','FAA1035426',
            'FAA1035476', 'FAA1035477','FAA1035478'],
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
        "egr3_exp": 231029,
        "corrected_labels" : {'LHL': 'rF', 'RHL': 'lF', 'LFL': 'rH', 'RFL': 'lH'},
        "corrected_inclines" : {'deg-40': '40 deg', 'deg-20': '20 deg', 'deg0': '0 deg', 'deg20': '-20 deg', 'deg40': '-40 deg'},
        "passiveOpto_relations": {
            '2022-08-18': {
                'snoutBodyAngle': '2021-10-26',
                'incline': '2022-04-04'
                },
            '2023-09-21': {'snoutBodyAngle':'2021-10-26'},
            '2023-08-14': {'snoutBodyAngle':'2023-11-06'},                      
                                  }
        }
    
    mtTreadmill_config = {
        "mice_incline" : ['FAA1034924', 'FAA1034925', 'FAA1034926', 'FAA1034927',
                          'FAA1034928', 'FAA1034929', 'FAA1034930', 'FAA1034931',
                          'FAA1034932', 'FAA1034933'],
        "mice_level" : ['FAA1034608', 'FAA1034609', 'FAA1034610', 'FAA1034612',
                        'FAA1034613', 'FAA1034614', 'FAA1034626', 'FAA1034627',
                        'FAA1034630', 'FAA1034662', 'FAA1034663', 'FAA1034664'],
        "egr3_mice": ['FAA1035503', 'FAA1035523', 'FAA1035526', 'FAA1035528',
                      'FAA1035561', 'FAA1035567', 'FAA1035569', 'FAA1035572',
                      'FAA1035603', 'FAA1035607', 'FAA1035608', 'FAA1035610',
                      'FAA1035611'],
        "egr3_ctrl_mice": ['FAA1035504', 'CAA1120310', 'FAA1035563', 'FAA1035568',
                           'FAA1035571', 'FAA1035599', 'FAA1035600', 'FAA1035601',
                           'FAA1035602', 'FAA1035604', 'FAA1035612'],
        "stride_num_threshold" : 40,
        "px_per_cm" : {'lH': 38.5, 'rH': 40, 'lF': 38.5, 'rF': 40,
                       'lH1': 38.5, 'lH2': 38.5, 'rH1': 40, 'rH2': 40,
                       'lF1': 38.5, 'lF2': 38.5, 'rF1': 40, 'rF2': 40,
                       'lF0': 38.5, 'lH0': 38.5, 'rH0': 40, 'rF0': 40,
                       'snout': 40, 'body': 40, 'tailbase': 40},
        "fps": 400, # camera frame rate
        "sample_rate": 30000, # DAQ sample rate
        "max_trial_duration": 21,
        "cm/s_per_V" : int(150/5)
        }
    
    injection_config = {
        "left_inj_imp": ["FAA1034870","FAA1034942", "FAA1034947",
                           "FAA1034944", "FAA1034945", "FAA1034948", "FAA1034949", "FAA1034950",
                           "FAA1035017", "FAA1035018", "FAA1035021", "FAA1035022", "FAA1035103",
                           "FAA1035067", "FAA1035065", "FAA1035066", 'FAA1035414', 'FAA1035415',
                           'FAA1035416', 'FAA1035420', 'FAA1035504', 'FAA1035567', 'FAA1035568',
                           'FAA1035561', 'FAA1035562', 'FAA1035599', 'FAA1035562', 'FAA1035604',
                           'FAA1035602', 'FAA1035603', 'FAA1035691', 'FAA1035607', 'FAA1035610',
                           'FAA1035696', 'FAA1035611', 'FAA1035612', 'FAA1035636', 'FAA1035656', 
                           'FAA1035670', 'FAA1035684', 'FAA1035694', 'FAA1035699', 'FAA1035690'], 
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
                       'FAA1035417', 'FAA1035418', 'FAA1035419', 'FAA1035421', 'FAA1035501',
                       'FAA1035503', 'FAA1035523', 'FAA1035526', 'FAA1035528', 'FAA1035505',
                       'CAA1120310', 'FAA1035563', 'FAA1035572', 'FAA1035571', 'FAA1035569',
                       'FAA1035600', 'FAA1035601', 'FAA1035608', 'FAA1035609', 'FAA1035613',
                       'FAA1035620', 'FAA1035638', 'FAA1035700', 'FAA1035671', 'FAA1035676'],
        "both_inj_left_imp": ["FAA1034840", "FAA1034867", "FAA1035655"]
        }
    
    openField_config = {
        "fps": 100, # camera frame rate
        "sample_rate": 30000, # DAQ sample rate
        "rec_duration": 5, # minutes
        "px_per_cm" : 32.7,
        "mouse_procedures" : {
            '' :{}, # mice with no procedures (dates should be exp day 0)
            'lcTeLC' : {
                         'FAA1034906': (220603,1),
                         'FAA1034907': (220604,1),
                         'BAA1101344': (220605,1), 
                         'BAA1101343': (220611,1)
                         },
            'dtx' : {
                   'FAA1035268': (230126,1),
                   'FAA1035269': (230126,1),
                   'FAA1035270': (230130,1),
                   'FAA1035271': (230130,1),
                   'FAA1035272': (230126,1),
                   'FAA1035273': (230126,1),
                   'FAA1035265': (230130,1),
                   'FAA1035289': (230224,1),
                   'FAA1035288': (230224,1),
                   'FAA1035291': (230224,1),
                   'FAA1035313': (230224,1),
                   'FAA1035316': (230224,1),
                   'FAA1035312': (230224,1),
                   'FAA1035286': (230228,1),
                   'FAA1035287': (230228,1),
                   'FAA1035337': (230313,0.25),
                   'FAA1035338': (230313,0.25),
                   'FAA1035339': (230313,0.125),
                   'FAA1035340': (230313,0.125),
                   'FAA1035335': (230313,0.5),
                   'FAA1035336': (230313,0.125),
                   'FAA1035354': (230315,0.125),
                   'FAA1035366': (230315,0.125),
                   'FAA1035367': (230315,0.5),
                   'FAA1035357': (230315,0.25),
                   'FAA1035358': (230315,0.5),
                   'FAA1035413': (230424, 0.125),
                   'FAA1035414': (230424, 0.125),
                   'FAA1035415': (230424, 0.0625),
                   'FAA1035416': (230424, 0.0833),
                   'FAA1035417': (230424, 0.0625),
                   'FAA1035418': (230424, 0.0625),
                   'FAA1035419': (230424, 0.125),
                   'FAA1035420': (230424, 0.0833),
                   'FAA1035421': (230424, 0.0625),
                   'FAA1035423': (230524, 0.02),
                   'FAA1035426': (230524, 0.02),
                   'FAA1035447': (230807, 0.02), # no ataxia 
                   'FAA1035476': (230807, 0.25), 
                   'FAA1035477': (230807, 0.125),
                   'FAA1035478': (230807, 0.25),
                   },
            'egr3':{
                'FAA1035501': (230803,1),
                'FAA1035503': (230803,1),
                #'FAA1035523': (230822,1), no optogenetic response
                'FAA1035526': (230822,1),
                'FAA1035528': (230822,1),
                'FAA1035504': (230830,1), #wt
                'FAA1035505': (230830,1),
                'CAA1120310': (230830,1), #wt
                'FAA1035567': (231007,1),
                'FAA1035568': (231007,1), #wt
                'FAA1035561': (231007,1),
                'FAA1035562': (231007,1),
                'FAA1035563': (231007,1), #wt
                'FAA1035572': (231008,1),
                'FAA1035571': (231008,1), #wt
                'FAA1035569': (231008,1),
                'FAA1035599': (231103,1), #wt
                'FAA1035600': (231103,1), #wt
                'FAA1035601': (231103,1), #wt
                'FAA1035602': (231103,1), #wt
                'FAA1035603': (231103,1),
                'FAA1035604': (231104,1), #wt
                'FAA1035607': (231116,1),
                'FAA1035608': (231116,1),
                'FAA1035609': (231116,1),
                'FAA1035610': (231116,1),
                'FAA1035613': (231116,1), #wt
                'FAA1035611': (231116,1),
                'FAA1035612': (231116,1), #wt
                'FAA1035620': (231116,1),
                'FAA1035636': (240105,1),
                'FAA1035638': (240105,1),
                'FAA1035656': (240105,1),
                'FAA1035655': (240105,1), #wt
                'FAA1035670': (240105,1),
                'FAA1035671': (240105,1), #wt
                'FAA1035674': (240105,1),
                'FAA1035676': (240105,1),
                'FAA1035684': (240120,1),
                'FAA1035694': (240120,1),
                'FAA1035699': (240120,1),
                'FAA1035690': (240121,1),
                'FAA1035691': (240121,1), #wt
                'FAA1035696': (240121,1),
                'FAA1035700': (240121,1),
                }
            },
        'mice' : {
            'avil_pv' : [
                'FAA1035265','FAA1035269','FAA1035273','FAA1035288','FAA1035289',
                'FAA1035291','FAA1035313','FAA1035286','FAA1035287','FAA1035338',
                'FAA1035339','FAA1035354','FAA1035366','FAA1035367','FAA1035357',
                'FAA1035358','FAA1035414','FAA1035415','FAA1035418','FAA1035419',
                'FAA1035420','FAA1035421','FAA1035423','FAA1035426', 'FAA1035447', 
                'FAA1035476', 'FAA1035477','FAA1035478'
                ],
            'egr3' : ['FAA1035501', 'FAA1035503', 'FAA1035526', 'FAA1035528',
                      'FAA1035567', 'FAA1035561', 'FAA1035562', 'FAA1035572', 
                      'FAA1035569', 'FAA1035603', 'FAA1035607', 'FAA1035608',
                      'FAA1035609', 'FAA1035610', 'FAA1035611', 'FAA1035620',
                      'FAA1035636', 'FAA1035638', 'FAA1035656', 'FAA1035670',
                      'FAA1035674', 'FAA1035676', 'FAA1035684', 'FAA1035694',
                      'FAA1035699', 'FAA1035690', 'FAA1035696', 'FAA1035700']
            }
        
        
        }
    
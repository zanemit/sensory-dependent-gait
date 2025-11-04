remove_outliers <- function(vec){
  iqr = IQR(vec, na.rm = TRUE)
  q = quantile(vec, probs = c(0.25, 0.75), na.rm = TRUE)
  upper = q[2]+1.5*iqr
  lower = q[1]-1.5*iqr
  vec_no_outliers = vec[vec>lower & vec<upper]
  return(vec_no_outliers)
}

convert_deg <- function(x){
  return(x *-1)
}

convert_char <- function(char, elements, removeType){
  if (removeType == "start"){
    return(substr(char,elements,nchar(char)))
  } else if (removeType == "end"){
    return(substr(char,1,nchar(char)-elements))
  }
}

library(lme4)
library(MuMIn)
library(lmerTest)

### LIMB SUPPORT PC vs SLOPE, SBA, SPEED, TRIAL TYPE - VGLUT2
yyyymmdd = '2022-08-18'
outcome_variable = 'limbSupportPC3'
#outcome_variable = 'frac2hmlt'
cont_predictors = c('speed', 'snoutBodyAngle', 'incline')
cat_predictors = c('trialType') 
outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
filename = "_strideParamsMerged_incline_COMBINEDtrialType_"
refLimb = 'lH1'
mice = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949')
slope=TRUE
slope_type='pred2pred3'
interaction=TRUE
interaction_type='threeway'

#cont_predictors=c('speed', 'snoutBodyAngle')
#cat_predictors = c() 
#slope_type='pred2'
#interaction_type=''

cont_predictors=c('speed', 'snoutBodyAngle', 'incline')
cat_predictors = c() 
slope_type='pred2pred3'
interaction_type='threeway'

#------------------------------------------------------------
### LIMB SUPPORT PC vs SLOPE, SBA, SPEED, TRIAL TYPE - VGLUT2 - STRATIFIED BY SPEED

# FIG 4 
fit_limbSupportPC_vs_predictors(yyyymdd='2022-08-18', 
                                outcome_variable='limbSupportPC4', 
                                cont_predictors=c('speed', 'snoutBodyAngle'), 
                                cat_predictors=c(), 
                                filename="_strideParamsMerged_", 
                                speed_str='rHRlead_',
                                refLimb='lH1', 
                                mice=c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                                interaction=TRUE,
                                interaction_type='',
                                slope=TRUE,
                                slope_type='pred2',
                                outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
)

# FIG 4
fit_limbSupportPC_vs_predictors(yyyymdd='2022-08-18', 
                                outcome_variable='limbSupportPC1', 
                                cont_predictors=c('speed', 'snoutBodyAngle', 'incline'), 
                                cat_predictors=c(), 
                                filename="_strideParamsMerged_incline_", 
                                speed_str='rHRlead_',
                                refLimb='lH1', 
                                mice=c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                                interaction=TRUE,
                                interaction_type='threeway',
                                slope=TRUE,
                                slope_type='pred2pred3',
                                outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
)


#------------------------------------------------------------
### LIMB SUPPORT PC vs SLOPE, SBA, SPEED, TRIAL TYPE - MOTORISED TREADMILL vGLUT2 LEVEL
yyyymmdd = '2021-10-23'
outcome_variable = 'limbSupportPC4'
cont_predictors = c('speed', 'snoutBodyAngle')
cat_predictors = c() 
interaction = TRUE
interaction_type=''
outputDir = "C:\\Users\\MurrayLab\\Documents\\MotorisedTreadmill\\"
filename = "_strideParamsMerged_"
refLimb = 'lH1'
mice = c('FAA1034608', 'FAA1034609', 'FAA1034610', 'FAA1034612',
         'FAA1034613', 'FAA1034614', 'FAA1034626', 'FAA1034627',
         'FAA1034630', 'FAA1034662', 'FAA1034663', 'FAA1034664')
slope=TRUE
slope_type='pred2'

#---------------------------------------------------------------

#------------------------------------------------------------
### LIMB SUPPORT PC vs SLOPE, SBA, SPEED, TRIAL TYPE - MOTORISED TREADMILL vGLUT2 SLOPE
yyyymmdd = '2022-05-06'
outcome_variable = 'limbSupportPC2'
cont_predictors = c('speed', 'snoutBodyAngle', 'incline')
cat_predictors = c() 
interaction = TRUE
interaction_type='threeway'
outputDir = "C:\\Users\\MurrayLab\\Documents\\MotorisedTreadmill\\"
filename = "_strideParamsMerged_"
refLimb = 'lH1'
mice = c('FAA1034924', 'FAA1034925', 'FAA1034926', 'FAA1034927',
         'FAA1034928', 'FAA1034929', 'FAA1034930', 'FAA1034931',
         'FAA1034932', 'FAA1034933')
slope=TRUE
slope_type='pred2pred3'

#---------------------------------------------------------------

#------------------------------------------------------------
### LIMB SUPPORT PC vs SLOPE, SBA, SPEED, TRIAL TYPE - MOTORISED TREADMILL vGLUT2 SLOPE+LEVEL (MERGED)
yyyymmdd = '2022-05-06'
outcome_variable = 'limbSupportPC2'
cont_predictors = c('speed', 'snoutBodyAngle', 'incline')
cat_predictors = c('trialType') 
interaction = TRUE
interaction_type='threeway'
outputDir = "C:\\Users\\MurrayLab\\Documents\\MotorisedTreadmill\\"
filename = "_strideParamsMerged_COMBINEDtrialType_"
refLimb = 'lH1'
mice = c('FAA1034924', 'FAA1034925', 'FAA1034926', 'FAA1034927',
         'FAA1034928', 'FAA1034929', 'FAA1034930', 'FAA1034931',
         'FAA1034932', 'FAA1034933','FAA1034608', 'FAA1034609', 'FAA1034610', 'FAA1034612',
         'FAA1034613', 'FAA1034614', 'FAA1034626', 'FAA1034627',
         'FAA1034630', 'FAA1034662', 'FAA1034663', 'FAA1034664')
slope=TRUE
slope_type='pred2pred3'
#---------------------------------------------------------------

fit_limbSupportPC_vs_predictors <- function(yyyymdd, 
                             outcome_variable, 
                             cont_predictors, 
                             cat_predictors, 
                             filename, 
                             refLimb, 
                             mice,
                             speed_str='',
                             interaction=TRUE,
                             interaction_type='',
                             slope=TRUE,
                             slope_type='pred2',
                             outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
                             ){
    filePath = paste(outputDir, yyyymmdd, filename, speed_str, refLimb, ".csv", sep="")
    df <- read.csv(file = filePath)
    df <- df[df$mouseID %in% mice, ]
    
    #df <- subset(df, trialType == 'headHeight')
    #df <- subset(df, trialType == 'slope')
    
    if ('incline' %in% cont_predictors){
      df$incline = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
      df$incline = as.numeric(df$incline)
      df$incline = lapply(df$incline, convert_deg)
      df$incline = as.numeric(df$incline)
    }
    
    for (i in seq_along(cont_predictors)){
      df[paste('pred',i, '_centred', sep="")] = (df[[cont_predictors[i]]] - mean(df[[cont_predictors[i]]], na.rm = TRUE))/sd(df[[cont_predictors[i]]], na.rm = TRUE)
    }
    
    predictors = c(cont_predictors, cat_predictors)
    
    df["outcome_centred"] = (df[[outcome_variable]] - mean(df[[outcome_variable]], na.rm = TRUE))/sd(df[[outcome_variable]], na.rm = TRUE)
    if ('trialType' %in% cat_predictors){
      df["trialType"] = as.factor(df$trialType)
    }
    
    if (length(cont_predictors)==2 & length(cat_predictors)==0){
      file_ext = paste("_mixedEffectsModel_linear_", speed_str, outcome_variable, "_vs_", cont_predictors[1], "_", cont_predictors[2], "_randSlopes", slope_type, "_interaction", interaction, interaction_type, ".csv", sep="")
      
      if (interaction==FALSE){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + (1|mouseID), data = df)
          
        } else{
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + (pred2_centred|mouseID), data = df)
        }
      } else{
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + (1|mouseID), data = df)
          
        } else{
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + (pred2_centred|mouseID), data = df)
        }
      }
    } else if (length(cont_predictors)==3 & length(cat_predictors)==0){
      file_ext = paste("_mixedEffectsModel_linear_", speed_str, outcome_variable, "_vs_", cont_predictors[1], "_", cont_predictors[2], "_", cont_predictors[3],  "_randSlopes", slope_type, "_interaction", interaction, interaction_type, ".csv", sep="")
      
      if (interaction==FALSE){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + pred3_centred + (1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + pred3_centred + (pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else if (interaction_type=='threeway'){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred + (1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred + (pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else if (interaction_type=='secondary'){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred * pred3_centred + (1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred * pred3_centred + (pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else { # interaction TRUE, type =''
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + pred3_centred + (1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + pred3_centred + (pred2_centred+pred3_centred|mouseID), data = df)
        }
      }
    } else if (length(cont_predictors)==3 & length(cat_predictors)==1){
      file_ext = paste("_mixedEffectsModel_linear_", speed_str, outcome_variable, "_vs_", cont_predictors[1], "_", cont_predictors[2], "_", cont_predictors[3], "_", cat_predictors[1], "_randSlopes", slope_type, "_interaction", interaction, interaction_type, ".csv", sep="")
      
      if (interaction==FALSE){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + pred3_centred + trialType +(1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred + pred3_centred + trialType +(pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else if (interaction_type=='threeway'){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred + trialType +(1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          print('fitting here!')
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred + trialType +(pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else if (interaction_type=='secondary'){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred * pred3_centred + trialType +(1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred + pred2_centred * pred3_centred + trialType +(pred2_centred+pred3_centred|mouseID), data = df)
        }
      } else if (interaction_type=='fourway'){
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred * trialType +(1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred * pred3_centred * trialType +(pred2_centred+pred3_centred|mouseID), data = df)
        }
      }
      
      else { # interaction TRUE, type =''
        if (slope==FALSE){
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + pred3_centred + trialType +(1|mouseID), data = df)
          
        } else{ # slope type must be updated manually!
          modelLinear = lmer(outcome_centred ~ pred1_centred * pred2_centred + pred3_centred + trialType +(pred2_centred+pred3_centred|mouseID), data = df)
        }
      }
      
    }
    print(summary(modelLinear)$coefficients)
    write.csv(summary(modelLinear)$coefficients,paste(outputDir, yyyymmdd, "_contCoefficients",file_ext, sep=""))
    write.csv(ranef(modelLinear)$mouseID, paste(outputDir, yyyymmdd, "_randCoefficients", file_ext, sep=""))
}


# ------------------------------------------------------
### LIMB SUPPORT PC vs HOMOLATERAL PHASE (head height vs slope)
fit_limbSupportPC_vs_limbphase(yyyymdd='2022-08-18', 
                               outcome_variable='limbSupportPC1', 
                               mice=mice,
                               cat_predictors=c('trialType'),
                               filename= "_strideParamsMerged_incline_COMBINEDtrialType_",
                               refLimb= 'lH1', 
                               speed_str='rHalt_',
                               outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
)

fit_limbSupportPC_vs_limbphase(yyyymdd='2022-08-18', 
                               outcome_variable='limbSupportPC2', 
                               mice=mice,
                               filename= "_strideParamsMerged_incline_", 
                               refLimb= 'lH1', 
                               speed_str='rHsync_',
                               outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
)


fit_limbSupportPC_vs_limbphase(yyyymdd='2022-08-18', 
                               outcome_variable='limbSupportPC1', 
                               mice=mice,
                               filename= "_strideParamsMerged_", 
                               refLimb= 'lH1', 
                               speed_str='rHsync_',
                               outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
)

fit_limbSupportPC_vs_limbphase <- function(yyyymdd, 
                                            outcome_variable, 
                                            cat_predictors=c(''),
                                            mice,
                                            filename= "_strideParamsMerged_incline_", 
                                            refLimb= 'lH1', 
                                            speed_str='rHalt_',
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"
                                             ){
    filePath = paste(outputDir, yyyymmdd, filename, speed_str, refLimb, ".csv", sep="")
    df <- read.csv(file = filePath)
    
    # convert phase to radians [0, 2pi]
    df$limb = (df[[limb]]*2*pi)
    limb_bool = df$limb<0
    limb_bool[which(is.na(limb_bool))] = FALSE
    df$limb[limb_bool] = df$limb[limb_bool]+(2*pi)
    
    df$limb_sin = sin(df[[limb]])
    df$limb_cos = cos(df[[limb]])
    
    cols_to_keep = c("limb_sin", "limb_cos", limb, outcome_variable, "mouseID", "strideNum")
    if (grepl('trialType', cat_predictors)){
      cols_to_keep = c(cols_to_keep, 'trialType')
    }
    df_new = na.omit(df[cols_to_keep])
    if (grepl('trialType', cat_predictors)){
      df_new$trialType = as.factor(df_new$trialType)
    }
    df_new$outcome = df_new[[outcome_variable]]
    
    if (grepl('incline', filename)){
      inc_str = 'incline_'
    } else{
      inc_str = ''
    }
    
    if (grepl('trialType', cat_predictors)){
      print('here')
      file_ext = paste("_mixedEffectsModel_linear_",speed_str, outcome_variable, "_vs_", limb, '_',inc_str, "trialType_randSlopesCosSin.csv", sep="") 
      modelLinear = lmer(outcome ~ limb_sin + limb_cos + (limb_sin + limb_cos|mouseID), data = df_new)
    } else{
      file_ext = paste("_mixedEffectsModel_linear_",speed_str, outcome_variable, "_vs_", limb, '_',inc_str, "randSlopesCosSin.csv", sep="") 
      modelLinear = lmer(outcome ~ limb_sin + limb_cos + trialType + (limb_sin + limb_cos|mouseID), data = df_new)
    }
      
    write.csv(summary(modelLinear)$coefficients,paste(outputDir, yyyymmdd, "_contCoefficients",file_ext, sep=""))
    write.csv(ranef(modelLinear)$mouseID, paste(outputDir, yyyymmdd, "_randCoefficients", file_ext, sep=""))
}

# ------------------------------------------------------
### LIMB SUPPORT PC vs HOMOLATERAL PHASE (different strains: egr3 vs egr3ctrl vs vglut2)
#path = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\2023-09-21_strideParamsMerged_COMBINEDstrain_egr3ctrl.csv"
#mice = c('FAA1035504', 'CAA1120310', 'FAA1035571', 'FAA1035563',
#         'FAA1035568', 'FAA1035599', 'FAA1035600', 'FAA1035601',
#         'FAA1035604', 'FAA1035612', 'FAA1035613', 'FAA1035671',
#         'FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 
#         'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
#         'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949')
#yyyymmdd = '2023-09-21'

path = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\2023-08-14_strideParamsMerged_COMBINEDstrain_egr3.csv"
mice = c('FAA1035501', 'FAA1035528', 'FAA1035572', 'FAA1035603',
         'FAA1035607', 'FAA1035608', 'FAA1035609', 'FAA1035676', 
         'FAA1035696', 'FAA1035700',
         'FAA1035504', 'CAA1120310', 'FAA1035571', 'FAA1035563',
         'FAA1035568', 'FAA1035599', 'FAA1035600', 'FAA1035601',
         'FAA1035604', 'FAA1035612', 'FAA1035613', 'FAA1035671')
yyyymmdd = '2023-08-14'

outcome_variable = 'limbSupportPC4'
cat_predictors = c('strain') 
interaction = TRUE
interaction_type='secondary'
refLimb = 'lH1'
limb = 'lF0'
outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\"



df <- read.csv(path)

# convert phase to radians [0, 2pi]
df$limb = (df[[limb]]*2*pi)
limb_bool = df$limb<0
limb_bool[which(is.na(limb_bool))] = FALSE
df$limb[limb_bool] = df$limb[limb_bool]+(2*pi)

df$limb_sin = sin(df[[limb]])
df$limb_cos = cos(df[[limb]])

df_new = na.omit(df[c("limb_sin", "limb_cos", limb, "strain", outcome_variable, "mouseID")])
df_new$strain = as.factor(df_new$strain)
df_new$outcome = df_new[[outcome_variable]]

modelLinear = lmer(outcome ~ limb_sin + limb_cos + strain +(limb_sin + limb_cos|mouseID), data = df_new)

file_ext = paste("_mixedEffectsModel_linear_", outcome_variable, "_vs_", limb, "_strain_randSlopesCosSin.csv", sep="")

write.csv(summary(modelLinear)$coefficients,paste(outputDir, yyyymmdd, "_contCoefficients",file_ext, sep=""))
write.csv(ranef(modelLinear)$mouseID, paste(outputDir, yyyymmdd, "_randCoefficients", file_ext, sep=""))


source("C:\Users\MurrayLab\sensory-dependent-gait\lme\utils_R.R")

fp_lme_forcedata_vs_param <- function(
                                      yyyymmdd, 
                                      param, 
                                      interaction = TRUE, 
                                      outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\"
                                      ){
  # This function fits a linear mixed effects model to force sensor data
  # (hindlimb fraction, forelimb fraction, CoM x position, CoM y position, total
  # weight, hindlimb weight fraction, and forelimb weight fraction) with param 
  # ("headHW", "snoutBodyAngle", or "levels") as fixed effects and "mouseID" as 
  # a random effect.
  #     param "headHW" : weight-adjusted head height, relevant only to data from
  #                     head height trials
  #     param "snuotBodyAngle" : snout-hump angles
  #     param "levels" : re-centred head height in head height trials or 
  #                     corrected incline, depending on the supplied dataset
  #
  # If interaction is FALSE or if filename does not contain "COMBINED":
  #     snoutHumpAngle ~ param + 1|mouseID
  #
  # If interaction is TRUE and filename contains "COMBINED":
  #     snoutHumpAngle ~ param * headHeight + 1|mouseID 
  #
  # If interaction is FALSE and filename contains "COMBINED":
  #     forceSensorVariable ~ param + headHeight + 1|mouseID
  #
  # In each scenario, two models are fitted: one with the first predictor 
  # (param) as a linear predictor and one with it as a quadratic 
  # predictor.
  #
  # Three files are saved:
  #     {yyyymmdd}_mixedEffectsModel_linear_{predictor}_{param}_interaction{BOOL}.csv
  #               summary of linear model coefficient
  #     {yyyymmdd}_mixedEffectsModel_quadratic_{predictor}_{param}_interaction{BOOL}.csv
  #               summary of quadratic model coefficients
  #     {yyyymmdd}_mixedEffectsModel_AICRsq_{predictor}_{param}_interaction{BOOL}.csv
  #               AIC and R-squared values for both models
  
  
  if(grepl('x', yyyymmdd)){
    filePath = paste(outputDir, yyyymmdd, "_meanParamDF_", param, "_COMBINED.csv", sep="")
  } else{
    filePath = paste(outputDir, yyyymmdd, "_meanParamDF_", param, ".csv", sep="")
  }
  df <- read.csv(filePath)
  
  # CENTRE THE VARIABLES
  df$param_centred = df$param-mean(df$param, na.rm = TRUE) # headHW, snoutBodyAngle, or levels (rl or deg depending on yyyymmdd)
  
  predictors = c('hind_frac', 'fore_frac', 'CoMx_mean', 'CoMy_mean', 'total_pressure', 'fore_weight_frac', 'hind_weight_frac', 'headplate_weight_frac')
  predictor_strs = c('hindfrac', 'forefrac', 'CoMx', 'CoMy', 'tpressure', 'foreWfrac', 'hindWfrac', 'headWfrac')
  
  for (i in 1:length(predictors)){
    df$pred_centred = df[[predictors[i]]] - mean(df[[predictors[i]]], na.rm=TRUE)
    file_ext = paste(predictor_strs[i], "_", param, sep='')
    if (grepl('COMBINED', filePath)){
      df$headHeight = as.factor(df$headHeight)
      df$mouseID = df$mouse
      if (interaction == TRUE){
        file_ext = paste(predictor_strs[i], "_", param, "_interactionTRUE", sep='')
        modelLinear = lmer(pred_centred ~ param_centred * headHeight + (1|mouseID), data = df )
        modelQuadratic = lmer(pred_centred  ~ poly(param_centred,2, raw = TRUE) * headHeight + (1|mouseID), data = df )
      }else{
        modelLinear = lmer(pred_centred ~ param_centred + headHeight + (1|mouseID), data = df )
        modelQuadratic = lmer(pred_centred  ~ poly(param_centred,2, raw = TRUE) + headHeight + (1|mouseID), data = df )
      }
    } else{
      modelLinear = lmer(pred_centred ~ param_centred + (1|mouse), data = df )
      modelQuadratic = lmer(pred_centred ~ poly(param_centred, 2, raw = TRUE) + (1|mouse), data = df )
    }
    write.csv(summary(modelLinear)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext,".csv", sep=""))
    write.csv(summary(modelQuadratic)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_", file_ext, ".csv", sep=""))
    
    AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
    write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_", file_ext,".csv", sep=""))
    
  }
}

## EXAMPLES ##
## head height dataset: force sensor data vs snout-hump angle
# fp_lme_forcedata_vs_param(yyyymmdd = '2021-10-26', param = 'snoutBodyAngle', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## head height dataset: force sensor data vs head height
# fp_lme_forcedata_vs_param(yyyymmdd = '2021-10-26', param = 'levels', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## head height dataset: force sensor data vs weight-adjusted head height
# fp_lme_forcedata_vs_param(yyyymmdd = '2021-10-26', param = 'headHW', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## incline (head low) dataset: force sensor data vs snout-hump angle
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-02', param = 'snoutBodyAngle', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## incline (head low) dataset: force sensor data vs incline
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-02', param = 'levels', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## incline (head high) dataset: force sensor data vs snout-hump angle
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-04', param = 'snoutBodyAngle', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## incline (head high) dataset: force sensor data vs incline
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-04', param = 'levels', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## combined incline dataset: force sensor data vs snout-hump angle
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-0x', param = 'snoutBodyAngle', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

## combined incline dataset: force sensor data vs incline
# fp_lme_forcedata_vs_param(yyyymmdd = '2022-04-0x', param = 'levels', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

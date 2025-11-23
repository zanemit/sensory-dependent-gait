library(lme4)
library(MuMIn)
library(lmerTest)

convert_char <- function(char, elements, removeType){
  if (removeType == "start"){
    return(substr(char,elements,nchar(char)))
  } else if (removeType == "end"){
    return(substr(char,1,nchar(char)-elements))
  }
}

convert_rl <- function(x){
  print(x)
  return(x *-1 + 6)
}

convert_deg <- function(x){
  return(x *-1)
}

generate_simple_LINEAR_mixed_effects_models <- function(yyyymmdd, 
                                                indep_var,
                                                 dep_var,
                                                slope_enforced=TRUE,
                                                filename = "meanParamDF",
                                                  outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\"){
  
  filePath = paste(outputDir, yyyymmdd, "_", filename, "_", indep_var, ".csv", sep="")
  if (dep_var == "rH1x" | indep_var == 'stimFreq' | grepl("strideParams", filename)){
    filePath = paste(outputDir, yyyymmdd, "_", filename, ".csv", sep="")
  }
  df <- read.csv(filePath)
  
  if (indep_var == "stimFreq"){
    df$param = as.numeric(convert_char(df$stimFreq, 2, removeType = 'end'))
  }
  
  if (filename == "forceplateAngleParamsR"){
    param = 'headLVL'
    df$param = df[[param]]
  }
  if (dep_var == "rH1x"){
    df$param = lapply(df[[indep_var]], convert_char, elements = 4, removeType = 'start')
    df$param = as.numeric(df$param)
    df$param = lapply(df$param, convert_deg)
    df$param = as.numeric(df$param) 
  }
  if (indep_var == 'speed'){
    df$param = df$speed
  }
  
  if (slope_enforced==TRUE){
    slope_enforced_str = 'slopeENFORCED'
  } else{
    slope_enforced_str = 'BEST'
  }
  
  predictor_str_dict = list(
    'hind_frac' = 'hindfrac',
    'fore_frac' = 'forefrac',
    'CoMx_mean' = 'CoMx',
    'CoMy_mean' = 'CoMy',
    'total_pressure' = 'tpressure',
    'fore_weight_frac' = 'foreWfrac',
    'hind_weight_frac' = 'hindWfrac',
    'headplate_weight_frac' = 'headWfrac',
    'snoutBodyAngle' = 'snoutBodyAngle', 
    'rH1x' = 'rH1x', 
    'meanSpeed' = 'meanSpeed', 
    'maxSpeed' = 'maxSpeed', 
    'strideLength' = 'strideLength', 
    'strideFreq' = 'strideFreq', 
    'duty_lH' = 'duty_lH'
  )
  
  dep_var_str = predictor_str_dict[dep_var]
  file_ext = paste(dep_var_str, indep_var, slope_enforced_str,  sep='_')
  
  # CENTRE THE VARIABLES ("param" column in df is the same as indep_var due to the way meanParamDF is created)
  df$indep_var_centred = df$param-mean(df$param, na.rm = TRUE) # headHW, snoutBodyAngle, or levels (rl or deg depending on yyyymmdd)
  
  if (dep_var=='headplate_weight_frac'){
    df[[dep_var]] = -(df[[dep_var]]-1)*100
  }
  
  df$pred_centred = df[[dep_var]] - mean(df[[dep_var]], na.rm=TRUE)
  
  df$mouseID = df$mouse
  models <- list()
  conv_fails <- list(model1 = TRUE)
  
  
  # RANDOM INTERCEPT MODEL
  models$model1 = lmer(pred_centred ~ indep_var_centred + (1|mouseID), data = df)
  r2_model1 = r.squaredGLMM(models$model1)
  aic_model1 = AIC(models$model1)
  conv_fails$model1 = is.null(models$model1@optinfo$conv$lme4$messages)
  
  # RANDOM SLOPE MODEL
  models$model2 = lmer(pred_centred ~ indep_var_centred + (indep_var_centred|mouseID), data = df)
  r2_model2 = r.squaredGLMM(models$model2)
  aic_model2 = AIC(models$model2)
  conv_fails$model2 = is.null(models$model2@optinfo$conv$lme4$messages)
  
  # returns R2m - marginal R squared (variance explained by fixed effects alone)
  # R2c - conditional R squared (variance explained by both fixed and random effects)
  
  AICsq_df <- data.frame(
    R2m = c(r2_model1[1], r2_model2[1]),
    R2c = c(r2_model1[2], r2_model2[2]),
    AIC = c(aic_model1, aic_model2)
  )
  row.names(AICsq_df) <- c("random_int", "random_int_slope")

  write.csv(AICsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_", file_ext,".csv", sep=""))
  
  model_strs = c('Intercept', 'Slope')
  
  # find the minimum AIC
  minAIC_id = which.min(AICsq_df$AIC)
  if (slope_enforced==TRUE){
    write.csv(summary(models[[2]])$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext, "_rand", model_strs[[2]],".csv", sep=""))
  }
  else if (conv_fails[[minAIC_id]]){
    # model has converged, can be saved
    write.csv(summary(models[[minAIC_id]])$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext,"_rand", model_strs[[minAIC_id]], ".csv", sep=""))
    
    # print model parameters
    cat("\nIntercept: ", summary(models[[minAIC_id]])$coefficients['(Intercept)', 'Estimate']+mean(df[[dep_var]], na.rm=TRUE))
    cat("\nSlope: ", summary(models[[minAIC_id]])$coefficients['param_centred', 'Estimate'])
    
    # store optimal model ID
    optimal_model_id = minAIC_id
    
  } else{
    AICsq_df_remove1 = AICsq_df[-minAIC_id, ]
    # find the minimum AIC among the remaining AICs
    minAIC_id_remove1 = which(AICsq_df$AIC == min(AICsq_df_remove1$AIC))
    if (conv_fails[[minAIC_id_remove1]]){
      # model has converged, can be saved
      write.csv(summary(models[[minAIC_id_remove1]])$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext,"_rand",model_strs[[minAIC_id_remove1]],".csv", sep=""))
      
      # print model parameters
      cat("\nIntercept: ", summary(models[[minAIC_id_remove1]])$coefficients['(Intercept)', 'Estimate']+mean(df[[dep_var]], na.rm=TRUE))
      cat("\nSlope: ", summary(models[[minAIC_id_remove1]])$coefficients['param_centred', 'Estimate'])
      
      optimal_model_id = minAIC_id_remove1
    } else{
      AICsq_df_remove2 = AICsq_df_remove1[-minAIC_id_remove1, ]
      # find the minimum AIC among the remaining AICs
      minAIC_id_remove2 = which(AICsq_df$AIC == min(AICsq_df_remove2$AIC))
      write.csv(summary(models[[minAIC_id_remove2]])$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext,"_rand", model_strs[[minAIC_id_remove2]],".csv", sep=""))
      
      # print model parameters
      cat("\nIntercept: ", summary(models[[minAIC_id_remove2]])$coefficients['(Intercept)', 'Estimate']+mean(df[[dep_var]], na.rm=TRUE))
      cat("\nSlope: ", summary(models[[minAIC_id_remove2]])$coefficients['param_centred', 'Estimate'])
      
      optimal_model_id = minAIC_id_remove2
      }
  }
  if (!slope_enforced==TRUE){
    cat("\nOptimal model is model", optimal_model_id)
  } else{
    optimal_model_id=2
  }
  
  if (optimal_model_id %in% c(2,3)){ #random effects models
    cat("\nThe optimal model has random effects. Intercept STDEV is", attr(summary(models[[optimal_model_id]])$varcor$mouseID, "stddev"))
    print(paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_randomEffects_", file_ext,"_rand", model_strs[[optimal_model_id]],".csv", sep=""))
    write.csv(ranef(models[[optimal_model_id]])[["mouseID"]], paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_randomEffects_", file_ext,"_rand", model_strs[[optimal_model_id]],".csv", sep=""))
  }
  
  plot(fitted(models[[optimal_model_id]]), resid(models[[optimal_model_id]]))
  abline(h=0, col='red')
  
  qqnorm(resid(models[[optimal_model_id]]))
  qqline(resid(models[[optimal_model_id]]), col='red')
}

# Fig 1D
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2021-10-26', 
                                            indep_var='snoutBodyAngle', 
                                            dep_var='CoMy_mean', 
                                            slope_enforced=TRUE)

# Fig 1E
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2021-10-26', 
                                            indep_var='snoutBodyAngle', 
                                            dep_var='headplate_weight_frac', 
                                            slope_enforced=TRUE) 

# Fig 1F
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2022-04-04', 
                                            indep_var='levels', 
                                            dep_var='CoMy_mean', 
                                            slope_enforced=TRUE)

# Fig 1G
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2022-04-04', 
                                            indep_var='levels', 
                                            dep_var='headplate_weight_frac', 
                                            slope_enforced=TRUE)

# Fig S1B
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2021-10-26', 
                                            indep_var='headHW', 
                                            dep_var='hind_weight_frac',
                                            slope_enforced=TRUE) 

# Fig S1E
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2021-10-26', 
                                            indep_var='headHW', 
                                            dep_var='CoMx_mean',
                                            slope_enforced=TRUE) 

# Fig S1G
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2022-04-04', 
                                            indep_var='levels', 
                                            dep_var='fore_weight_frac',
                                            slope_enforced=TRUE)

# Fig S1H
generate_simple_LINEAR_mixed_effects_models(yyyymmdd='2022-04-04', 
                                            indep_var='levels', 
                                            dep_var='hind_weight_frac',
                                            slope_enforced=TRUE)

# Fig 2D right
generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'stimFreq',
                                            dep_var = 'maxSpeed',
                                            filename = "locomParamsAcrossMice",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\",
                                            slope_enforced=TRUE)

# Fig 2D left
generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'stimFreq',
                                            dep_var = 'meanSpeed',
                                            filename = "locomParamsAcrossMice",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\",
                                            slope_enforced=TRUE)




#
generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'incline',
                                            dep_var = 'snoutBodyAngle',
                                            filename = "passiveOpto_angles",
                                            outputDir = "C:\\Users\\MurrayLab\\Desktop\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-05-06', 
                                            indep_var = 'incline',
                                            dep_var = 'snoutBodyAngle',
                                            filename = "mtTreadmill_angles",
                                            outputDir = "C:\\Users\\MurrayLab\\Desktop\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'incline',
                                            dep_var = 'snoutBodyAngle',
                                            filename = "passiveOpto_anglesLOCOM",
                                            outputDir = "C:\\Users\\MurrayLab\\Desktop\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'trialType',
                                            dep_var = 'rH1x',
                                            filename = "limbPositionRegressionArray_preOpto_levels_incline",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'trialType',
                                            dep_var = 'rH1x',
                                            filename = "limbPositionRegressionArray_locom_levels_incline",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-05-06', 
                                            indep_var = 'trialType',
                                            dep_var = 'rH1x',
                                            filename = "limbPositionRegressionArray_egocentric",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\MotorisedTreadmill\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-04-04', 
                                            indep_var = 'level',
                                            dep_var = 'rH1x',
                                            filename = "limbPositionRegressionArray",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")


generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'speed',
                                            dep_var = 'strideLength',
                                            filename = "strideParams_lH1",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'speed',
                                            dep_var = 'strideFreq',
                                            filename = "strideParams_lH1",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'speed',
                                            dep_var = 'duty_lH',
                                            filename = "strideParamsMerged_lH1",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\")
  
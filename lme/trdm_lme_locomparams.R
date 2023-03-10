source("C:\Users\MurrayLab\sensory-dependent-gait\lme\utils_R.R")

lme_locomParams_vs_stimFreq <- function(yyyymmdd, 
                                        param, 
                                        appdx, 
                                        interaction = FALSE, 
                                        outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                                        filename = "_locomParamsAcrossMice", 
                                        mouselist = FALSE){
  # This function fits a linear mixed effects model to locomotor parameters 
  # (duration, latency, mean/max/median speed) from the treadmill trials with 
  # param ("headHW", "headLVL", "snoutBodyAngle") as fixed effects and "mouseID" 
  # as a random effect.
  #     appdx should be "" for head height trials or "incline"
  #
  #     param "snoutBodyAngle" : snout-hump angles
  #     param "headHW" : weight-adjusted head height, relevant only to data from
  #                     head height trials
  #     param "headLVL" : re-centred head height in head height trials or 
  #                     corrected incline 
  #
  # If param is "", "limbDist" is added to the locomParam variables. This is 
  # distance that a limb moves during a stride.
  #
  # The fitted model:
  #   if param is not "" and appdx is "COMBINED":
  #     locomParam ~ stimFreq + param + trialType + 1|mouseID
  #
  #   if param is not "" and appdx is not "COMBINED":
  #     locomParam ~ stimFreq + param + 1|mouseID
  #
  #   if param is "" and appdx is "COMBINED" and interaction is TRUE:
  #     locomParam ~ stimFreq * trialType + 1|mouseID
  #
  #   if param is "" and appdx is "COMBINED" and interaction is FALSE:
  #     locomParam ~ stimFreq + trialType + 1|mouseID
  #
  #   if param is "" and appdx is not "COMBINED":
  #     locomParam ~ stimFreq + 1|mouseID
  #
  # In each scenario, two models are fitted: one with param as a linear 
  # predictor and one with it as a quadratic predictor.
  #
  # Three files are saved:
  #     {yyyymmdd}_mixedEffectsModel_linear_{locomParam}_v_stimFreq_{param}{appdx}.csv
  #               summary of linear model coefficient
  #     {yyyymmdd}_mixedEffectsModel_quadratic_{locomParam}_v_stimFreq_{param}{appdx}.csv
  #               summary of quadratic model coefficients
  #     {yyyymmdd}_mixedEffectsModel_AICRsq_{locomParam}_v_stimFreq_{param}{appdx}.csv
  #               AIC and R-squared values for both models
  
  appdx = get_appdx(appdx)
  
  filePath = paste(outputDir, yyyymmdd, filename, appdx, sep="")
  df <- read.csv(filePath)
  
  if (typeof(mouselist)=='character'){ 
    df = df[df$mouseID %in% mouselist,]
  }
  
  # CONVERT STIM_FREQ TO NUMERIC
  df$stimFreq_int = lapply(df$stimFreq, convert_char, elements = 2, removeType = 'end')
  df$stimFreq_int = as.numeric(df$stimFreq_int)
  df$stimFreq_int_centred = df$stimFreq_int-mean(df$stimFreq_int, na.rm = TRUE)
  if (param == 'headLVL'){
    if (grepl('rl',df$headLVL[1])){
      df$param_int = lapply(df$headLVL, convert_char, elements = 3, removeType = 'start')
      df$param_int = as.numeric(df$param_int) 
      df$param_int = lapply(df$param_int, convert_rl)
      df$param_int = as.numeric(df$param_int) 
    } else if (grepl('deg',df$headLVL[1])){
      df$param_int = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
      df$param_int = as.numeric(df$param_int)
      df$param_int = lapply(df$param_int, convert_deg)
      df$param_int = as.numeric(df$param_int) 
    } 
  } else if (param == 'headHW'){
    df$param_int = df[['headHW']]
  } else if (param == 'snoutBodyAngle'){
    df$param_int = df[['snoutBodyAngle']]
  }
  
  outcome_variables = c('duration', 'latency', 'meanSpeed', 'maxSpeed', 'medianSpeed')
  if (param != ''){
    outcome_variables = append(outcome_variables, 'limbDist')
  } 
  
  for (i in 1:length(outcome_variables)){
    df$ov_centred = df[[outcome_variables[i]]]-mean(df[[outcome_variables[i]]], na.rm = TRUE)
    file_ext =  paste(outcome_variables[i], "_v_stimFreq_", param, appdx, sep = '')
    
    if (param != ""){
      df$param_int_centred = df$param_int-mean(df$param_int, na.rm = TRUE)
      
      if (grepl('COMBINED',appdx)){
        df$trialType = as.factor(df$trialType)
        modelLinear = lmer(ov_centred ~ stimFreq_int_centred + param_int_centred + trialType + (1|mouseID), data = df )
        modelQuadratic = lmer(ov_centred ~ stimFreq_int_centred + poly(param_int_centred,2, raw = TRUE) +  trialType +(1|mouseID), data = df )
        
      } else{
        modelLinear = lmer(ov_centred ~ stimFreq_int_centred + param_int_centred + (1|mouseID), data = df )
        modelQuadratic = lmer(ov_centred ~ stimFreq_int_centred + poly(param_int_centred,2, raw = TRUE) + (1|mouseID), data = df )
      } 
    }  else{
      #appdx = substr(appdx, 2, nchar(appdx))
      if (grepl('COMBINED',appdx)){
        df$trialType = as.factor(df$trialType)
        if (interaction == TRUE ) {
          file_ext = paste(outcome_variables[i], "_v_stimFreq_", param, "_interactionTRUE_", appdx, sep = '')
          modelLinear = lmer(ov_centred ~ stimFreq_int_centred *trialType + (1|mouseID), data = df )
          modelQuadratic = lmer(ov_centred ~ poly(stimFreq_int_centred,2, raw = TRUE) * trialType +(1|mouseID), data = df )
          
        } else{
          modelLinear = lmer(ov_centred ~ stimFreq_int_centred +trialType + (1|mouseID), data = df )
          modelQuadratic = lmer(ov_centred ~ poly(stimFreq_int_centred,2, raw = TRUE) + trialType +(1|mouseID), data = df )
        }
      } else{
        modelLinear = lmer(ov_centred ~ stimFreq_int_centred + (1|mouseID), data = df )
        modelQuadratic = lmer(ov_centred ~ stimFreq_int_centred + (1|mouseID), data = df )
      }
      
      AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
      write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_", file_ext, sep=""))
      write.csv(summary(model1)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext, sep=""))
      write.csv(summary(model2)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_", file_ext, sep=""))
    } 
    
    if (param != "" & outcome_variables[i] == 'limbDist'){
      AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
      write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_limbDist_v_", param, appdx, sep=""))
      write.csv(summary(model2)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_limbDist_v_", param, appdx, sep=""))
      
    } 
  }
}

## EXAMPLES ##
## head height dataset: locom params vs stim freq
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = '', appdx = "COMBINED", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))

## incline dataset: locom params vs stim freq and incline levels
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = 'headLVL', appdx = "incline", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))

## incline dataset: locom params vs stim freq and snout-hump angle
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = 'snoutBodyAngle', appdx = "incline", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))

## head height dataset: locom params vs stim freq and snout-hump angle
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = 'snoutBodyAngle', appdx = "", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))

## head height dataset:locom params vs stim freq and weight-adjusted head height
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = 'headHW', appdx = "", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))

## combined dataset: locom params vs stim freq, interaction = TRUE
# lme_locomParams_vs_stimFreq(yyyymmdd = '2022-08-18', param = '', appdx = "COMBINED", interaction = TRUE, outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParamsAcrossMice", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942', 'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'))


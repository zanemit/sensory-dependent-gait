source("C:\Users\MurrayLab\sensory-dependent-gait\lme\utils_R.R")

trdm_lme_bodyangles_vs_param <- function(yyyymmdd, appdx, param, outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParams", mouselist = FALSE){
  # This function fits a linear mixed effects model to snout-hump angle data
  # from the treadmill trials with param ("headHW" or "headLVL") as fixed 
  # effects and "mouseID" as a random effect.
  #     appdx should be "" for head height trials or "incline"
  #
  #     param "headHW" : weight-adjusted head height, relevant only to data from
  #                     head height trials
  #     param "headLVL" : re-centred head height in head height trials or 
  #                     corrected incline 
  #
  # The fitted model:
  #     snoutHumpAngle ~ param + 1|mouseID
  #
  # In each scenario, two models are fitted: one with param as a linear 
  # predictor and one with it as a quadratic predictor.
  #
  # Three files are saved:
  #     {yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_{param}{appdx}.csv
  #               summary of linear model coefficient
  #     {yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_{param}{appdx}.csv
  #               summary of quadratic model coefficients
  #     {yyyymmdd}_mixedEffectsModel_AICRsq_snoutBodyAngle_{param}{appdx}.csv
  #               AIC and R-squared values for both models
  
  appdx = get_appdx(appdx)
  
  filePath = paste(outputDir, yyyymmdd, filename, appdx, sep="")
  df <- read.csv(filePath)
  
  # KEEP ONLY THE MICE IN MOUSELIST 
  if (typeof(mouselist)=='character'){ 
    df = df[df$mouseID %in% mouselist,]
  }
  
  # CENTRE THE VARIABLES
  df$snoutBodyAngle_centred = df$snoutBodyAngle-mean(df$snoutBodyAngle, na.rm = TRUE)
  
  if (param == 'headLVL'){
    if (grepl('rl',df$headLVL[1])){
      df$headLVL_int = lapply(df$headLVL, convert_char, elements = 3, removeType = 'start')
      df$headLVL_int = as.numeric(df$headLVL_int)
      df$headLVL_int = lapply(df$headLVL_int, convert_rl)
    } else if (grepl('deg',df$headLVL[1])){
      df$headLVL_int = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
      df$headLVL_int = as.numeric(df$headLVL_int)
      df$headLVL_int = lapply(df$headLVL_int, convert_deg)
    }
    df$param_int = as.numeric(df$headLVL_int) 
  } else if (param == 'headHW'){
    df$param_int = as.numeric(df$headHW)
  } else if (param == 'trialType'){
    if (grepl('deg',df$trialType[1])){
      df$trialType_int = lapply(df$trialType, convert_char, elements = 4, removeType = 'start')
      df$trialType_int = as.numeric(df$trialType_int)
      df$trialType_int = lapply(df$trialType_int, convert_deg)
      df$param_int = as.numeric(df$trialType_int)
    } else{
      stop("trialType without deg is not defined!")
    }
  }
  
  df$param_int_centred = df$param_int-mean(df$param_int, na.rm = TRUE)
  
  modelLinear = lmer(snoutBodyAngle_centred ~ headLVL_int_centred + (1|mouseID), data = df )
  modelQuadratic = lmer(snoutBodyAngle_centred ~ poly(headLVL_int_centred,2, raw = TRUE) + (1|mouseID), data = df )
  
  AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
  write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_snoutBodyAngle_", param, appdx, sep=""))
  
  write.csv(summary(model21)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_snoutBodyAngle_", param, appdx, sep=""))
  write.csv(summary(model22)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_snoutBodyAngle_", param, appdx, sep=""))
  
}

## EXAMPLES ##
## incline dataset: snout-hump angle vs incline
# trdm_lme_bodyangles_vs_param(yyyymmdd = '2022-08-18', appdx = "incline", param = 'headLVL', outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParams", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869'))

## head height dataset: snout-hump angle vs weight-adjusted head height
# trdm_lme_bodyangles_vs_param(yyyymmdd = '2022-08-18', appdx = "", param = 'headHW', outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_locomParams", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869'))

## combined motorised treadmill dataset: snout-hump angles vs incline
# trdm_lme_bodyangles_vs_param(yyyymmdd = '2022-05-06', appdx = "", param = 'trialType', outputDir = "C:\\Users\\MurrayLab\\Documents\\MotorisedTreadmill\\", filename = "_strideParams_lH1.csv", mouselist= FALSE )


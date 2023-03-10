source("C:\Users\MurrayLab\sensory-dependent-gait\lme\utils_R.R")


fp_lme_limbPos_vs_param <- function(
                                    yyyymmdd, 
                                    limb, 
                                    param, 
                                    outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", 
                                    filename = "_limbPositionRegressionArray", 
                                    appdx = ""
                                    ){
  # This function fits a linear mixed effects model to limb position data
  # with param ("headHW" or "headLVL") as fixed effects and "mouseID" as a
  # random effect.
  #     param "headHW" : weight-adjusted head height, relevant only to data from
  #                     head height trials
  #     param "headLVL" : re-centred head height in head height trials or 
  #                     corrected incline 
  #
  # If filename contains "COMBINED":
  #     limbPosition ~ headLVL + headHeight + 1|mouseID 
  #
  # Otherwise:
  #     limbPosition ~ param + 1|mouseID
  #
  # In each scenario, two models are fitted: one with the first predictor 
  # (param or headLVL) as a linear predictor and one with it as a quadratic 
  # predictor.
  #
  # Three files are saved:
  #     {yyyymmdd}_mixedEffectsModel_linear_{limb}_{param}.csv
  #               summary of linear model coefficient
  #     {yyyymmdd}_mixedEffectsModel_quadratic_{limb}_{param}.csv
  #               summary of quadratic model coefficients
  #     {yyyymmdd}_mixedEffectsModel_AICRsq_{limb}_{param}.csv
  #               AIC and R-squared values for both models
  
  
  filePath = paste(outputDir, yyyymmdd, filename, appdx, ".csv",sep="")
  df <- read.csv(filePath)
  
  if (grepl('deg', df$level[1])){
    df$level = lapply(df$level, convert_char, elements = 4, removeType = 'start')
    df$level = as.numeric(df$level)
    df$level = lapply(df$level, convert_deg)
    df$level = as.numeric(df$level)
  }
  
  # CENTRE THE VARIABLES
  df$limbPos_centred = df[[limb]]-mean(df[[limb]], na.rm = TRUE)
  df$headLVL_centred = df$level-mean(df$level, na.rm = TRUE)
  
  if (grepl('COMBINED', filePath)){
    df$headHeight = as.factor(df$headHeight)
    modelLinear = lmer(limbPos_centred ~ headLVL_centred + headHeight + (1|mouseID), data = df )
    modelQuadratic = lmer(limbPos_centred ~ poly(headLVL_centred,2, raw = TRUE) + headHeight + (1|mouseID), data = df )
  } else if (param == 'headLVL'){
    modelLinear = lmer(limbPos_centred ~ headLVL_centred + (1|mouseID), data = df )
    modelQuadratic = lmer(limbPos_centred ~ poly(headLVL_centred,2, raw = TRUE) + (1|mouseID), data = df )
  } else if (param == 'headHW'){
    df$headHW_centred = df$headHW-mean(df$headHW, na.rm = TRUE)
    modelLinear = lmer(limbPos_centred ~ headHW_centred + (1|mouseID), data = df )
    modelQuadratic = lmer(limbPos_centred ~ poly(headHW_centred,2, raw = TRUE) + (1|mouseID), data = df )
  }
  
  write.csv(summary(modelLinear)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", limb, "_", param,".csv", sep=""))
  write.csv(summary(modelQuadratic)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_", limb, "_", param, ".csv", sep=""))
  
  AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
  write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_", limb, "_", param,".csv", sep=""))
}

## EXAMPLES ##
## combined incline dataset: rH position vs incline
# fp_lme_limbPos_vs_param(yyyymmdd = '2022-04-0x', limb = 'rH1x', param = 'headLVL',outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\",filename = "_limbPositionRegressionArray", appdx = "_COMBINED")

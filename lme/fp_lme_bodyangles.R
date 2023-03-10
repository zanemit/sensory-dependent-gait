source("C:\Users\MurrayLab\sensory-dependent-gait\lme\utils_R.R")

fp_lme_bodyangles_vs_param <- function(
                                    yyyymmdd, 
                                    param, 
                                    interaction = FALSE, 
                                    appdx = "",
                                    mouselist = FALSE,
                                    outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", 
                                    filename = "_forceplateAngleParamsR_headHW.csv"
                                    ){
  # This function fits a linear mixed effects model to snout-hump angle data
  # from the force sensor trials with param ("headHW" or "headLVL") as fixed 
  # effects and "mouseID" as a random effect.
  #     param "headHW" : weight-adjusted head height, relevant only to data from
  #                     head height trials
  #     param "headLVL" : re-centred head height in head height trials or 
  #                     corrected incline 
  #
  # If interaction is FALSE or if filename does not contain "COMBINED":
  #     snoutHumpAngle ~ param + 1|mouseID
  #
  # If interaction is TRUE and filename contains "COMBINED":
  #     snoutHumpAngle ~ headLVL * headHeight + 1|mouseID 
  #
  # If interaction is FALSE and filename contains "COMBINED":
  #     snoutHumpAngle ~ headLVL + headHeight + 1|mouseID
  #
  # In each scenario, two models are fitted: one with the first predictor 
  # (param or headLVL) as a linear predictor and one with it as a quadratic 
  # predictor.
  #
  # Three files are saved:
  #     {yyyymmdd}_mixedEffectsModel_linear_snoutBodyAngle_{param}_interaction{BOOL}.csv
  #               summary of linear model coefficient
  #     {yyyymmdd}_mixedEffectsModel_quadratic_snoutBodyAngle_{param}_interaction{BOOL}.csv
  #               summary of quadratic model coefficients
  #     {yyyymmdd}_mixedEffectsModel_AICRsq_snoutBodyAngle_{param}_interaction{BOOL}.csv
  #               AIC and R-squared values for both models
  appdx = get_appdx(appdx)
  
  filePath = paste(outputDir, yyyymmdd, filename, sep="")
  df <- read.csv(filePath)
  
  if (typeof(mouselist)=='character'){ 
    df = df[df$mouseID %in% mouselist,]
  }
  
  # CENTRE THE VARIABLES
  df$snoutBodyAngle_centred = df$snoutBodyAngle-mean(df$snoutBodyAngle, na.rm = TRUE)
  
  # HEADLVL DATA ARE ALREADY CONVERTED TO NON-RL AND NON-DEG!
  if (param == 'headLVL'){
    df$headLVL_int = as.numeric(df2$headLVL) 
  } else if (param == 'headHW'){
    df$headLVL_int = as.numeric(df2$headHW)
  } else{
    stop("Invalid param supplied!")
  }
  
  df$headLVL_centred = df$headLVL_int-mean(df$headLVL_int, na.rm = TRUE)
  file_ext = param
  
  if (grepl('COMBINED', filePath)){
    df$headHeight = as.factor(df$headHeight)
    if (interaction == TRUE){
      file_ext = paste(param, "_interactionTRUE", sep="")
      modelLinear = lmer(snoutBodyAngle_centred ~ headLVL_centred * headHeight + (1|mouseID), data = df )
      modelQuadratic = lmer(snoutBodyAngle_centred ~ poly(headLVL_centred, 2, raw = TRUE) * headHeight + (1|mouseID), data = df )
    } else{
      modelLinear = lmer(snoutBodyAngle_centred ~ headLVL_centred + headHeight + (1|mouseID), data = df )
      modelQuadratic = lmer(snoutBodyAngle_centred ~ poly(headLVL_centred, 2, raw = TRUE) + headHeight + (1|mouseID), data = df )
    }
  } else {
    modelLinear = lmer(snoutBodyAngle_centred ~ headLVL_centred + (1|mouseID), data = df )
    modelQuadratic = lmer(snoutBodyAngle_centred ~ poly(headLVL_centred, 2, raw = TRUE) + (1|mouseID), data = df )
  } 
  
  write.csv(summary(modelLinear)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_snoutBodyAngle_", file_ext,".csv", sep=""))
  write.csv(summary(modelQuadratic)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_quadratic_snoutBodyAngle_", file_ext, ".csv", sep=""))
  
  AICRsq_df = data.frame('Model' = c(rep('Linear',3), rep('Quadratic',3)), 'Metric' =  rep(c('R2marginal', 'R2total','AIC'),2), 'Value' = c(r.squaredGLMM(modelLinear)[1],r.squaredGLMM(modelLinear)[2],AIC(modelLinear),r.squaredGLMM(modelQuadratic)[1],r.squaredGLMM(modelQuadratic)[2],AIC(modelQuadratic)))
  write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_snoutBodyAngle_", file_ext,".csv", sep=""))
}

## EXAMPLES ##
## head height dataset: snout-hump angle vs weight-adjusted head height
# fp_lme_bodyangles_vs_param(yyyymmdd = '2021-10-26', param = 'headHW', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_headHW.csv")

## head height dataset: snout-hump angle vs absolute head height
# fp_lme_bodyangles_vs_param(yyyymmdd = '2021-10-26', param = 'headLVL', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_levels.csv")

## incline dataset (head low): snout-hump angle vs incline 
# fp_lme_bodyangles_vs_param(yyyymmdd = '2022-04-02', param = 'headLVL', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_levels.csv")

## incline dataset (head high): snout-hump angle vs incline 
# fp_lme_bodyangles_vs_param(yyyymmdd = '2022-04-04', param = 'headLVL', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_levels.csv")

## combined incline dataset: snout-hump angle vs incline, interaction FALSE
# fp_lme_bodyangles_vs_param(yyyymmdd = '2022-04-0x', param = 'headLVL', outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_levels_COMBINED.csv")

## combined incline dataset: snout-hump angle vs incline, interaction TRUE
# fp_lme_bodyangles_vs_param(yyyymmdd = '2022-04-0x', param = 'headLVL', interaction = TRUE, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", filename = "_forceplateAngleParamsR_levels_COMBINED.csv")

## treadmill pre-stimulus incline data
# fp_lme_bodyangles_vs_param(yyyymmdd = '2022-08-18', param = 'headLVL', appdx = "incline", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_bodyAngles_preOpto_", mouselist = c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 'FAA1034867', 'FAA1034868', 'FAA1034869','FAA1034942','FAA1034944','FAA1034945','FAA1034947','FAA1034949'))



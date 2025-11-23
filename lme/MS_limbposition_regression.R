### LIMB X POSITION REGRESSION

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

#--------------------------------------------------------

generate_mixed_effects_model_xpos <- function(yyyymmdd, ablationType, param = 'levels', data = 'preOpto', outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", filename = "_limbPositionRegressionArray"){
  if (ablationType == ""){
    appdx = ".csv"
  } else if (ablationType == "incline"){
    appdx = "_incline.csv"
  } else if (ablationType == "lcTeLC"){
    appdx = "_lcTeLC.csv"
  }
  
  if (data == 'mtTreadmill'){
    filepath = paste(outputDir, yyyymmdd, filename, appdx, sep="")
  }else if (data == 'forceplate'){
    filepath = paste(outputDir, yyyymmdd, filename, '.csv', sep = "")
  } else{
    filepath = paste(outputDir, yyyymmdd, filename, "_", data, "_", param, appdx, sep = "")
  }
  
  df <- read.csv(filepath)
  
  if (data == 'forceplate'){
    df$trialType = df$level
  }
  
  if (param == 'snoutBodyAngle'){
    df$param_centred = df$snoutBodyAngle - mean(df$snoutBodyAngle, na.rm = TRUE)
  } else if (param == 'headHW'){
    df$param_centred = df$headHW - mean(df$headHW, na.rm= TRUE)
  } else if (param == 'levels'){
    if (grepl('rl',df$trialType[1])){
      df$trialType_int = lapply(df$trialType, convert_char, elements = 3, removeType = 'start')
      df$trialType_int = as.numeric(df$trialType_int)
      df$trialType_int = lapply(df$trialType_int, convert_rl)
    } else if (grepl('deg',df$trialType[1])){
      df$trialType_int = lapply(df$trialType, convert_char, elements = 4, removeType = 'start')
      df$trialType_int = as.numeric(df$trialType_int)
      df$trialType_int = lapply(df$trialType_int, convert_deg)
    }
    df$trialType_int = as.numeric(df$trialType_int)
    df$param_centred = df$trialType_int - mean(df$trialType_int, na.rm = TRUE)
  } else{
    print("Invalid param supplied!")
  }
  
  # MODEL
  limbs =  c('lH1x', 'rH1x', 'lF1x', 'rF1x','lH1y', 'rH1y', 'lF1y', 'rF1y')
  if (data == 'forceplate'){
    limbs = c('rH1x', 'rH2x', 'rF1x', 'rF2x', 'rH1y', 'rH2y','rF1y', 'rF2y')
  }
  for (x in limbs){
    print(x)
    df$limb = df[[x]] - mean(df[[x]], na.rm = TRUE) # centred!!
    modelLIN = lmer(limb ~ param_centred + (1|mouseID), data = df )
    modelQDR = lmer(limb ~ poly(param_centred,2, raw = TRUE) + (1|mouseID), data = df )
    if (x == limbs[1]){
      dfOutputLIN = summary(modelLIN)$coefficients
      dfOutputQDR = summary(modelQDR)$coefficients
      rowsLIN = c(paste("Intercept_", x, sep = ""), paste("param_centred_", x, sep = ""))
      rowsQDR = c(paste("Intercept_", x, sep = ""), paste("param_centred_1_", x, sep = ""), paste("param_centred_2_", x, sep = ""))
      
      values = c(r.squaredGLMM(modelLIN)[1],r.squaredGLMM(modelLIN)[2],AIC(modelLIN), r.squaredGLMM(modelQDR)[1],r.squaredGLMM(modelQDR)[2],AIC(modelQDR))
      model = c(rep(paste('Linear_',x,sep=""),3), rep(paste('Quadratic_',x,sep=""),3))
      
    } else{
      dfOutputLIN = rbind(dfOutputLIN, summary(modelLIN)$coefficients)
      dfOutputQDR = rbind(dfOutputQDR, summary(modelQDR)$coefficients)
      rowsLIN = append(rowsLIN,c(paste("Intercept_", x, sep = ""), paste("param_centred_", x, sep = "")))
      rowsQDR = append(rowsQDR, c(paste("Intercept_", x, sep = ""), paste("param_centred_1_", x, sep = ""), paste("param_centred_2_", x, sep = "")))
      
      values = append(values, c(r.squaredGLMM(modelLIN)[1],r.squaredGLMM(modelLIN)[2],AIC(modelLIN),r.squaredGLMM(modelQDR)[1],r.squaredGLMM(modelQDR)[2],AIC(modelQDR)))
      model = append(model, c(rep(paste('Linear_',x,sep=""),3), rep(paste('Quadratic_',x,sep=""),3)))
    }
    print(anova(modelLIN), sep = "")
  }
  rownames(dfOutputLIN) = rowsLIN
  rownames(dfOutputQDR) = rowsQDR
  write.csv(dfOutputLIN, paste(outputDir, yyyymmdd, "_limbPositionRegressionArray_MIXEDMODEL_linear_", data,"_", param, appdx, sep=""))
  write.csv(dfOutputQDR, paste(outputDir, yyyymmdd, "_limbPositionRegressionArray_MIXEDMODEL_quadratic_", data,"_", param, appdx, sep=""))
  
  metric =  rep(c('R2marginal', 'R2total','AIC'),2*length(limbs))
  AICsq_df = data.frame('Model'  = model, 'Metric' = metric, 'Value' = values)
  write.csv(AICsq_df, paste(outputDir, yyyymmdd, "_limbPositionRegressionArray_MIXEDMODEL_AIC_", data,"_", param, appdx, sep=""))
}  

# Fig S6C
generate_mixed_effects_model_xpos(yyyymmdd = '2022-08-18', 
                                  ablationType = 'incline', 
                                  param = 'levels', 
                                  data = 'preOpto', 
                                  outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                                  filename = "_limbPositionRegressionArray")

generate_mixed_effects_model_xpos(yyyymmdd = '2022-04-04', 
                                  ablationType = '', 
                                  param = 'levels', 
                                  data = 'forceplate', 
                                  outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", 
                                  filename = "_limbPositionRegressionArray")


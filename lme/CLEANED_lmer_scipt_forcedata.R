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

# param can be 'levels', 'snoutBodyAngle', 'headHW'
generate_mixed_effects_model_forceplate <- function(yyyymmdd, 
                                                    indep_var, 
                                                    predictors,
                                                    interaction = FALSE, # relevant only when there are multiple predictors at once
                                                    pc_num = FALSE, 
                                                    slope=FALSE,
                                                    outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\"
){
  
  if(grepl('x', yyyymmdd)){
    filePath = paste(outputDir, yyyymmdd, "_meanParamDF_", indep_var, "_COMBINED.csv", sep="")
  } else{
    filePath = paste(outputDir, yyyymmdd, "_meanParamDF_", indep_var, ".csv", sep="")
  }
  df4 <- read.csv(filePath)
  
  # IF PC, ADD THE PC_NUM
  if (grepl('PC', indep_var)){
    n = unlist(gregexpr("PC", indep_var))+2
    indep_var = paste(substr(indep_var,1,n-1), pc_num, substr(indep_var,n,nchar(indep_var)), sep="")
  }
  
  # CENTRE THE VARIABLES
  df4$indep_var_centred = df4$indep_var-mean(df4$indep_var, na.rm = TRUE) # headHW, snoutBodyAngle, or levels (rl or deg depending on yyyymmdd)
  
  if(grepl('posX', indep_var)){
    df4$indep_var_centred = df4$indep_var_centred/76.2
  }
  
  predictor_str_dict = list(
    'hind_frac' = 'hindfrac',
    'fore_frac' = 'forefrac',
    'CoMx_mean' = 'CoMx',
    'CoMy_mean' = 'CoMy',
    'total_pressure' = 'tpressure',
    'fore_weight_frac' = 'foreWfrac',
    'hind_weight_frac' = 'hindWfrac',
    'headplate_weight_frac' = 'headWfrac'
  )
  predictor_strs = predictor_str_dict[predictors]
  
  for (i in 1:length(predictors)){
    if (predictors[i]=='headplate_weight_frac'){
      df4[[predictors[i]]] = -(df4[[predictors[i]]]-1)*100
    }
    df4$pred_centred = df4[[predictors[i]]] - mean(df4[[predictors[i]]], na.rm=TRUE)
    file_ext = paste(predictor_strs[i], "_", indep_var, sep='')
    if (grepl('COMBINED', filePath)){
      df4$headHeight = as.factor(df4$headHeight)
      df4$mouseID = df4$mouse
      
      # INTERACTION TRUE
      if (interaction == TRUE){
        file_ext = paste(predictor_strs[i], "_headHeight_interactionTRUE_", sep='')
        modelLinearSlope1 = lmer(pred_centred ~ indep_var_centred * headHeight + (headHeight|mouseID), data = df4 )
        modelLinearSlope12 = lmer(pred_centred ~ indep_var_centred * headHeight + (indep_var_centred+headHeight|mouseID), data = df4 )
        modelLinearIntercept = lmer(pred_centred ~ indep_var_centred * headHeight + (1|mouseID), data = df4 )
      } else{
        file_ext = paste(predictor_strs[i], "_headHeight_interactionTRUE_", sep='')
        modelLinearSlope1 = lmer(pred_centred ~ indep_var_centred + headHeight + (headHeight|mouseID), data = df4 )
        modelLinearSlope12 = lmer(pred_centred ~ indep_var_centred + headHeight + (indep_var_centred+headHeight|mouseID), data = df4 )
        modelLinearIntercept = lmer(pred_centred ~ indep_var_centred + headHeight + (1|mouseID), data = df4 )
      }
      
      # CREATE A LIST OF MODELS
      models = list(modelLinearIntercept, modelLinearSlope1, modelLinearSlope12)
      model_descriptions = list("randInt", "randSlope1", "randSlope12")
      
    } else{
      modelLinearSlope1 = lmer(pred_centred ~ indep_var_centred + (indep_var_centred|mouse), data = df4 )
      modelLinearIntercept = lmer(pred_centred ~ indep_var_centred + (1|mouse), data = df4 )
      
      # CREATE A LIST OF MODELS
      models = list(modelLinearIntercept, modelLinearSlope1)
      model_descriptions = list("randInt", "randSlope1")
      
    }
    
    # IF SLOPE MODEL IS ENFORCED
    if (slope==TRUE){
      best_model = modelLinearSlope1
      file_ext = paste0(file_ext, "slopeENFORCED_")
    } else{
      # DETERMINE WHICH MODEL IS THE BETTER ONE
      AIC_list = sapply(models, AIC)
      best_idx = which.min(AIC_list)
      best_model = models[[best_idx]]
      best_model_str = model_descriptions[[best_idx]]
      cat("Best model for", predictor_strs[[i]] , "is number", best_idx, "with AIC", AIC_list[best_idx], "\n")
    }
    
    write.csv(summary(best_model)$coefficients, paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_", file_ext, best_model_str, ".csv", sep=""))
    
    if (grepl('COMBINED', filePath)){
      mouse_str = "mouseID"
    } else{
      mouse_str = "mouse"
    }
    write.csv(ranef(best_model)[[mouse_str]], paste(outputDir, yyyymmdd, "_mixedEffectsModel_linear_randomEffects_", file_ext, best_model_str,".csv", sep=""))
    
    AICRsq_df = data.frame('Metric' = c('R2marginal', 'R2total','AIC'), 
                           'Value' = c(r.squaredGLMM(best_model)[1],r.squaredGLMM(best_model)[2],AIC(best_model))
    )
    write.csv(AICRsq_df, paste(outputDir, yyyymmdd, "_mixedEffectsModel_AICRsq_", file_ext, best_model_str, ".csv", sep=""))
    
  }
}

# predictors = c('hind_frac', 'fore_frac', 'CoMx_mean', 'CoMy_mean', 'total_pressure', 'fore_weight_frac', 'hind_weight_frac', 'headplate_weight_frac')

# Figure S1B, S1E
generate_mixed_effects_model_forceplate(yyyymmdd='2021-10-26', 
                                        indep_var='headHW', 
                                        predictors=c('hind_weight_frac', 'CoMx_mean'),
                                        interaction=FALSE, 
                                        pc_num=FALSE, 
                                        slope=TRUE,
                                        outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

# Figure S1G, S1H, 1F, 1G
generate_mixed_effects_model_forceplate(yyyymmdd='2022-04-04', 
                                        indep_var='levels', 
                                        predictors=c('fore_weight_frac', 'hind_weight_frac', 'CoMy_mean', 'headplate_weight_frac'),
                                        interaction=FALSE, 
                                        pc_num=FALSE, 
                                        slope=TRUE,
                                        outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

# Figure 1D, 1E
generate_mixed_effects_model_forceplate(yyyymmdd='2021-10-26', 
                                        indep_var='snoutBodyAngle', 
                                        predictors=c('CoMy_mean', 'headplate_weight_frac'),
                                        interaction=FALSE, 
                                        pc_num=FALSE, 
                                        slope=TRUE,
                                        outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\")

generate_simple_LINEAR_mixed_effects_models(yyyymmdd = '2022-08-18', 
                                            indep_var = 'stimFreq',
                                            dep_var = 'maxSpeed',
                                            filename = "locomParamsAcrossMice",
                                            outputDir = "C:\\Users\\MurrayLab\\Documents\\passiveOptoTreadmill\\")

trdm_circular_glm <- function(
                              yyyymmdd, 
                              limb, 
                              cont_predictors, 
                              cat_predictors, 
                              sBA_split = FALSE, 
                              interaction = TRUE, 
                              interaction_type = "", 
                              param_type = 'categorical', 
                              iters = 10000, 
                              sample_frac = 0.2, 
                              outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                              filename = "_strideParams_incline_", 
                              refLimb = 'lH1', 
                              mice = FALSE
                              ){
  # This function fits a Bayesian circular mixed effects model 
  # (https://search.r-project.org/CRAN/refmans/bpnreg/html/bpnme.html) to
  # various {cont_predictors}, {cat_predictors} as fixed effects and mouseID as 
  # a random effect. 
  #     {limb} should be "lF0", "rF0", "lH0", "rH0" for single-refLimb datasets
  #             and "homologous0", "homolateral0", "diagonal0" for combined 
  #             refLimb datasets
  #     {cont_predictors} can be any colname in the dataset
  #     {cat_predictors} can be 'sex', 'refLimb', 'snoutBodyAngle', 'incline'
  #     {sBA_split} is necessary only if {cat_predictors} include 'snoutBodyAngle'
  #                 and one wishes to pre-define the categories
  #     {interaction} : a bool determining the presence of an interaction
  #                     between variables specified by {interaction_type} 
  #     {interaction_type} : if"", then this interaction is between pred1 and 
  #                     pred2; if {interaction_type} is "secondary", then 
  #                     interaction is between pred2 and pred3; if "threeway",
  #                     then the interaction is between pred1, pred2, pred3                
  #     {param_type} : 
  #     {iters} specify the number of iterations in the model fitting process
  #     {sample} : the fraction of datapoints used in model fitting
  #               (around 10k datapoints is usually reasonable)
  #     {mice} : list of mouseIDs to include in the model
  #
  #
  # The fitted model:
  #     {limb}phase ~ cont_predictor{1} + ... + cont_predictor{N} + cat_predictor1 + 1|mouseID
  #
  # The code has not been tested with more than two cat_predictors and allows 
  # a maximum of five predictors overall.
  #
  # Three files are saved:
  #     {yyyymmdd}_beta1_{file_ext}.csv
  #               beta1 values of all continuous predictors and/or categorical 
  #               predictor pairs
  #     {yyyymmdd}_beta2_{file_ext}.csv
  #               beta2 values of all continuous predictors and/or categorical 
  #               predictor pairs
  #     {yyyymmdd}_WAIC_{file_ext}.csv
  #               model DIC and WAIC values
  #     {yyyymmdd}_coefCircContinuous_{file_ext}.csv
  #               coefficients of the continuous predictors
  #     {yyyymmdd}_randOutputs_{file_ext}.csv
  #                posterior samples of the mean resultant length of the circular 
  #                random intercept for each mouse
  #     {yyyymmdd}_coefCircCategorical_{file_ext}.csv
  #               coefficients of the categorical predictors
  
  library(bpnreg)
  filePath = paste(outputDir, yyyymmdd, filename, refLimb, ".csv", sep="")
  df <- read.csv(file = filePath)
  
  # REFORMAT THE DATAFRAME
  # convert phase to radians [0, 2pi]
  df$limb = (df[[limb]]*2*pi)
  limb_bool = df$limb<0
  limb_bool[which(is.na(limb_bool))] = FALSE
  df$limb[limb_bool] = df$limb[limb_bool]+(2*pi)
  
  # deal with incline (if present)
  if ('incline' %in% cont_predictors | 'incline' %in% cat_predictors){
    df$incline = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
    df$incline = as.numeric(df$incline)
    df$incline = lapply(df$incline, convert_deg)
    df$incline = as.numeric(df$incline)
  }
  
  # centre continuous predictors
  for (pred in cont_predictors){
    df[paste(pred, '_centred', sep="")] = df[[pred]] - mean(df[[pred]], na.rm = TRUE)
  }
  
  # make mouse IDs numeric - necessary for random effect testing
  if (typeof(mice) != 'logical'){
    df = df[df$mouseID %in% mice,]
  }
  mice = unique(df$mouseID)
  df$mouseIDnum = df$mouseID
  for (i in 1:length(mice)){
    mouse = mice[i]
    df$mouseIDnum[df$mouseID == mouse] <- i
  }
  df$mouseIDnum = as.numeric(df$mouseIDnum)
  
  # isolate columns as vector - for some reason, this is needed for bnpme to work
  limbx = df$limb
  mousex = df$mouseIDnum
  pred1 <- df[[paste(cont_predictors[1], '_centred', sep="")]]
  df_new <- data.frame(limb = limbx, mouse =mousex, pred1 = pred1)
  for (k in 2:length(cont_predictors)){
    predk = df[[paste(cont_predictors[k], '_centred', sep="")]]
    df_new[paste("pred",k,sep="")] <- predk
    }
  
  predictors = cont_predictors
  
  # deal with categorical predictors
  if (length(cat_predictors) > 0){
    if ('sex' %in% cat_predictors){
      df$sex = as.factor(df$sex) #some problems - only has one level!
      sexx = df$sex
      df_new[paste("pred", length(predictors)+1, sep="")] = sexx
      predictors = append(predictors, 'sex')
    }
    if ('refLimb' %in% cat_predictors){ 
      df$refLimb = as.factor(df$refLimb) #some problems - only has one level!
      reff = df$refLimb
      df_new[paste("pred", length(predictors)+1, sep="")] = reff
      predictors = append(predictors, 'refLimb')
    }
    if ('snoutBodyAngle' %in% cat_predictors){
      # split snout-body angle into 5 categories (6 borders)
      df$snoutBodyAngle_centred = df[['snoutBodyAngle']] - mean(df[['snoutBodyAngle']], na.rm = TRUE)
      snoutBodyAnglex = df$snoutBodyAngle_centred
      sBA_noOutliers = remove_outliers(snoutBodyAnglex)
      library(pracma)
      if (typeof(sBA_split) == 'logical'){
        sBA_split_str = sBA_split
        sBA_split = linspace(min(sBA_noOutliers, na.rm = TRUE), max(sBA_noOutliers, na.rm = TRUE), 6)
      }  else{
        sBA_split_str = paste(sBA_split, collapse="-")
        sBA_split = sBA_split - mean(df[['snoutBodyAngle']], na.rm = TRUE)
      } #sBA_split is the entered vector
      
      # turn snout-body angle into a categorical predictor
      sBA_categorical = snoutBodyAnglex
      sBA_categorical[snoutBodyAnglex <= sBA_split[1] | snoutBodyAnglex > sBA_split[length(sBA_split)]] = NaN
      for (i in 1:(length(sBA_split)-1)){
        sBA_categorical[snoutBodyAnglex > sBA_split[i] & snoutBodyAnglex <= sBA_split[i+1]] = i
      }
      for (i in length(predictors):1){
        df_new[paste("pred", i+1, sep="")] = df_new[paste("pred", i, sep="")]
      }
      df_new$pred2 = sBA_categorical
      predictors = append(predictors, predictors[2])
      predictors[2] = 'snoutBodyAngle'
    }  else{
      sBA_split_str = sBA_split
    }
    
    if ('incline'%in% cat_predictors){
      incline_split = linspace(min(df$incline, na.rm = TRUE), max(df$incline, na.rm = TRUE), 6)
      
      # turn incline into a categorical predictor
      inclinex = df$incline
      incline_categorical = inclinex
      incline_categorical[inclinex < incline_split[1] | inclinex > incline_split[6]] = NaN
      for (i in 1:(length(incline_split)-1)){
        if (i == 1){
          incline_categorical[inclinex >= incline_split[i] & inclinex <= incline_split[i+1]] = i
        } else{
          incline_categorical[inclinex > incline_split[i] & inclinex <= incline_split[i+1]] = i
        }
      }
      if (length(cont_predictors)>1 & 'snoutBodyAngle' %in% cat_predictors){
        df_new[paste("pred", length(cont_predictors)+2, sep="")] = pred3
        df_new$pred3 = incline_categorical
        df_new = na.omit(df_new)
        df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
        df_new$pred3 = factor(df_new$pred3, ordered = FALSE)
        predictors = append(predictors, predictors[3])
        predictors[3] = 'incline'
      }  else if (length (cont_predictors)>1){
        df_new[paste("pred", length(cont_predictors)+1, sep="")] = pred2
        df_new$pred2 = incline_categorical
        df_new = na.omit(df_new)
        df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
        predictors = append(predictors, predictors[2])
        predictors[2] = 'incline'
      } else if ('snoutBodyAngle' %in% cat_predictors){
        df_new$pred3 = incline_categorical
        predictors = append(predictors, predictors[3])
        predictors[3] = 'incline'
        df_new = na.omit(df_new)
        df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
        df_new$pred3 = factor(df_new$pred3, ordered = FALSE)
      } else{ # only one cont predictor and snoutBodyAngle not in cat_predictors
        df_new$pred2 = incline_categorical
        df_new = na.omit(df_new)
        df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
        predictors = append(predictors, predictors[2])
        predictors[2] = 'incline'
      }
    } else{ # no 'incline' in cat_predictors
      df_new = na.omit(df_new)
      if ('snoutBodyAngle' %in% cat_predictors){
        df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
      }
      #     if (('refLimb' %in% cat_predictors) == FALSE){
      #       df_new$pred2 = factor(df_new$pred2, ordered = FALSE)
      #     }
    }
  } else{
    sBA_split_str = sBA_split
  }
  
  # SUBSAMPLE THE DATAFRAME
  subsample_size = as.integer(nrow(df_new)*sample_frac)
  df_new_sample = df_new[sample(nrow(df_new), subsample_size),]
  df_new_sample = na.omit(df_new_sample)
  
  
  if (length(cont_predictors) + length(cat_predictors) == 1){
    model = bpnme(pred.I = limb ~ pred1 + (1|mouse),  data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
    file_ext = paste(limb, "_ref", refLimb, "_", predictors[1], "_interaction", interaction, "_", param_type, "_randMouse_","sBAsplit", sBA_split_str, "_",  sample_frac, "data", subsample_size, "s_", iters, "its_100burn_3lag.csv", sep="")
  }   else if (length(cont_predictors) + length(cat_predictors) == 2){
    file_ext = paste(limb, "_ref", refLimb, "_", predictors[1],"_", predictors[2], "_interaction", interaction, "_", param_type, "_randMouse_", "sBAsplit", sBA_split_str, "_", sample_frac, "data", subsample_size, "s_", iters, "its_100burn_3lag.csv", sep="")
    
    if(interaction == TRUE){
      model = bpnme(pred.I = limb ~ pred1 * pred2 + (1|mouse),   data = df_new_sample,its = iters, burn = 100, n.lag = 3, seed = 101)
    }    else{
      model = bpnme(pred.I = limb ~ pred1 + pred2 + (1|mouse), data = df_new_sample,its = iters, burn = 100, n.lag = 3, seed = 101)
    }
  }  else if (length(cont_predictors) + length(cat_predictors) == 3){
    file_ext = paste(limb, "_ref", refLimb, "_", predictors[1],"_", predictors[2],"_", predictors[3], "_interaction", interaction, interaction_type, "_", param_type, "_randMouse_", "sBAsplit", sBA_split_str, "_", sample_frac, "data", subsample_size, "s_", iters, "its_100burn_3lag.csv", sep="")
    
    if(interaction == TRUE){
      if (interaction_type == 'threeway'){
        model = bpnme(pred.I = limb ~ pred1 * pred2 * pred3 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      } else if (interaction_type == 'secondary') {
        model = bpnme(pred.I = limb ~ pred1 + pred2 * pred3 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
        
      } else{
        model = bpnme(pred.I = limb ~ pred1 * pred2 + pred3 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      }
    }    else{
      model = bpnme(pred.I = limb ~ pred1 + pred2 + pred3 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
    }
  }  else if (length(cont_predictors) + length(cat_predictors) == 4){
    file_ext = paste(limb, "_ref", refLimb, "_", predictors[1],"_", predictors[2],"_", predictors[3],"_", predictors[4], "_interaction", interaction, interaction_type, "_", param_type, "_randMouse_", "sBAsplit", sBA_split_str, "_", sample_frac, "data", subsample_size, "s_", iters, "its_100burn_3lag.csv", sep="")
    
    if(interaction == TRUE){
      if (interaction_type == 'threeway'){
        model = bpnme(pred.I = limb ~ pred1 * pred2 * pred3 + pred4 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      } else if (interaction_type == 'secondary') {
        model = bpnme(pred.I = limb ~ pred1 + pred2 * pred3 + pred4 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
        
      }else{
        model = bpnme(pred.I = limb ~ pred1 * pred2 + pred3 + pred4 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      }
    }    else{
      model = bpnme(pred.I = limb ~ pred1 + pred2 + pred3 + pred4 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
    }
  }  else if (length(cont_predictors) + length(cat_predictors) == 5){
    file_ext = paste(limb, "_ref", refLimb, "_", predictors[1],"_", predictors[2],"_", predictors[3],"_", predictors[4],"_", predictors[5], "_interaction", interaction, interaction_type, "_", param_type, "_randMouse_", "sBAsplit", sBA_split_str, "_", sample_frac, "data", subsample_size, "s_", iters, "its_100burn_3lag.csv", sep="")
    
    if(interaction == TRUE){
      if (interaction_type == 'threeway'){
        model = bpnme(pred.I = limb ~ pred1 * pred2 * pred3 + pred4 + pred5 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      } else if (interaction_type == 'secondary') {
        model = bpnme(pred.I = limb ~ pred1 + pred2 * pred3 + pred4 +pred5+ (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
        
      } else{
        model = bpnme(pred.I = limb ~ pred1 * pred2 + pred3 + pred4 + pred5 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
      }
    }    else{
      model = bpnme(pred.I = limb ~ pred1 + pred2 + pred3 + pred4 + pred5 + (1|mouse), data = df_new_sample, its = iters, burn = 100, n.lag = 3, seed = 101)
    }
  }  else{
    print("TOO MANY PREDICTORS SUPPLIED! NO MODEL WILL BE FIT!")
  }
  
  write.csv(model$beta1, paste(outputDir, yyyymmdd, "_beta1_", file_ext, sep=""))
  write.csv(model$beta2, paste(outputDir, yyyymmdd, "_beta2_", file_ext, sep=""))
  write.csv(fit(model), paste(outputDir, yyyymmdd, "_WAIC_", file_ext, sep=""))
  write.csv(coef_circ(model, type = 'continuous'), paste(outputDir, yyyymmdd, "_coefCircContinuous_", file_ext, sep=""))
  columns = c(c('cRI mean', 'cRI variance'), paste("mouse", unique(mousex)))
  values = c(mean(model$cRI), var(model$cRI), apply(model$circular.ri, 1, mean_circ))
  write.csv(data.frame(values, colnames = columns), paste(outputDir, yyyymmdd, "_randOutputs_", file_ext, sep=""))
  
  if (length(cat_predictors)>0 & isFALSE(is.ordered(df_new_sample$pred2))){
    write.csv(coef_circ(model, type = 'categorical')$Differences, paste(outputDir, yyyymmdd, "_coefCircCategorical_", file_ext, sep=""))
    write.csv(coef_circ(model, type = 'categorical')$Means, paste(outputDir, yyyymmdd, "_coefCircCategorical_MEANS_", file_ext, sep=""))
  }
  plot(1:iters, model$beta1[,'pred1'])
  plot(1:iters, model$beta2[,'pred1'])
}

library(bpnreg)
library(stringr)

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

get_limb_dict <- function(refLimb){
  if (grepl('lH', refLimb)){
    limb_dict <- c("lH"="ref", "rH"="homologous", "lF"="homolateral", "rF"="diagonal")
  } else if (grepl('rH', refLimb)){
    limb_dict <- c("lH"="homologous", "rH"="ref", "lF"="diagonal", "rF"="homolateral")
  } else if (grepl('lF', refLimb)){
    limb_dict <- c("lH"="homolateral", "rH"="diagonal", "lF"="ref", "rF"="homologous")
  } else if (grepl('rF', refLimb)){
    limb_dict <- c("lH"="diagonal", "rH"="homolateral", "lF"="homologous", "rF"="ref")
  }
}

perform_GLM_incline <- function(yyyymmdd, 
                                limb, 
                                cont_predictors, 
                                cat_predictors, 
                                sBA_split = FALSE,
                                slope = FALSE,
                                slope_type = "pred2",
                                interaction = TRUE, 
                                interaction_type = "", 
                                param_type = 'categorical', 
                                iters = 10000, 
                                sample_frac = 0.2, 
                                outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                                filename = "_strideParams_incline_", 
                                refLimb = 'lH1', 
                                nonref_subset_interval_str = c('blncd'), #'alt'
                                mice = FALSE){
  
  # load data
  filePath = paste(outputDir, yyyymmdd, filename, refLimb, ".csv", sep="")
  df <- read.csv(file = filePath)
  
  # FLIP LIMB
  if (((limb == "homologous0") | (limb == "rF0" & refLimb == 'lF1') | (limb == "rH0" & refLimb == 'lH1'))&(!grepl('MotorisedTreadmill', outputDir))){
    left_inj = unique(c("FAA1034870","FAA1034942", "FAA1034947",
                        "FAA1034944", "FAA1034945", "FAA1034948", "FAA1034949", "FAA1034950",
                        "FAA1035017", "FAA1035018", "FAA1035021", "FAA1035022", "FAA1035103",
                        "FAA1035067", "FAA1035065", "FAA1035066", 'FAA1035414', 'FAA1035415',
                        'FAA1035416', 'FAA1035420', 'FAA1035504', 'FAA1035567', 'FAA1035568',
                        'FAA1035561', 'FAA1035562', 'FAA1035599', 'FAA1035562', 'FAA1035604',
                        'FAA1035602', 'FAA1035603', 'FAA1035691', 'FAA1035607', 'FAA1035610',
                        'FAA1035696', 'FAA1035611', 'FAA1035612', 'FAA1035636', 'FAA1035656', 
                        'FAA1035670', 'FAA1035684', 'FAA1035694', 'FAA1035699', 'FAA1035690'))
    right_inj = unique(c('BAA1098955',  'BAA1099004', 'FAA1034468', 'FAA1034469', 'FAA1034471', 
                         'FAA1034472', 'FAA1034570', 'FAA1034572', 'FAA1034573', 'FAA1034575', 
                         'FAA1034576', 'FAA1034665', 'FAA1034821', 'FAA1034827', 'FAA1034836', 
                         'FAA1034839', 'FAA1034833', 'FAA1034825', 'BAA1098921', 'FAA1034823', 
                         'FAA1034824', 'BAA1098923', 'BAA1099003', 'FAA1034470', 'BAA1098923',
                         'BAA1099270', 'FAA1034509', 'FAA1034510', "FAA1034825", "FAA1034827",
                         "FAA1034835", "FAA1034821", "FAA1034823", "FAA1034824", "FAA1034833",
                         "FAA1034836", "FAA1034839", "FAA1034841", "FAA1034842", "FAA1034868",
                         "FAA1034869", "FAA1034858", "FAA1034861", "FAA1034906", "FAA1034907",
                         "BAA1101344", "BAA1101343", "FAA1034943", "FAA1034946", "FAA1035062",
                         "FAA1035063", "FAA1035064", "FAA1035107", "FAA1035108", 'FAA1035413',
                         'FAA1035417', 'FAA1035418', 'FAA1035419', 'FAA1035421', 'FAA1035501',
                         'FAA1035503', 'FAA1035523', 'FAA1035526', 'FAA1035528', 'FAA1035505',
                         'CAA1120310', 'FAA1035563', 'FAA1035572', 'FAA1035571', 'FAA1035569',
                         'FAA1035600', 'FAA1035601', 'FAA1035608', 'FAA1035609', 'FAA1035613',
                         'FAA1035620', 'FAA1035638', 'FAA1035700', 'FAA1035671', 'FAA1035676'))#,
    
    
    dfR = df[df$mouseID %in% right_inj,]
    dfL = df[df$mouseID %in% left_inj,]
    if (limb == "homologous0"){
      dfL$homologous0 = dfL$homologous0*-1
    } else if (limb == "rF0" & refLimb == 'lF1'){
      dfL$rF0 = dfL$rF0*-1  # assuming injection on the right side
    } else if (limb == "rH0" & refLimb == 'lH1'){
      dfL$rH0 = dfL$rH0*-1  # assuming injection on the right side
    } else{
      print("Invalid combination of limb and refLimb parameters")
    }
    df = rbind(dfR, dfL)
  }
  
  # deal with PC in refLimb
  if (grepl("PC", refLimb)){
    n = unlist(gregexpr("PC", refLimb))[1]
    refLimb = substr(refLimb, 1, n-2)
  }
  
  # DEAL WITH INCLINE (if present)
  if ('incline' %in% cont_predictors | 'incline' %in% cat_predictors){
    if (grepl('Motorised', outputDir)&(length(cat_predictors)>0)){
      if (grepl('trialType', cat_predictors)){
        if (grepl('deg', df$trialType[1])){
          df$incline = lapply(df$trialType, convert_char, elements = 4, removeType = 'start')
        } else{
          df$incline = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
        }
      }
    } else{
      df$incline = lapply(df$headLVL, convert_char, elements = 4, removeType = 'start')
    }
    df$incline = as.numeric(df$incline)
    df$incline = lapply(df$incline, convert_deg)
    df$incline = as.numeric(df$incline)
  }
  
  
  # DEAL WITH NON-REF LIMB
  if (refLimb!='COMBINED'){  # HOMOLATERAL coordination is studied, nonref is homologous
    if(substr(limb,1,1)==substr(refLimb,1,1)){
      prefix = ifelse(grepl('r', refLimb), 'l', 'r')
      nonrefLimb = paste0(prefix, substr(refLimb, nchar(refLimb)-1, nchar(refLimb)-1), '0')
    } else{  # LEFT-RIGHT coordination is studied, nonref is homologous
      prefix = substr(refLimb,1,1)
      nonrefLimb = paste0(prefix, setdiff(c('H','F'),substr(limb, nchar(limb)-1, nchar(limb)-1)), '0')
    }
  } else if(substr(limb,1,1)==substr(refLimb,1,1)){ # HOMOLATERAL coordination is studied, nonref is homologous
    nonrefLimb='homologous0'
  } else{ # RIGHT-LEFT coordination is studied
    nonrefLimb='homolateral0'
  }
  
  # balance data and/or combine data across limb phase categories
  if (!identical(nonref_subset_interval_str, c(''))){
    if (any(c('advanced', 'delayed', 'alt', 'sync') %in% nonref_subset_interval_str)){
      df = df[df[[paste0(nonrefLimb, '_categorical')]] %in% nonref_subset_interval_str, ]
    }    else{
      df = df[df[[paste0(nonrefLimb, '_categorical')]] %in% c('advanced', 'delayed', 'alt', 'sync'), ]
    }
    
    if (any(grepl('blncd', nonref_subset_interval_str))){ 
      # always process advanced and delayed separately
      df_nonan = df[!df[[paste0(nonrefLimb, '_categorical')]]=='',]
      df_nonan = df_nonan[complete.cases(df_nonan[,c(limb,cont_predictors)]),]
      df_balanced = data.frame()
      set.seed(42)
      
      for(mouse in mice){
        df_mouse = df_nonan[df_nonan$mouseID==mouse, ]
        
        # skip if no valid data for this mouse
        if (nrow(df_mouse)==0){
          cat("Skipping mouse", mouse, "(no valid rows)\n")
          next
        }
        
        unique_groups = unique(df_mouse[[paste0(nonrefLimb, '_categorical')]])
        
        # skip mice where categories are absent
        group_counts = table(df_mouse[[paste0(nonrefLimb, '_categorical')]])
        if (length(group_counts)==0){
          cat("Skipping mouse", mouse, "(no categories)\n")
          next
        }
        
        min_group_size = as.integer(min(table(df_mouse[[paste0(nonrefLimb, '_categorical')]]))*sample_frac)
        
        # skip if min_group_size becomes NA or zero
        if (is.na(min_group_size) || min_group_size<1){
          cat("Skipping mouse", mouse, "(min_groupsize invalid)\n")
          next
        }
        
        for (g in unique_groups){
          rows_in_group = df_mouse[df_mouse[[paste0(nonrefLimb, '_categorical')]]==g,]
          sampled_rows = rows_in_group[sample(nrow(rows_in_group), min_group_size),]
          cat(mouse, g, nrow(sampled_rows), '\n')
          df_balanced = rbind(df_balanced, sampled_rows)
        }
      }
      df = df_balanced
    }
  }
  
  
  # deal with duty_ratio if it is in predictors
  if (("duty_ratio" %in% cont_predictors) & grepl("COMBINED", refLimb)){
    df$duty_ratio = df$duty_ref/df$duty_homolateral
    df = df[!is.infinite(df$duty_ratio),]
  }
  
  # convert phase to radians [0, 2pi]
  df$limb = (df[[limb]]*2*pi)
  limb_bool = df$limb<0
  limb_bool[which(is.na(limb_bool))] = FALSE
  df$limb[limb_bool] = df$limb[limb_bool]+(2*pi)
  
  
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
  
  if (length(cont_predictors) >1){
    pred2 = df[[paste(cont_predictors[2], '_centred', sep="")]]
    df_new <- data.frame(limb = limbx, mouse =mousex, pred1 = pred1, pred2=pred2)
  }
  if (length(cont_predictors) >2){
    pred3 = df[[paste(cont_predictors[3], '_centred', sep="")]]
    df_new <- data.frame(limb = limbx, mouse =mousex, pred1 = pred1, pred2=pred2, pred3=pred3)
  }
  if (length(cont_predictors) >3){
    pred4 = df[[paste(cont_predictors[4], '_centred', sep="")]]
    df_new <- data.frame(limb = limbx, mouse =mousex,pred1 = pred1,  pred2=pred2, pred3=pred3, pred4=pred4)
  }
  if (length(cont_predictors) >4){
    pred5 = df[[paste(cont_predictors[5], '_centred', sep="")]]
    df_new <- data.frame(limb = limbx, mouse =mousex,pred1 = pred1,  pred2=pred2, pred3=pred3, pred4=pred4, pred5=pred5)
  }
  
  predictors = cont_predictors
  
  # deal with categorical predictors
  if (length(cat_predictors) > 0){
    initial_predictor_count = length(predictors)
    
    if ('sex' %in% cat_predictors){
      df$sex = as.factor(df$sex) #some problems - only has one level!
      sexx = df$sex
      df_new[paste("pred", length(predictors)+1, sep="")] = sexx
      predictors = append(predictors, 'sex')
    }
    if (paste0(nonrefLimb, '_categorical') %in% cat_predictors){ 
      col = paste0(nonrefLimb, '_categorical')
      df[[col]] = as.factor(df[[col]]) #some problems - only has one level!
      reff = df[[col]]
      df_new[paste("pred", length(predictors)+1, sep="")] = reff
      predictors = append(predictors, col)
    }
    if ('refLimb' %in% cat_predictors){ 
      df$refLimb = as.factor(df$refLimb) #some problems - only has one level!
      reff = df$refLimb
      df_new[paste("pred", length(predictors)+1, sep="")] = reff
      predictors = append(predictors, 'refLimb')
    }
    if ('trialType' %in% cat_predictors){ 
      df$trialType = as.factor(df$trialType) #some problems - only has one level!
      reff = df$trialType
      df_new[paste("pred", length(predictors)+1, sep="")] = reff
      predictors = append(predictors, 'trialType')
    }
    if (length(predictors)==initial_predictor_count){
      stop("The supplied categorical predictor(s) have not been implemented")
    }
  } 
  
  sBA_split_str = sBA_split
  
  # subsample the dataframe
  if (!any(grepl('blncd', nonref_subset_interval_str))){
    subsample_size = as.integer(nrow(df_new)*sample_frac)
    df_new_sample = df_new[sample(nrow(df_new), subsample_size),]
    df_new_sample = na.omit(df_new_sample)
  }else{
    df_new_sample = df_new
    subsample_size = nrow(df_new_sample)
  }
  
  # deal with random slopes
  if (slope == TRUE){
    num_slopes = str_count(slope_type, 'pred')
  }
  
  # convert slope_type (pred2, pred2pred3, etc) into
  if (isFALSE(slope) || identical(slope_type, FALSE)) {
    slope_vec = character(0)
  } else{ 
    st = gsub("\\s+", "", slope_type)
    slope_vec = regmatches(st, gregexpr("pred[0-9]+", st))[[1]] # separates preds in slope_type
    
    if (length(slope_vec)==0){
      stop("slope_type must contain terms like 'pred2' or 'pred2pred3'")
    }
    
    # verify that all predN are present in df_new_sample
    missing = slope_vec[!slope_vec %in% names(df_new_sample)] # identify predN not in df_new_sample
    if (length(missing)>0){
      stop("slope_type requests slopes that are not in df_new_sample: ",
           paste(missing, collapse=", "))
    }
  }
  
  
  # build random term
  if (length(slope_vec)==0){
    random_term = "(1 | mouse)"
  } else{
    random_term = paste0("(", paste(slope_vec, collapse=" + "), " | mouse)")
  }
  
  nonref_subset_interval_str = paste(nonref_subset_interval_str, collapse='')
  
  # get all predX names that exist in df_new_sample
  pred_names = grep("^pred\\d+$", names(df_new_sample), value=TRUE)
  
  # generate fixed effect string
  if (interaction == FALSE){
    fixed_string = paste(pred_names, collapse=" + ")
  } else if(interaction == TRUE && interaction_type==""){
    # first two interact, others additive
    if (length(pred_names)==1){
      fixed_string = pred_names[1]
    } else{
      fixed_string = paste0(pred_names[1], " * ", pred_names[2])
      if (length(pred_names) > 2){
        fixed_string = paste(fixed_string, paste(pred_names[-c(1,2)], collapse=" + "), sep=" + ")
      }
    } 
  } else if(interaction == TRUE && interaction_type=="threeway"){
    fixed_string = paste0(paste(pred_names[1:3], collapse=" * "))
    if (length(pred_names) > 3){
      fixed_string = paste(fixed_string, paste(pred_names[-(1:3)], collapse=" + "), sep=" + ")
    }
  } else if(interaction == TRUE && interaction_type=="fourway"){
    fixed_string = paste0(paste(pred_names[1:4], collapse=" * "))
    if (length(pred_names) > 4){
      fixed_string = paste(fixed_string, paste(pred_names[-(1:4)], collapse=" + "), sep=" + ")
    }
  } else if(interaction == TRUE && interaction_type=="secondary"){
    fixed_string = paste0(pred_names[1], " + ", pred_names[2], " * ", pred_names[3])
    if (length(pred_names) > 3){
      fixed_string = paste(fixed_string, paste(pred_names[-(1:3)], collapse=" + "), sep=" + ")
    }
  } else if(interaction == TRUE && interaction_type=="1st2nd4th"){
    fixed_string = paste0(pred_names[1], " * ", pred_names[2], " * ", pred_names[4], " + ", pred_names[3])
    if (length(pred_names) > 4){
      fixed_string = paste(fixed_string, paste(pred_names[-(1:4)], collapse=" + "), sep=" + ")
    }
  } else{
    stop(paste("Unknown interaction_type:", interaction_type))
  }
  
  # final formula string
  formula_string = paste("limb ~", fixed_string, "+", random_term)
  model_formula = as.formula(formula_string)
  
  cat("USING MODEL FORMULA:\n", formula_string, "\n")
  
  # run the model
  model = bpnme(
    pred.I = model_formula,
    data = df_new_sample,
    its = iters,
    burn = 100,
    n.lag = 3,
    seed = 101
  )
  
  # generate file_ext string
  file_ext = paste0(limb, "_ref", refLimb, nonref_subset_interval_str, "_",
                    paste(pred_names, collapse="_"),
                    "_interaction", interaction,
                    if(interaction==TRUE && interaction_type!="") paste0("_", interaction_type) else "",
                    "_", param_type,
                    if(slope==TRUE) paste0("_randMouseSLOPE_", slope_type) else "_randMouse_",
                    "s", sBA_split_str, "_",
                    sample_frac, "data", subsample_size,
                    "s_", iters, "its_100burn_3lag.csv"
  )
  
  write.csv(model$beta1, paste(outputDir, yyyymmdd, "_beta1_", file_ext, sep=""))
  write.csv(model$beta2, paste(outputDir, yyyymmdd, "_beta2_", file_ext, sep=""))
  write.csv(fit(model), paste(outputDir, yyyymmdd, "_WAIC_", file_ext, sep=""))
  if (slope == TRUE){
    for (k in 1:(num_slopes+1)){
      write.csv(model$b1[,k,], paste(outputDir, yyyymmdd, "_b1_pred", slope_preds[k], "_", file_ext, sep=""))
      write.csv(model$b2[,k,], paste(outputDir, yyyymmdd, "_b2_pred", slope_preds[k], "_", file_ext, sep=""))
      write.csv(model$omega1[,k,], paste(outputDir, yyyymmdd, "_omega1_pred", slope_preds[k], "_", file_ext, sep=""))
      write.csv(model$omega2[,k,], paste(outputDir, yyyymmdd, "_omega2_pred", slope_preds[k], "_", file_ext, sep=""))
    }
  } else{
    write.csv(model$b1, paste(outputDir, yyyymmdd, "_b1_", file_ext, sep=""))
    write.csv(model$b2, paste(outputDir, yyyymmdd, "_b2_", file_ext, sep=""))
    write.csv(model$omega1, paste(outputDir, yyyymmdd, "_omega1_", file_ext, sep=""))
    write.csv(model$omega2, paste(outputDir, yyyymmdd, "_omega2_", file_ext, sep=""))
  }
  write.csv(coef_circ(model, type = 'continuous'), paste(outputDir, yyyymmdd, "_coefCircContinuous_", file_ext, sep=""))
  columns = c(c('cRI mean', 'cRI variance'), paste("mouse", unique(mousex)))
  values = c(mean(model$cRI), var(model$cRI), apply(model$circular.ri, 1, mean_circ))
  write.csv(data.frame(values, colnames = columns), paste(outputDir, yyyymmdd, "_randOutputs_", file_ext, sep=""))
  write.csv(model$circ.res.varrand, paste(outputDir, yyyymmdd, "_circResVarrand_", file_ext, sep="")) 
  
  if (length(cat_predictors)>0 & isFALSE(is.ordered(df_new_sample$pred2))){
    write.csv(coef_circ(model, type = 'categorical')$Differences, paste(outputDir, yyyymmdd, "_coefCircCategorical_", file_ext, sep=""))
    write.csv(coef_circ(model, type = 'categorical')$Means, paste(outputDir, yyyymmdd, "_coefCircCategorical_MEANS_", file_ext, sep=""))
  }
  plot(1:iters, model$beta1[,'pred1'])
  plot(1:iters, model$beta2[,'pred1'])
}

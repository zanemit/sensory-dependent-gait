library(BAMBI)
library(loo) # for WAIC
library(glue) # for f-strings
library(coda) # for statistical test of stationarity
set.seed(42)

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

run_bambi_batch <- function(folder, yyyymmdd, mouselist,
                            limb='rH0',
                            reflimb='lH1',
                            pred='snoutBodyAngle', 
                            lower_bound=168, 
                            upper_bound=178, 
                            max_components=3,
                            niters=5000,
                            nchains=3
                            ){
  filepath = paste(folder, '\\', yyyymmdd, "_strideParams_", reflimb, ".csv")
  df = read.csv(filepath)
  
  df$limb = df[[limb]]*2*pi
  limb_bool = df$limb<0
  limb_bool[which(is.na(limb_bool))] = FALSE
  df$limb[limb_bool] = df$limb[limb_bool]+(2*pi)
  
  if (pred=='incline'){
    df$incline = lapply(df$trialType, convert_char, elements=4, removeType='start')
    df$incline = as.numeric(df$incline)
    df$incline = lapply(df$incline, convert_deg)
    df$incline = as.numeric(df$incline)
  }
  
  df_sub = df[df[,pred] > lower_bound & df[,pred] < upper_bound, ]
  
  results <- data.frame(mouse=character(), stringsAsFactors=FALSE)
  
  for (m in mouselist){
    df_sub_mouse = subset(df_sub, mouseID=m)
    df_sub_mouse = na.omit(df_sub_mouse$limb)
    if (length(df_sub_mouse)>30){
      model_set = fit_incremental_angmix(
        model='vm',
        data=df_sub_mouse,
        crit='WAIC',
        start_ncomp=1,
        max_ncomp=max_ncomponents,
        n.chains=nchains,
        n.iter=niters
      )
      
      model_best = bestmodel(model_set)
      ptest = pointest(model_best, fn='MODE')
      
      row_data = list(mouse_m)
      
      n_components = ncol(ptest)
      for (i in 1:max_ncomponents){
        if (i <= n_components){
          row_data[[paste0("pmix", i)]] <- ptest["pmix", as.character(i)]
          row_data[[paste0("kappa", i)]] <- ptest["kappa", as.character(i)]
          row_data[[paste0("mu", i)]] <- ptest["mu", as.character(i)]
        }else{
          row_data[[paste0("pmix", i)]] <- NA
          row_data[[paste0("kappa", i)]] <- NA
          row_data[[paste0("mu", i)]] <- NA
        }
      }
      row_df <- as.data.frame(row_data, stringsAsFactors=FALSE)
      results <- rbind(results, row_df)
    }else{
      cat(m, " skipped\n")
    }
    
  }
  
  savepath =  file.path(
          folder,
          sprintf(
            "%s_BAMBI_%s_ref%s_%s_%s_%s_chains%s_iters%s.csv",
            yyyymmdd, limb, reflimb, pred, lower_bound, upper_bound, nchains, niters)
          )
  write.csv(results, path)
}

# Fig S4A
run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-08-18", 
                mouselist= c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 
                             'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
                             'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                limb='lF0',
                reflimb='lH1',
                pred='snoutBodyAngle', 
                lower_bound=141, 
                upper_bound=149
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-08-18", 
                mouselist= c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 
                             'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
                             'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                limb='lF0',
                reflimb='lH1',
                pred='snoutBodyAngle', 
                lower_bound=166, 
                upper_bound=174
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-02-26", 
                mouselist=c('BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
                            'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576'),
                limb='lF0',
                reflimb='lH1',
                pred='snoutBodyAngle', 
                lower_bound=141, 
                upper_bound=149
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-02-26", 
                mouselist= c('BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
                             'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576'),
                limb='lF0',
                reflimb='lH1',
                pred='snoutBodyAngle', 
                lower_bound=166, 
                upper_bound=174
)
                            
# Fig S4F
run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-08-18", 
                mouselist= c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 
                             'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
                             'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                limb='lF0',
                reflimb='lH1',
                pred='incline', 
                lower_bound=-40, 
                upper_bound=-20
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-08-18", 
                mouselist= c('FAA1034836', 'FAA1034839', 'FAA1034840', 'FAA1034842', 
                             'FAA1034867', 'FAA1034868', 'FAA1034869', 'FAA1034942', 
                             'FAA1034944', 'FAA1034945', 'FAA1034947', 'FAA1034949'),
                limb='lF0',
                reflimb='lH1',
                pred='incline', 
                lower_bound=20, 
                upper_bound=40
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-02-26", 
                mouselist= c('BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
                             'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576'),
                limb='lF0',
                reflimb='lH1',
                pred='incline', 
                lower_bound=-40, 
                upper_bound=-20
)

run_bambi_batch(folder="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", 
                yyyymmdd="2022-02-26", 
                mouselist= c('BAA1098955', 'FAA1034469', 'FAA1034471', 'FAA1034570',
                             'FAA1034572', 'FAA1034573', 'FAA1034575', 'FAA1034576'),
                limb='lF0',
                reflimb='lH1',
                pred='incline', 
                lower_bound=20, 
                upper_bound=40
)




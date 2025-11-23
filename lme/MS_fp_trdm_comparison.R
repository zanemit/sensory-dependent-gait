# ----------------  COMPARE SNOUT-BODY-ANGLE VS INCLINE FROM FORCEPLATE AND TREADMILL (preOpto: Figure S6C)
fp_treadmill_comparison <- function(yyyymmdd_trdm, 
                                    yyyymmdd_fp,
                                    outputDir_trdm="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\",
                                    outputDir_fp="C:\\Users\\MurrayLab\\Documents\\Forceplate\\"
                                    ){

    filepath_treadmill = paste(outputDir_trdm, yyyymmdd_trdm, "_locomParams_incline.csv", sep = '')
    filepath_forceplate = paste(outputDir_fp, yyyymmdd_fp, "_forceplateAngleParamsR_levels.csv", sep='')
    
    treadmill = read.csv(filepath_treadmill)
    forceplate = read.csv(filepath_forceplate)
    
    forceplate_id = rep(c('FP'),times=nrow(forceplate))
    treadmill_preOpto_id = rep(c('TRDMstat'),times=nrow(treadmill))
    
    treadmill$headLVL_int = lapply(treadmill$headLVL, convert_char, elements = 4, removeType = 'start')
    treadmill$headLVL_int = as.numeric(treadmill$headLVL_int)
    treadmill$headLVL_int = lapply(treadmill$headLVL_int, convert_deg)
    treadmill$headLVL_int = as.numeric(treadmill$headLVL_int) 
    
    forceplate_df = data.frame('mouseID' = forceplate$mouseID, 'snoutBodyAngle' = forceplate$snoutBodyAngle, 'incline' = forceplate$headLVL, 'setup'= forceplate_id)
    treadmill_preOpto_df = data.frame('mouseID' = treadmill$mouseID, 'snoutBodyAngle' = treadmill$snoutBodyAngle_preOpto, 'incline' = treadmill$headLVL_int, 'setup'= treadmill_preOpto_id)
    
    combo_df= rbind(forceplate_df, treadmill_preOpto_df)#rbind(forceplate_df, treadmill_df, treadmill_preOpto_df)
    write.csv(combo_df, paste(outputDir_trdm,yyyymmdd_trdm, "_locomParams_FPcomparison_incline.csv", sep=''))
    
    # CENTRE THE VARIABLES
    combo_df$snoutBodyAngle_centred = combo_df$snoutBodyAngle-mean(combo_df$snoutBodyAngle, na.rm = TRUE)
    combo_df$incline_centred = combo_df$incline-mean(combo_df$incline, na.rm = TRUE)
    combo_df$setup= as.factor(combo_df$setup) #, ordered = FALSE)
    modelLinear_1 = lmer(snoutBodyAngle_centred ~ incline_centred + setup + (1|mouseID), data = combo_df )
    modelLinear_2 = lmer(snoutBodyAngle_centred ~ incline_centred + setup + (incline_centred|mouseID), data = combo_df )
    modelLinear_3 = lmer(snoutBodyAngle_centred ~ incline_centred + setup + (setup|mouseID), data = combo_df )
    modelLinear_4 = lmer(snoutBodyAngle_centred ~ incline_centred + setup + (incline_centred + setup|mouseID), data = combo_df )
    
    model_converged <- function(model){
      is.null(model@optinfo$conv$lme4$messages)
    }
    
    safe_anova <- function(m1, m2, name1, name2, p_threshold = 0.05) {
      if (is.null(m1) || is.null(m2)) return(NULL)
      if (!model_converged(m1)) {
        message(name1, " did not converge. Skipping.")
        return(NULL)
      }
      if (!model_converged(m2)) {
        message(name2, " did not converge. Skipping.")
        return(NULL)
      }
      
      comp <- anova(m1, m2)
      pval <- comp$`Pr(>Chisq)`[2]
      print(comp)
      
      if (!is.na(pval) && pval < p_threshold) {
        message("Significant difference (p = ", round(pval, 4), ").")
        return(TRUE)
      } else {
        message("No significant difference (p = ", round(pval, 4), ").")
        return(FALSE)
      }
    }
    
    # List of models and names
    models <- list(modelLinear_1, modelLinear_2, modelLinear_3, modelLinear_4)
    names(models) <- c("m1", "m2", "m3", "m4")
    
    # ---- Automatic comparison logic ----
    compare_models <- function(models, p_threshold = 0.05) {
      n <- length(models)
      best <- 1
      
      for (i in seq_len(n - 1)) {
        m1 <- models[[best]]
        m2 <- models[[i + 1]]
        
        if (!model_converged(m2)) {
          message("Skipping ", names(models)[i + 1], " (no convergence).")
          next
        }
        
        result <- safe_anova(m1, m2, names(models)[best], names(models)[i + 1], p_threshold)
        
        if (isTRUE(result)) {
          best <- i + 1
        }
      }
      
      message("Best (most complex justified) model: ", names(models)[best])
      return(list(model=models[[best]], name=names(models)[best]))
    }
    
    # CHECK WHICH MODEL IS BETTER
    result <- compare_models(models)
    best_model = result$model
    best_model_name = result$name
    
    file_ext = paste0("snoutBodyAngle_vs_incline_FP_TRDMstat_COMPARISON_", best_model_name, ".csv")
    write.csv(summary(best_model)$coefficients, paste(outputDir, yyyymmdd_trdm, "_mixedEffectsModel_linear_", file_ext, sep=""))
    
    #print(paste(outputDir, yyyymmdd_trdm, "_mixedEffectsModel_linear_", file_ext, sep=""))
    AICRsq_df = data.frame('Model' = c(rep('Linear',3)), 'Metric' =  c('R2marginal', 'R2total','AIC'), 'Value' = c(r.squaredGLMM(best_model)[1],r.squaredGLMM(best_model)[2],AIC(best_model)))
    write.csv(AICRsq_df, paste(outputDir, yyyymmdd_trdm, "_mixedEffectsModel_AICRsq_", file_ext, sep=""))
}

# Fig S6 C
fp_treadmill_comparison(yyyymmdd_trdm='2022-08-18', 
                        yyyymmdd_fp='2022-04-04',
                        outputDir_trdm="C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\",
                        outputDir_fp="C:\\Users\\MurrayLab\\Documents\\Forceplate\\"
                    )

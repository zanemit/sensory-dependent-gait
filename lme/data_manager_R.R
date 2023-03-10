combine_two_DFs <- function(
                            yyyymmdd1, 
                            yyyymmdd2,
                            type1, 
                            type2, 
                            newcolname,
                            newyyyymmdd,
                            appdx1 = "",
                            appdx2 = "",
                            outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", 
                            filename = "_forceplateAngleParamsR_levels"
                            ){
  # combines the two datasets into one dataframe so that a single model with 
  # {newcolname} as an additional predictor can be fit

  filePath1 = paste(outputDir, yyyymmdd1, filename, appdx1, ".csv", sep="")
  df1 <- read.csv(filePath1)
  df1[newcolname] = type1
  
  filePath2 = paste(outputDir, yyyymmdd2, filename, appdx2, ".csv", sep="")
  df2 <- read.csv(filePath2)
  df2[newcolname] = type2
  
  in1not2 = setdiff(colnames(df1), colnames(df2))
  n2not1 = setdiff(colnames(df2), colnames(df1))
  df1 = df1[,!(names(df1) %in% in1not2)]
  df2 = df2[,!(names(df2) %in% in2not1)]
  
  df_combined = rbind(df1, df2)
  write.csv(df_combined, paste(outputDir, newyyyymmdd, filename, "_COMBINED.csv", sep = ""))
}

## EXAMPLES ##
## FP filename = "_forceplateAngleParamsR_levels"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04', type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x', filename = "_forceplateAngleParamsR_levels")

## FP filename = "_meanParamDF_levels"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x', filename = "_meanParamDF_levels")

## FP filename = "_meanParamDF_snoutBodyAngle"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x',  filename = "_meanParamDF_snoutBodyAngle")

## FP filename = "_meanParamDF_pc"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x',  filename = "_meanParamDF_pc")

## FP filename = "_meanParamDF_posXrH1"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x',  filename = "_meanParamDF_posXrH1")

## FP filename = "_meanParamDF_posXrF1"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x',  filename = "_meanParamDF_posXrF1")

## FP filename = "_limbPositionRegressionArray"
# combine_two_DFs(yyyymmdd1 = '2022-04-02', yyyymmdd2 = '2022-04-04',  type1 = 12, type2 = 5, outputDir = "C:\\Users\\MurrayLab\\Documents\\Forceplate\\", newcolname = 'headHeight', newyyyymmdd = '2022-04-0x',  filename = "_limbPositionRegressionArray")

## TRDM filename = "_locomParamsAcrossMice"
# combine_two_DFs(yyyymmdd1 = '2022-08-18', yyyymmdd2 = '2022-08-18',  type1 = "headHeight", type2 = "incline", appdx1 = "", appdx2 = "_incline", outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", newcolname = 'trialType', newyyyymmdd = '2022-08-18',  filename = "_locomParamsAcrossMice")

## TRDM filename = "_strideParams"
# combine_two_DFs(yyyymmdd1 = '2022-08-18', yyyymmdd2 = '2022-08-18',  type1 = "headHeight", type2 = "incline", appdx1 = '_lH1', appdx2 = '_incline_lH1', outputDir = "C:\\Users\\MurrayLab\\Documents\\PassiveOptoTreadmill\\", newcolname = 'trialType', newyyyymmdd = '2022-08-18',  filename = "_strideParams")

  

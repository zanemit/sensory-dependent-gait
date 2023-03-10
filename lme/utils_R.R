library(lme4)
library(MuMIn)
library(lmerTest)
library(multcomp)
library(pracma)

convert_char <- function(char, elements, removeType){
  # removes characters from the "start" or "end" of a string
  if (removeType == "start"){
    return(substr(char,elements,nchar(char)))
  } else if (removeType == "end"){
    return(substr(char,1,nchar(char)-elements))
  }
}

convert_rl <- function(x){
  # extracts the head height ("rl") as an integer
  # converts the value so that the intermediate height is zero
  # x = number
  
  print(x)
  return(x *-1 + 6)
}

convert_deg <- function(x){
  # extracts the incline ("deg") as an integer
  # converts the value so that upward slopes are positive
  # x = number
  
  return(x *-1)
}

get_appdx <- function(x){
  # handles the file appendix
  # x = string
  
  if (x == ""){
    appdx = ".csv"
  } else if (x == "incline"){
    appdx = "_incline.csv"
  } else if (appdx == 'COMBINED'){
    appdx = "_COMBINED.csv"
  }else {
    stop("The supplied appdx is not allowed!")
  }
}

remove_outliers <- function(vec){
  iqr = IQR(vec, na.rm = TRUE)
  q = quantile(vec, probs = c(0.25, 0.75), na.rm = TRUE)
  upper = q[2]+1.5*iqr
  lower = q[1]-1.5*iqr
  vec_no_outliers = vec[vec>lower & vec<upper]
  return(vec_no_outliers)
}

table_glht <- function(x) {
  # Table to export results of multiple comparison (post hoc Tukey)
  # Source: Modified from https://gist.github.com/cheuerde/3acc1879dc397a1adfb0 
  # x is a ghlt object
  pq <- summary(x)$test
  mtests <- cbind(pq$coefficients, pq$sigma, pq$tstat, pq$pvalues)
  error <- attr(pq$pvalues, "error")
  pname <- switch(x$alternativ, less = paste("Pr(<", ifelse(x$df ==0, "z", "t"), ")", sep = ""), 
                  greater = paste("Pr(>", ifelse(x$df == 0, "z", "t"), ")", sep = ""), two.sided = paste("Pr(>|",ifelse(x$df == 0, "z", "t"), "|)", sep = ""))
  colnames(mtests) <- c("Estimate", "Std. Error", ifelse(x$df ==0, "z value", "t value"), pname)
  return(mtests)
  
}


#' identify number of NA
#' 
#' @description This function returns the number of NA for each column in x
#' @param x This is the dataframe.

identify_na <- function(x) {
  colSums(apply(x, 2, is.na))
}

#' find the mode of a column
#' 
#' @description This function returns the mode of a column
#' @param x This is the column.
mode_stat <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' determine if a track deviates in any audio feature. 
#' 
#' @description This function returns a logical vector of size `spotify`. It belongs to a map function. Selects the second column (ie. 'high deviation') and creates a new vector which is true if ANY audio feature deviates >2 sigma and false otherwise. 
#' @param df This is the dataframe which is .x in map. 

determine.deviation <- function (df) {
  if (sum(df[,2]) > 0) {
    test_v <- TRUE
  } else {
    test_v <- FALSE
  }
  return(test_v)
}
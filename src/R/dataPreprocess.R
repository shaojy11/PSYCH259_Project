setwd("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/test")
ad <- autodetec(threshold = 5, env = "hil", ssmooth = 300, power=1,
                bp=c(2,9), xl = 2, picsize = 2, res = 200, flim= c(1,11), osci = TRUE,
                wl = 300, ls = FALSE, sxrow = 2, rows = 4, mindur = 0.1, maxdur = 1, set = TRUE, redo = FALSE)
featureMatrix <- specan(ad, bp = c(0,22))
#view(featureMatrix)


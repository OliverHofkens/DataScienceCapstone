source('clean_data.R')
library(quanteda)

corp <-  corpus(corpus)
rm(corpus)

summary(corp)

features <- dfm(corp)

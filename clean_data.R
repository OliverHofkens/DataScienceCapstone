library(tm)

source('get_data.R')

getCorpus <- function(language){
    language <- 'en_US'
    dir <- getRawData(language)
    VCorpus(DirSource(dir, encoding = "UTF-8"), readerControl = list(language = language))
}

library(tm)
library(parallel)

source('get_data.R')

cores = parallel::detectCores()
tm_parLapply_engine(parallel::mclapply)

getRawCorpus <- function(language){
    dir <- getRawData(language)
    VCorpus(DirSource(dir, encoding = "UTF-8"), readerControl = list(language = language))
}

getProfaneWords <- function(language){
    if(nchar(language) > 2){
        language <- substr(language, 0, 2)
    }
    
    directory <- 'data/profanity/'
    destination <- paste(directory, language, '.txt', sep="")
    dir.create(directory, showWarnings = FALSE)
    
    if(!file.exists(destination)){
        source <- paste('https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/', language, sep = "")
        download.file(source, destination)
    }
    
    readLines(destination)
}

cleanCorpus <- function(corpus, language){
    transforms <- list(stripWhitespace,
                removePunctuation,
                removeNumbers,
                content_transformer(tolower))
    
    corpus <- tm_map(corpus, FUN = tm_reduce, tmFuns = transforms, mc.cores=cores)
    
    # Remove profanity
    profane <- getProfaneWords(language)
    corpus <- tm_map(corpus, removeWords, words=profane, mc.cores=cores)
    
    corpus
}

cleanDataFile <- 'data/clean/en_US.RData'

if(!file.exists(cleanDataFile)){
    rawCorpus <- getRawCorpus('en_US')
    corpus <- cleanCorpus(rawCorpus, 'en_US')
    
    rm(rawCorpus)
    
    dir.create('data/clean', showWarnings = FALSE, recursive = TRUE)
    
    saveRDS(corpus, cleanDataFile)
} else {
    corpus <- readRDS(cleanDataFile)
}



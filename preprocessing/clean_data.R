library(tm)
library(parallel)

cores = parallel::detectCores() - 1
tm_parLapply_engine(parallel::mclapply)

# Load in the files as a TM corpus:
getRawCorpus <- function(){
    dir <- 'data/split'
    VCorpus(DirSource(dir, encoding = "UTF-8"), readerControl = list())
}

# Download and import a list of profane words:
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

# Clean the given corpus:
cleanCorpus <- function(corpus, language){
    # Strip excessive whitespace
    # Remove numbers and punctuation
    # Lowercase all
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

getCleanData <- function(language){
    cleanDataDir <- 'data/clean'
    dir.create('data/clean', showWarnings = FALSE)
    
    rawCorpus <- getRawCorpus()
    corpus <- cleanCorpus(rawCorpus, language)
    
    writeCorpus(corpus, path = cleanDataDir)
    cleanDataDir
}
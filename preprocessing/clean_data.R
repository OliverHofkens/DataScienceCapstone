library(tm)
library(parallel)

cores = parallel::detectCores() - 1
tm_parLapply_engine(parallel::mclapply)

# Load in the files as a TM corpus:
getRawCorpus <- function(language){
    dir <- 'data/split'
    corp <- VCorpus(DirSource(dir, encoding="UTF-8-MAC"), readerControl = list(reader = readPlain, language = language))
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
    # Removes punctuation that usually doesn't appear inside a word (preserves dashes and single quotes for e.g. isn't)
    removePunctuationSafe <- function(x){
        gsub("[][!\"#$%&()*+,./:;<=>?@\\^_`{|}~]", "", x)
    }
    
    # Strip excessive whitespace
    # Remove numbers and punctuation
    # Lowercase all
    transforms <- list(stripWhitespace,
                removeNumbers,
                content_transformer(removePunctuationSafe),
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
    
    rawCorpus <- getRawCorpus(language)
    corpus <- cleanCorpus(rawCorpus, language)
    
    writeCorpus(corpus, path = cleanDataDir, filenames = c('test.txt','training.txt', 'validation.txt'))
    cleanDataDir
}
library(tm)
library(parallel)

cores = parallel::detectCores() - 1
tm_parLapply_engine(parallel::mclapply)

# Load in the files as a TM corpus:
getRawCorpus <- function(language){
    dir <- 'data/split'
    corp <- VCorpus(DirSource(dir, encoding="UTF-8"), readerControl = list(reader = readPlain, language = language))
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
        gsub("[][\"#$%&()*+,/:;<=>@\\^_`{|}~\u201C\u201D\u00AB\u00BB\u00BD\u00B7\u2026\u0093\u0094\u0092\u0096\u0097\u2032\u2033\u00B4\u00A3\u2013]", " ", x)
    }
    
    # Removes dashes and quotes that can actually appear inside a word (but don't)
    removeOtherPunctuation <- function(x){
        x <- gsub("\\W['-]", " ", x)
        gsub("['-]\\W", " ", x)
    }
    
    fillSentenceEnding <- function(x){
        gsub("[.!?]", " <eos> ", x)
    }
    
    corpus <- tm_map(corpus, FUN = tolower)
    corpus <- tm_map(corpus, FUN = removeNumbers)
    corpus <- tm_map(corpus, FUN = removePunctuationSafe)
    corpus <- tm_map(corpus, FUN = removeOtherPunctuation)
    corpus <- tm_map(corpus, FUN = fillSentenceEnding)
    
    # Remove profanity
    profane <- getProfaneWords(language)
    corpus <- tm_map(corpus, removeWords, words=profane, mc.cores=cores)
    
    # Remove excessive whitespace
    corpus <- tm_map(corpus, FUN = stripWhitespace)
    corpus <- tm_map(corpus, FUN = trimws)
    
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

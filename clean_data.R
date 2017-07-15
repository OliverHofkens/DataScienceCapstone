library(tm)

source('get_data.R')

getCorpus <- function(language){
    language <- 'en_US'
    dir <- getRawData(language)
    VCorpus(DirSource(dir, encoding = "UTF-8"), readerControl = list(language = language))
}

getProfaneWords <- function(language){
    if(nchar(language) > 2){
        language <- substr(language, 0, 2)
    }
    
    directory <- 'data/profanity'
    destination <- paste(directory, language, '.txt', sep="")
    dir.create(directory, showWarnings = FALSE)
    
    if(!file.exists(destination)){
        source <- paste('https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/', language, sep = "")
        download.file(source, destination)
    }
    
    readLines(destination)
}

cleanCorpus <- function(corpus, language){
    # Strip excessive whitespace
    corpus <- tm_map(corpus, stripWhitespace)
    
    # Remove numbers
    corpus <- tm_map(corpus, removeNumbers)
    
    # Remove punctuation
    corpus <- tm_map(corpus, removePunctuation)
    
    # Make everything lowercase
    corpus <- tm_map(corpus, content_transformer(tolower))
    
    # Remove profanity
    profane <- getProfaneWords(language)
    corpus <- tm_map(corpus, removeWords, profane)
    
    corpus
}

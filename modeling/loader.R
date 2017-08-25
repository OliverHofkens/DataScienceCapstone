library(quanteda)
library(data.table)
library(parallel)

Sys.setlocale("LC_ALL", "nl_BE.UTF-8")

# Load a corpus from the given filename
getCorpus <- function(filename){
    directory <- 'data/model_input/'
    file <- paste(directory, filename, sep="")
    content <- paste(readLines(file), collapse = " <eos> ")
    corpus <- corpus(content)
}

# Build the document-feature matrix of the contents of given filename
buildDfm <- function(filename){
    corp <- getCorpus(filename)
    result <- dfm(corp, what="fastestword")
    dfm_sort(result)
}

# Build a vocabulary dictionary consisting of word -> ID
buildVocab <- function(filename){
    freq <- buildDfm(filename)
    table <- as.data.table(freq)
    words <- as.data.table(colnames(table))
    words <- cbind(words, as.integer(rownames(words)))
    colnames(words) <- c('word', 'id')
    setkey(words, word, id)
    words
}

# Translate a word to an ID (based on dictionary vocab)
wordToId <- function(word, vocab){
    vocab[.(word)]$id
}

# Tokenize a file
getTokens <- function(filename){
    corp <- getCorpus(filename)
    words <- tokens(corp, what="fastestword")
    words <- words$text1
}

# Translate a file into IDs
getWordIds <- function(filename, vocab){
    tokens <- getTokens(filename)
    ids <- vocab[.(tokens)]$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

loadVocabulary <- function(){
    buildVocab('training.txt')
}

loadModelInputs <- function(){
    vocab <- loadVocabulary()
    train <- getWordIds('training.txt', vocab)
    validation <- getWordIds('validation.txt', vocab)
    test <- getWordIds('test.txt', vocab)
    list(train = train, validation = validation, test = test, vocabulary = vocab)
}
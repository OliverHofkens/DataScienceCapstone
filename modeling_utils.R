library(tensorflow)
library(quanteda)
library(data.table)
library(parallel)

directory <- 'data/modelprep/'

getCorpus <- function(filename){
    file <- paste(directory, filename, sep="")
    content <- paste(readLines(file), collapse = " <eos> ")
    corpus(content)
}

buildDfm <- function(filename){
    corp <- getCorpus(filename)
    result <- dfm(corp, what = "fasterword")
    dfm_sort(result)
}

buildVocab <- function(filename){
    freq <- buildDfm(filename)
    table <- as.data.table(freq)
    words <- colnames(table)
    words <- as.data.table(words)
    words <- cbind(words, as.integer(rownames(words)))
    colnames(words) <- c('word', 'id')
    setkey(words, word, id)
    words
}

wordToId <- function(word, vocab){
    vocab[.(word)]$id
}

fileToWordIds <- function(filename, vocab){
    corp <- getCorpus(filename)
    words <- tokens(corp, what = "fasterword")
    words <- words$text1
    rm(corp)
    gc()
    mclapply(words, wordToId, vocab = vocab, mc.cores = 4)
}

trainVocab <- buildVocab('train.txt')
gc()
ids <- fileToWordIds('train.txt', trainVocab)


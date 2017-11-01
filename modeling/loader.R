library(quanteda)
library(data.table)
library(parallel)

Sys.setlocale("LC_ALL", "nl_BE.UTF-8")

# Load a corpus from the given filename
getCorpus <- function(filename){
    directory <- 'data/model_input/'
    file <- paste(directory, filename, sep="")
    content <- paste(readLines(file), collapse = " <eof> ")
    corpus <- corpus(content)
}

# Build the document-feature matrix of the contents of given filename
buildDfm <- function(filename){
    corp <- getCorpus(filename)
    tokens <- tokens(corp, what="fastestword")
    result <- dfm(tokens)
    dfm_sort(result)
}

# Build a vocabulary dictionary consisting of word -> ID
buildVocab <- function(filename){
    freq <- buildDfm(filename)
    table <- as.data.table(freq)
    words <- as.data.table(colnames(table))
    words <- cbind(words, (as.integer(rownames(words)))) # Subtract 1 for zero-based indexing
    colnames(words) <- c('word', 'id')
    setkey(words, word, id)
    words
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
    ids <- vocab[tokens, on='word']$id
    
    # replace words that weren't found with <unk>
    unk <- vocab[vocab$word == "<unk>"]$id
    ids[is.na(ids)] <- unk
    
    # Split on <eof> by first pasting all together:
    ids <- paste(ids, collapse = " ")
    eofId <- vocab[vocab$word == "<eof>"]$id
    # Surround the ID with spaces, we want to split on _id_ only
    eofId <- paste(" ", eofId, " ", sep = "")
    splits <- unlist(strsplit(ids, eofId, fixed = TRUE))
    # Finally, split each sentence on spaces and convert to int:
    sapply(splits, function(sentence){
            as.integer(
                unlist(
                    strsplit(sentence, " ", fixed = TRUE)
                )
            )
        }, USE.NAMES = FALSE)
}

loadVocabulary <- function(){
    vocabFile <- 'data/vocab.rds'
    if(file.exists(vocabFile)){
        vocab <- readRDS(vocabFile)
        return(vocab)
    }
    
    vocab <- buildVocab('training.txt')
    
    vocab <- vocab[order(id)]
    
    saveRDS(vocab, vocabFile)
    
    vocab
}

loadModelInputs <- function(){
    inputsFile <- 'data/inputs.rds'
    if(file.exists(inputsFile)){
        inputs <- readRDS(inputsFile)
        return(inputs)
    }
    
    vocab <- loadVocabulary()
    train <- getWordIds('training.txt', vocab)
    validation <- getWordIds('validation.txt', vocab)
    test <- getWordIds('test.txt', vocab)
    
    inputs <- list(train = train, validation = validation, test = test, vocabulary = vocab)
    saveRDS(inputs, inputsFile)
    inputs
}

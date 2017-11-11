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
    # Calculate frequencey weights to use when training:
    wt <- dfm_weight(freq, type = "relmaxfreq")
    weights <- as.data.table(t(wt))
    # Invert the weights, so the most frequent word has the lowest weight
    # Then take the log to compress the space
    weights$text1 <- log(1 / weights$text1 + 0.1)
    
    words <- as.data.table(colnames(wt))
    words <- cbind(words, as.integer(rownames(words)), weights)
    colnames(words) <- c('word', 'id', 'weight')
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
    res <- sapply(splits, function(sentence){
            as.integer(
                unlist(
                    strsplit(sentence, " ", fixed = TRUE)
                )
            )
        }, USE.NAMES = FALSE)
    
    # Remove any sentences that are only 1 word long
    cond <- lapply(res, function(x) length(x) > 1)
    res[unlist(cond)]
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

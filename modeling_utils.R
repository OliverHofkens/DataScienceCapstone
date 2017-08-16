library(tensorflow)
library(quanteda)
library(data.table)
library(parallel)

directory <- 'data/model_input/'

# Load a corpus from the given filename
getCorpus <- function(filename){
    file <- paste(directory, filename, sep="")
    content <- paste(readLines(file), collapse = " <eos> ")
    corpus <- corpus(content)
}

# Build the document-feature matrix of the contents of given filename
buildDfm <- function(filename){
    corp <- getCorpus(filename)
    result <- dfm(corp, what="fastestword", remove_url=TRUE)
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
    words <- tokens(corp, what="fastestword", remove_url=TRUE)
    words <- words$text1
}

# Translate a file into IDs
getWordIds <- function(filename, vocab){
    tokens <- getTokens(filename)
    ids <- vocab[.(tokens)]$id
    
    # remove words that weren't found:
    ids[!is.na(ids)]
}

loadVocabulary <- function(){
    buildVocab('train.txt')
}

loadModelInputs <- function(){
    vocab <- loadVocabulary()
    train <- getWordIds('train.txt', vocab)
    validation <- getWordIds('validation.txt', vocab)
    test <- getWordIds('test.txt', vocab)
    list(train = train, validation = validation, test = test, vocabulary <- vocab)
}

batchData <- function(data, batchSize, steps, isTrain = FALSE){
    batches <- floor(length(data) / batchSize)
    lastBatchedElement <- (batches * batchSize) - 1
    
    with(tf$name_scope('batcher'), {
        # Load the data
        data = tf$convert_to_tensor(data, name="raw_data", dtype=tf$int32)
        
        # Calculate how many batches there are, and reshape the vector to a table of batches.
        dataLen = tf$size(data)
        batchLen = dataLen %/% batchSize
        data = tf$reshape(data[0L : lastBatchedElement], list(batchSize, batchLen))
        
        # The last batch is incomplete because data is not exactly a multiple of batchSize
        epochSize = (batchLen - 1L)
        assertion = tf$assert_positive(epochSize, message="epochSize == 0, decrease batchSize or numSteps")
        with(tf$control_dependencies(list(assertion)), {
            epochSize = tf$identity(epochSize, name="epoch_size")
        })
        
        i = tf$train$range_input_producer(epochSize, shuffle=FALSE)$dequeue()
        
        x = tf$strided_slice(data, list(0L, i * steps), list(batchSize, (i + 1L) * steps))
        x$set_shape(shape(batchSize, steps))
        
        y = tf$strided_slice(data, list(0L, i * steps + 1L), list(batchSize, (i + 1L) * steps + 1L))
        y$set_shape(shape(batchSize, steps))
        
        return(c(x,y))
    })
}

#detach('package:quanteda')

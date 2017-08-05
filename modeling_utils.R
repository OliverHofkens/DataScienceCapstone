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

getTokens <- function(filename){
    corp <- getCorpus(filename)
    words <- tokens(corp, what = "fasterword")
    words <- words$text1
}

getWordIds <- function(filename){
    vocab <- buildVocab(filename)
    tokens <- getTokens(filename)
    vocab[.(tokens)]$id
}

#train <- getWordIds('train.txt')
#validation <- getWordIds('validation.txt')
#test <- getWordsIds('test.txt')
#gc()

batchData <- function(data, batchSize, steps){
    #detach('package:quanteda')
    
    with(tf$name_scope('batcher'), {
        # Load the data
        data = tf$convert_to_tensor(data, name="raw_data", dtype=tf$int32)
        
        # Calculate how many batches there are, and reshape the vector to a table of batches.
        dataLen = tf$size(data)
        batchLen = dataLen %/% batchSize
        data = tf$reshape(data[1L : 20832900L], list(batchSize, batchLen))
        
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

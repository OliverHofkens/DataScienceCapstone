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

train <- getWordIds('train.txt')
validation <- getWordIds('validation.txt')
test <- getWordsIds('test.txt')
gc()

batchData <- function(data, batchSize, steps){
    with(tf$name_scope('batcher'), {
        data = tf$convert_to_tensor(rdata, name="raw_data", dtype=tf$int32)
        
        dataLen = tf$size(data)
        batchLen = dataLen / batchSize
        data = tf$reshape(data[0 : batchSize * batchLen],c(batch_size, batch_len))
        
        epochSize = (batchLen - 1)
        assertion = tf$assert_positive(epochSize, message="epochSize == 0, decrease batchSize or numSteps")
        with(tf$control_dependencies(c(assertion)), {
            epochSize = tf$identity(epochSize, name="epoch_size")
        })
        
        i = tf$train$range_input_producer(epochSize, shuffle=FALSE)$dequeue()
        
        x = tf$strided_slice(data, c(0, i * steps), c(batchsize, (i + 1) * steps))
        x$set_shape(c(batchSize, numSteps))
        
        y = tf$strided_slice(data, c(0, i * steps + 1), c(batchSize, (i + 1) * steps + 1))
        y$set_shape(c(batch_size, num_steps))
        
        c(x,y)
    })
}

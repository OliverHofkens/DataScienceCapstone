source("modeling/loader.R")
library(data.table)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

config <- list(
    sequenceLengthWords = 5L,
    strideStep = 1L,
    nHiddenLayers = 256,
    learningRate = 0.01,
    batchSize = 100,
    nEpochs = 100,
    trainMaxQueueSize = 20
    )

# Data Prep
inputs <- loadModelInputs()
train <- inputs$train[1:1000000]
#validation <- inputs$validation
#test <- inputs$test
vocab <- inputs$vocabulary
rm(inputs)

#write.table(vocab, file='vocab.tsv', quote=FALSE, sep='\t', row.names = FALSE)

inputGenerator <- function(dataset, vocabulary, config, startAt=1) {
    # Add 1 as 0 is reserved for masking unused inputs
    vocabSize <- length(vocabulary$word) + 1
    stride <- config$strideStep
    seqLength <- config$sequenceLengthWords
    batchSize <- config$batchSize
    
    index = startAt
    
    # Input = (batchSize x Sequence Length)
    # Will be converted to Embedding by model.
    X <- array(0, dim = c(batchSize, seqLength))
    
    # 2-D output (batchSize x one-hot vocab)
    Y <- array(0, dim = c(batchSize, seqLength))
    
    function() {
        nextIndex <-  index + (batchSize * stride) - 1
        
        # If we reached the end (+ prediction), start over:
        if(nextIndex + seqLength + 1 > length(dataset)){
            index <<- 1
            nextIndex <- index + (batchSize * stride) - 1
        }
        
        X <<- sapply(seq(index, nextIndex, by = stride), function(i) {dataset[i:(i + seqLength - 1L)]})
        X <<- t(X)
        
        Y <<- sapply(seq(index, nextIndex, by = stride), function(i) {dataset[i + seqLength]})
        Y <<- sapply(Y, function(yi){
            # Add a 0 in front, to be used when masking unused inputs
            c(0, as.integer(vocab$id == yi))
        })
        Y <<- t(Y)
        
        index <<- nextIndex + 1
    
        return(list(X,Y))
    }
}

# Correct because of 1-based indexing:
lastCompleteBatch = length(train) - config$sequenceLengthWords
batchesPerEpoch <- floor(lastCompleteBatch / (config$batchSize * config$strideStep))

modelPattern <- "model.(\\d+)-\\d+.\\d+.hdf5"
checkpointFiles <- list.files(pattern=glob2rx("model.*.hdf5"))

if(length(checkpointFiles) > 0){
    checkpointFiles <- sort(checkpointFiles, decreasing = TRUE)
    checkpoint <- checkpointFiles[1]
    
    model <- load_model_hdf5(checkpoint)
    
    matches <- regmatches(checkpoint, regexec(modelPattern, checkpoint))
    startEpoch <- as.integer(matches[[1]][[2]]) 
} else {
    startEpoch <- 0L
}

# Model Definition
model <- keras_model_sequential()

model %>%
    layer_embedding(length(vocab$id) + 1, config$nHiddenLayers, input_length = config$sequenceLengthWords, mask_zero = TRUE) %>%
    layer_lstm(config$nHiddenLayers, return_sequences = TRUE) %>%
    layer_dropout(0.1) %>%
    layer_lstm(config$nHiddenLayers) %>%
    layer_dense(length(vocab$id) + 1) %>%
    layer_activation("softmax")

model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer_rmsprop(lr = config$learningRate),
    metrics = c('accuracy')
)

# Training and prediction
history <- model %>%
    fit_generator(
        generator = inputGenerator(train, vocab, config),
        steps_per_epoch = batchesPerEpoch,
        max_queue_size = config$trainMaxQueueSize,
        epochs=config$nEpochs, 
        initial_epoch = startEpoch,
        callbacks = list(
            callback_model_checkpoint("model.{epoch:02d}-{loss:.2f}.hdf5"),
            callback_reduce_lr_on_plateau(monitor = "loss",factor = 0.5, patience = 3, verbose = 1, min_lr = 0.0005),
            callback_tensorboard(log_dir = "log", embeddings_freq = 5, embeddings_metadata = 'vocab.tsv')
        ))

save_model_hdf5(model, 'keras_model.h5', include_optimizer = TRUE)
rModel <- serialize_model(model, include_optimizer = TRUE)
saveRDS(rModel, 'keras_model_r.rds')
saveRDS(history, 'history.rds')

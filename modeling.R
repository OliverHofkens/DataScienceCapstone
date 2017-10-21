source("modeling/loader.R")
library(data.table)
library(purrr)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

config <- list(
    sequenceLengthWords = 5L,
    strideStep = 2L,
    nHiddenLayers = 256,
    learningRate = 0.001,
    batchSize = 100,
    nEpochs = 20,
    trainMaxQueueSize = 20
    )

# Data Prep
inputs <- loadModelInputs()
input <- inputs$train[1:100000]
#validation <- c(inputs$test)
vocab <- inputs$vocabulary
rm(inputs)

buildDataset <- function(inputVector, config){
    # Build a strided slice of lengths sequenceLengthWords, with striding step strideStep
    dataset <- map(
        seq(1, length(inputVector) - config$sequenceLengthWords - 1L, by = config$strideStep), 
        ~list(sentence = inputVector[.x:(.x + config$sequenceLengthWords - 1)], next_word = inputVector[.x + config$sequenceLengthWords])
    )
    
    dataset <- transpose(dataset)
    
    dataset
}

inputGenerator <- function(dataset, vocabulary, config, startAt=1) {
    # Add 1 as 0 is reserved for masking unused inputs
    vocabSize <- length(vocabulary$word) + 1
    
    index = startAt
    
    # Transform our dataset into the one-hot matrices for the model:
    # 3-D input (words, sequences, one-hot vocab)
   #X <- array(0, dim = c(config$batchSize, config$sequenceLengthWords, vocabSize))
    X <- array(0, dim = c(config$batchSize, config$sequenceLengthWords))
    
    # 2-D output (sequences, one-hot vocab)
    y <- array(0, dim = c(config$batchSize, vocabSize))
    
    function() {
        # Global dataset indexes:
        next_index = index + config$batchSize - 1
        
        # If we reached the end, start over at a random spot between 1 and config$strideStep
        if(next_index > length(dataset$sentence)){
            index <<- sample(1:config$strideStep, 1)
            next_index <- index + config$batchSize - 1
        }
        
        # local dataset index:
        current_i = 1
        for(i in index:next_index){
            #X[current_i,,] <- sapply(vocabulary$id, function(x){
            #    as.integer(x == dataset$sentence[[i]])
            #})
            X[current_i,] <- dataset$sentence[[i]]
                    
            # Add a 0 in front, to be used when masking unused inputs
            y[current_i,] <- c(0, as.integer(vocabulary$id == dataset$next_word[[i]]))
            current_i <- current_i + 1
        }
        index <<- next_index
        
        return(list(X,y))
    }
}

inputDataset <- buildDataset(input, config)
rm(input)
#validationDataset <- buildDataset(validation, config)
#rm(validation)

batchesPerEpoch <- 1000
batchesPerEpoch <- floor(length(inputDataset$sentence) / config$batchSize)
#validationBatchesPerEpoch <- floor(length(validationDataset$sentence) / config$batchSize)

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
startAtSample = (batchesPerEpoch * startEpoch * config$batchSize) + 1

# Model Definition
model <- keras_model_sequential()

model %>%
    layer_embedding(length(vocab$id) + 1, 128, input_length = config$sequenceLengthWords, mask_zero = TRUE) %>%
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
        generator = inputGenerator(inputDataset, vocab, config, startAt=startAtSample),
        steps_per_epoch = batchesPerEpoch,
        max_queue_size = config$trainMaxQueueSize,
        epochs=config$nEpochs, 
        initial_epoch = startEpoch,
        callbacks = list(
            callback_model_checkpoint("model.{epoch:02d}-{loss:.2f}.hdf5")
        )
        )

save_model_hdf5(model, 'keras_model.h5', include_optimizer = TRUE)
rModel <- serialize_model(model, include_optimizer = TRUE)
saveRDS(rModel, 'keras_model_r.rds')
saveRDS(history, 'history.rds')
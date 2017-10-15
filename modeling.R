source("modeling/loader.R")
library(purrr)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

config <- list(
    sequenceLengthWords = 5L,
    strideStep = 3L,
    nHiddenLayers = 128,
    learningRate = 0.05,
    batchSize = 500,
    nEpochs = 3,
    trainMaxQueueSize = 10
    )

# Data Prep
inputs <- loadModelInputs()
input <- c(inputs$train, inputs$validation)
validation <- c(inputs$test)
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

input_generator <- function(dataset, vocabulary, config) {
    vocabSize <- length(vocabulary$word)
    
    index = 1
    
    # Transform our dataset into the one-hot matrices for the model:
    # 3-D input (words, sequences, one-hot vocab)
    X <- array(0, dim = c(config$batchSize, config$sequenceLengthWords, vocabSize))
    
    # 2-D output (sequences, one-hot vocab)
    y <- array(0, dim = c(config$batchSize, vocabSize))
    
    function() {
        # Global dataset indexes:
        next_index = index+config$batchSize
        
        # If we reached the end, start over at a random spot between 1 and config$strideStep
        if(next_index > length(dataset)){
            index <<- sample(1:config$strideStep, 1)
            next_index <- index+config$batchSize
        }
        
        for(i in index:next_index){
            # local dataset index:
            current_i = 1
            X[current_i,,] <- sapply(vocabulary$id, function(x){
                as.integer(x == dataset$sentence[[i]])
            })
            
            y[current_i,] <- as.integer(vocabulary$id == dataset$next_word[[i]])
            current_i = current_i + 1
        }
        index <<- next_index
        
        return(list(X,y))
    }
}

inputDataset <- buildDataset(input, config)
rm(input)
#validationDataset <- buildDataset(validation, config)
rm(validation)

batchesPerEpoch <- floor(length(inputDataset$sentence) / config$batchSize)
#validationBatchesPerEpoch <- floor(length(validationDataset$sentence) / config$batchSize)

# Model Definition

model <- keras_model_sequential()

model %>%
    layer_lstm(config$nHiddenLayers, input_shape = c(config$sequenceLengthWords, length(vocab$id))) %>%
    layer_dense(length(vocab$id)) %>%
    layer_activation("softmax")

optimizer <- optimizer_rmsprop(lr = config$learningRate)

model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer,
    metrics = c('accuracy')
)

# Training and prediction
history <- model %>%
    fit_generator(
        generator = input_generator(inputDataset, vocab, config),
        steps_per_epoch = 1000, # batchesPerEpoch, # Replaced to speed up initial model dev
        max_queue_size = config$trainMaxQueueSize,
        epochs=config$nEpochs)

save_model_hdf5(model, 'keras_model.h5', include_optimizer = TRUE)
rModel <- serialize_model(model, include_optimizer = TRUE)
saveRDS(rModel, 'keras_model_r.rds')
saveRDS(history, 'history.rds')
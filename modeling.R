source("modeling/loader.R")
library(data.table)
library(purrr)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

config <- list(
    sequenceLengthWords = 4L,
    strideStep = 2L,
    nHiddenLayers = 512,
    learningRate = 0.01,
    batchSize = 100,
    nEpochs = 100,
    trainMaxQueueSize = 10
    )

# Data Prep
inputs <- loadModelInputs()
input <- inputs$train[1:1000]
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
    vocabSize <- length(vocabulary$word)
    
    index = startAt
    
    # Transform our dataset into the one-hot matrices for the model:
    # 3-D input (words, sequences, one-hot vocab)
    X <- array(0, dim = c(config$batchSize, config$sequenceLengthWords, vocabSize))
    
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
            X[current_i,,] <- sapply(vocabulary$id, function(x){
                as.integer(x == dataset$sentence[[i]])
            })
            
            y[current_i,] <- as.integer(vocabulary$id == dataset$next_word[[i]])
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
#batchesPerEpoch <- floor(length(inputDataset$sentence) / config$batchSize)
#validationBatchesPerEpoch <- floor(length(validationDataset$sentence) / config$batchSize)

modelPattern <- "model.(\\d+)-\\d+.\\d+.hdf5"
checkpointFiles <- list.files(pattern=glob2rx("model.*.hdf5"))

if(length(checkpointFiles) > 0){
    checkpointFiles <- sort(checkpointFiles, decreasing = TRUE)
    checkpoint <- checkpointFiles[1]
    
    model <- load_model_hdf5(checkpoint)
    
    matches <- regmatches(checkpoint, regexec(modelPattern, checkpoint))
    startEpoch <- as.integer(matches[[1]][[2]]) + 1
} else {
    startEpoch <- 1
}
startAtSample = batchesPerEpoch * startEpoch * config$batchSize

# Model Definition
model <- keras_model_sequential()

model %>%
    layer_masking(mask_value = 0, input_shape = list(NULL, length(vocab$id))) %>%
    layer_lstm(config$nHiddenLayers, input_shape = c(config$sequenceLengthWords, length(vocab$id))) %>%
    layer_dense(length(vocab$id)) %>%
    layer_activation("softmax")

model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer_rmsprop(lr = config$learningRate),
    metrics = c('accuracy')
)

# Training and prediction
history <- model %>%
    fit_generator(
        generator = input_generator(inputDataset, vocab, config, startAt=startAtSample),
        steps_per_epoch = batchesPerEpoch,
        max_queue_size = config$trainMaxQueueSize,
        epochs=config$nEpochs, 
        initial_epoch = startEpoch#,
        #callbacks = list(
        #    callback_model_checkpoint("model.{epoch:02d}-{loss:.2f}.hdf5")
        #)
        )

save_model_hdf5(model, 'keras_model.h5', include_optimizer = TRUE)
rModel <- serialize_model(model, include_optimizer = TRUE)
saveRDS(rModel, 'keras_model_r.rds')
saveRDS(history, 'history.rds')
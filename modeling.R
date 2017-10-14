source("modeling/loader.R")
library(purrr)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

inputs <- loadModelInputs()

config <- list(
    sequenceLengthWords = 5L,
    strideStep = 3L,
    learningRate = 0.01,
    batchSize = 500
    )
vocabSize <- length(inputs$vocabulary$word)


# Data Prep

input <- inputs$validation

# Build a strided slice of lengths sequenceLengthWords, with striding step strideStep
dataset <- map(
        seq(1, length(input) - config$sequenceLengthWords - 1L, by = config$strideStep), 
        ~list(sentence = input[.x:(.x + config$sequenceLengthWords - 1)], next_word = input[.x + config$sequenceLengthWords])
    )

dataset <- transpose(dataset)

input_generator <- function(batch_size) {
    index = 1
    
    # Transform our dataset into the one-hot matrices for the model:
    # 3-D input (words, sequences, one-hot vocab)
    X <- array(0, dim = c(batch_size, config$sequenceLengthWords, vocabSize))
    
    # 2-D output (sequences, one-hot vocab)
    y <- array(0, dim = c(batch_size, vocabSize))
    
    function() {
        # Global dataset indexes:
        next_index = index+batch_size
        for(i in index:next_index){
            # local dataset index:
            current_i = 1
            X[current_i,,] <- sapply(inputs$vocabulary$id, function(x){
                as.integer(x == dataset$sentence[[i]])
            })
            
            y[current_i,] <- as.integer(inputs$vocabulary$id == dataset$next_word[[i]])
            current_i = current_i + 1
        }
        index <<- next_index
        
        return(list(X,y))
    }
}

batchesPerEpoch <- floor(length(dataset$sentence) / config$batchSize)

# Model Definition

model <- keras_model_sequential()

model %>%
    layer_lstm(128, input_shape = c(config$sequenceLengthWords, vocabSize)) %>%
    layer_dense(vocabSize) %>%
    layer_activation("softmax")

optimizer <- optimizer_rmsprop(lr = config$learningRate)

model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer,
    metrics = c('accuracy')
)

# Training and prediction
history <- model %>%
    fit_generator(input_generator(batch_size = config$batchSize),
        steps_per_epoch = batchesPerEpoch,
        epochs=1L)

save_model_hdf5(model, 'keras_model.h5')

sampleFunction <- function(preds, temperature = 1){
    preds <- log(preds)/temperature
    exp_preds <- exp(preds)
    preds <- exp_preds/sum(exp(preds))
    
    rmultinom(1, 1, preds) %>% 
        as.integer() %>%
        which.max()
}
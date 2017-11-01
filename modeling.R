source("modeling/loader.R")
library(data.table)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

config <- list(
    sequenceLengthWords = 10L,
    strideStep = 1L,
    embeddingSize = 200,
    nHiddenLayers = 200,
    learningRate = 0.001,
    batchSize = 100,
    nEpochs = 100,
    trainMaxQueueSize = 20,
    lrDecay = 0.8,
    lrMin = 0.0001,
    decreaseLrPatience = 5,
    dropout = 0.2,
    validationSplit = 0.1
    )

# Data Prep
inputs <- loadModelInputs()
train <- inputs$train[1:1000]
#validation <- inputs$validation
vocab <- inputs$vocabulary
rm(inputs)

#write.table(vocab, file='vocab.tsv', quote=FALSE, sep='\t', row.names = FALSE)

inputGenerator <- function(dataset, vocabulary, config) {
    # Add 1 as 0 is reserved for masking unused inputs
    vocabSize <- length(vocabulary$word) + 1
    stride <- config$strideStep
    seqLength <- config$sequenceLengthWords
    batchSize <- config$batchSize
    
    function() {
        resultsPerSentence <- sapply(dataset, function(sentence){
            endIndexOfLastBatch <- length(sentence) - 1L
            sapply(seq.int(1, endIndexOfLastBatch, by=stride), function(i){
                # Start at current index minus seqLength, or 1 if not enough words.
                from <- max(i - seqLength + 1, 1)
                batch <- sentence[from:i]
                # Pad batch with 0 if not long enough:
                padSize <- seqLength - length(batch)
                if(padSize > 0){
                    batch <- c(rep.int(0, padSize), batch)
                }
                pred <- sentence[i + 1]
                list(batch, pred)
            })
        })
        
        resultsPerSentence <- unlist(resultsPerSentence, recursive = FALSE)
        # All odd results are inputs, even results are outputs:
        X <- resultsPerSentence[seq.int(1, length(resultsPerSentence), by = 2)]
        X <- matrix(unlist(X), ncol = seqLength, byrow = TRUE)
        
        Y <- resultsPerSentence[seq.int(2, length(resultsPerSentence), by = 2)]
        Y <- sapply(Y, function(yi){
            # Add a 0 in front, to be used when masking unused inputs
            c(0, as.integer(vocab$id == yi))
        })
        Y <- t(Y)
    
        return(list(X,Y))
    }
}

embeddingMatrix <- readRDS('matrix.RDS')

inputGen = inputGenerator(train, vocab, config)
inputDataset = inputGen()

#modelPattern <- "model.(\\d+)-\\d+.\\d+.hdf5"
#checkpointFiles <- list.files(pattern=glob2rx("model.*.hdf5"))

#if(length(checkpointFiles) > 0){
#    checkpointFiles <- sort(checkpointFiles, decreasing = TRUE)
#    checkpoint <- checkpointFiles[1]
    
    #model <- load_model_hdf5(checkpoint)
    
#    matches <- regmatches(checkpoint, regexec(modelPattern, checkpoint))
#    startEpoch <- as.integer(matches[[1]][[2]]) 
#} else {
    startEpoch <- 0L
#}

# Model Definition
model <- keras_model_sequential()

model %>%
    layer_embedding(length(vocab$id) + 1, config$embeddingSize, 
                    input_length = config$sequenceLengthWords, 
                    mask_zero = TRUE, weights = list(embeddingMatrix)) %>%
    layer_lstm(config$nHiddenLayers, return_sequences = TRUE, 
               dropout = config$dropout) %>%
    layer_lstm(config$nHiddenLayers, dropout = config$dropout) %>%
    layer_dense(length(vocab$id) + 1) %>%
    layer_activation("softmax")

model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer_rmsprop(lr = config$learningRate),
    metrics = c('accuracy')
)

history <- model %>%
    fit(
        x=inputDataset[[1]],
        y=inputDataset[[2]],
        epochs=config$nEpochs, 
        initial_epoch = startEpoch,
        validation_split = config$validationSplit,
        callbacks = list(
            callback_model_checkpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only = TRUE),
            callback_reduce_lr_on_plateau(monitor = "val_loss",factor = config$lrDecay, patience = config$decreaseLrPatience, min_lr = config$lrMin),
            callback_tensorboard(log_dir = "log", embeddings_freq = 10, embeddings_metadata = 'vocab.tsv')
            #callback_early_stopping(monitor = "val_loss", patience = 10)
        ))


# history <- model %>%
#     fit_generator(
#         generator = inputGenerator(train, vocab, config),
#         steps_per_epoch = batchesPerEpoch,
#         max_queue_size = config$trainMaxQueueSize,
#         epochs=config$nEpochs, 
#         initial_epoch = startEpoch,
#         validation_data = validationDataset,
#         callbacks = list(
#             callback_model_checkpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only = TRUE),
#             callback_reduce_lr_on_plateau(monitor = "val_loss",factor = 0.8, patience = 3, min_lr = 0.00001),
#             callback_tensorboard(log_dir = "log", embeddings_freq = 5, embeddings_metadata = 'vocab.tsv'),
#             callback_early_stopping(monitor = "val_loss", patience = 10)
#         ))

save_model_hdf5(model, 'keras_model.h5', include_optimizer = TRUE)
rModel <- serialize_model(model, include_optimizer = TRUE)
saveRDS(rModel, 'keras_model_r.rds')
saveRDS(history, 'history.rds')

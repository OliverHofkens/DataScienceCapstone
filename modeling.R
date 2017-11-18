source("modeling/loader.R")
library(data.table)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

FLAGS <- flags(
    flag_numeric("sequenceLengthWords", 8L),
    flag_numeric("strideStep", 1L),
    flag_numeric("embeddingSize", 200L),
    flag_numeric("nHiddenLayers", 200L),
    flag_numeric("learningRate", 0.002),
    flag_numeric("nEpochs", 10),
    flag_numeric("lrDecay", 0.5),
    flag_numeric("lrMin", 0.0001),
    flag_numeric("decreaseLrPatience", 3),
    flag_numeric("dropout", 0.25),
    flag_numeric("sentencesPerBatch", 30000L),
    flag_numeric("inputSentences", 30000L),
    flag_numeric("validationSentences", 3000L),
    flag_numeric("batchSize", 256),
    flag_string("continueFrom", FALSE),
    flag_boolean("trainEmbedding", TRUE),
    flag_string("weights", FALSE),
    flag_numeric("startSuperEpoch", 1),
    flag_numeric("topKMetric", 3)
)

# Data Prep
inputs <- loadModelInputs()
train <- inputs$train[1:FLAGS$inputSentences]
validation <- inputs$validation[1:FLAGS$validationSentences]
vocab <- inputs$vocabulary
rm(inputs)

classWeights <- as.list(c(0, vocab$weight))
names(classWeights) <- seq.int(0, 10002)

#write.table(vocab, file='vocab.tsv', quote=FALSE, sep='\t', row.names = FALSE)

inputGenerator <- function(dataset, vocabulary, config) {
    # Add 1 as 0 is reserved for masking unused inputs
    vocabSize <- length(vocabulary$word) + 1
    stride <- config$strideStep
    seqLength <- config$sequenceLengthWords
    
    function() {
        resultsPerSentence <- sapply(dataset, function(sentence){
            endIndexOfLastBatch <- length(sentence) - 1L
            
            perSentence <- sapply(seq.int(1, endIndexOfLastBatch, by=stride), function(i){
                # Start at current index minus seqLength, or 1 if not enough words.
                pred <- sentence[i + 1]
                
                from <- max(i - seqLength + 1, 1)
                batch <- sentence[from:i]
                # Pad batch with 0 if not long enough:
                padSize <- seqLength - length(batch)
                if(padSize > 0){
                    batch <- c(rep.int(0, padSize), batch)
                }
                
                list(batch, pred)
            })
        })
        
        resultsPerSentence <- unlist(resultsPerSentence, recursive = FALSE)
        # All odd results are inputs, even results are outputs:
        X <- resultsPerSentence[seq.int(1, length(resultsPerSentence), by = 2)]
        X <- matrix(unlist(X), ncol = seqLength, byrow = TRUE)
        
        Y <- as.integer(resultsPerSentence[seq.int(2, length(resultsPerSentence), by = 2)])
        
        # Downsample based on frequency:
        #dropChances <- vocabulary[Y, 'drop_chance']
        #drops <- as.logical(rbinom(length(dropChances$drop_chance), 1, dropChances$drop_chance))
        #X <- X[!drops,]
        #Y <- Y[!drops]
        
        return(list(X,Y))
    }
}

embeddingMatrix <- readRDS('matrix.RDS')

sparse_top_3_acc <- function(y_pred, y_true){
    metric_sparse_top_k_categorical_accuracy(y_pred, y_true, k = FLAGS$topKMetric)
}
# Workaround to fix re-loading the model:
attr(sparse_top_3_acc, "py_function_name") <- "top_3_acc"

# Model Definition
if(FLAGS$continueFrom == "FALSE") {
    model <- keras_model_sequential()

    model %>%
        layer_embedding(length(vocab$id) + 1, FLAGS$embeddingSize, 
                        input_length = FLAGS$sequenceLengthWords, 
                        mask_zero = TRUE, weights = list(embeddingMatrix),
                        trainable = FLAGS$trainEmbedding, 
                        name = 'embedding') %>%
        layer_lstm(FLAGS$nHiddenLayers, return_sequences = TRUE, 
                   name = 'lstm-transfer-1') %>%
        layer_lstm(FLAGS$nHiddenLayers, 
                   dropout = FLAGS$dropout, recurrent_dropout = FLAGS$dropout,
                   name = 'lstm-last') %>%
#        layer_batch_normalization() %>%
        layer_dense(length(vocab$id) + 1,
                    name = 'dense') %>%
        layer_activation("softmax",
                         name = 'activation')
    
    model %>% compile(
        loss = "sparse_categorical_crossentropy", 
        optimizer = optimizer_nadam(lr = FLAGS$learningRate),
        metrics = c(top_3_acc = sparse_top_3_acc)
    )
} else {
    model <- load_model_hdf5(FLAGS$continueFrom, custom_objects=c(top_3_acc=sparse_top_3_acc))
}

if(FLAGS$weights != "FALSE"){
    cat("Loading model weights: ")
    cat(FLAGS$weights)
    load_model_weights_hdf5(model, FLAGS$weights, by_name = TRUE)
}

rm(embeddingMatrix)

validationGen <- inputGenerator(validation, vocab, FLAGS)
validationDataset <- validationGen()

superBatches <- floor(length(train) / FLAGS$sentencesPerBatch) 
for(i in FLAGS$startSuperEpoch:superBatches){
    cat(sprintf("\n\nSuperEpoch %d of %d\n\n",i,superBatches))
    startIndex <- ((i - 1) * FLAGS$sentencesPerBatch) + 1
    nextIndex <- startIndex + FLAGS$sentencesPerBatch - 1
    sentences <- train[startIndex:nextIndex]
    inputGen = inputGenerator(sentences, vocab, FLAGS)
    inputDataset = inputGen()
    
    history <- model %>%
        fit(
            x=inputDataset[[1]],
            y=inputDataset[[2]],
            batch_size=FLAGS$batchSize,
            epochs=FLAGS$nEpochs, 
            validation_data = validationDataset,
            class_weight = classWeights,
            callbacks = list(
                callback_model_checkpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5"),
                callback_reduce_lr_on_plateau(monitor = "loss",factor = FLAGS$lrDecay, patience = FLAGS$decreaseLrPatience, min_lr = FLAGS$lrMin, verbose=TRUE),
                callback_tensorboard(log_dir = "log", embeddings_freq = 10, embeddings_metadata = 'vocab.tsv'),
                callback_early_stopping(monitor = "loss", patience = 8)
            ))
    
    save_model_weights_hdf5(model, paste('keras_weights', i, '.h5', sep = ""))
    save_model_hdf5(model, paste('keras_model', i, '.h5', sep = ""), include_optimizer = TRUE)
    saveRDS(history, paste('history', i, '.rds', sep = ""))
}

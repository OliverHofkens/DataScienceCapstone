source("modeling/loader.R")
library(data.table)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

FLAGS <- flags(
    flag_numeric("sequenceLengthWords", 8L),
    flag_numeric("strideStep", 1L),
    flag_numeric("embeddingSize", 200L),
    flag_numeric("nHiddenLayers", 200L),
    flag_numeric("learningRate", 0.01),
    flag_numeric("nEpochs", 20),
    flag_numeric("lrDecay", 0.5),
    flag_numeric("lrMin", 0.001),
    flag_numeric("decreaseLrPatience", 3),
    flag_numeric("dropout1", 0.1),
    flag_numeric("dropout2", 0.1),
    flag_numeric("sentencesPerBatch", 100000L),
    flag_numeric("inputSentences", 1000000L),
    flag_numeric("validationSentences", 1000L),
    flag_numeric("batchSize", 32),
    flag_string("continueFrom", FALSE)
)

# Data Prep
inputs <- loadModelInputs()
train <- inputs$train[1:FLAGS$inputSentences]
validation <- inputs$validation[1:FLAGS$validationSentences]
vocab <- inputs$vocabulary
rm(inputs)

#write.table(vocab, file='vocab.tsv', quote=FALSE, sep='\t', row.names = FALSE)

inputGenerator <- function(dataset, vocabulary, config) {
    # Add 1 as 0 is reserved for masking unused inputs
    vocabSize <- length(vocabulary$word) + 1
    stride <- config$strideStep
    seqLength <- config$sequenceLengthWords
    
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
        
        Y <- as.integer(resultsPerSentence[seq.int(2, length(resultsPerSentence), by = 2)])

        
        return(list(X,Y))
    }
}

embeddingMatrix <- readRDS('matrix.RDS')

# Model Definition
if(FLAGS$continueFrom == "FALSE") {
    model <- keras_model_sequential()

    model %>%
        layer_embedding(length(vocab$id) + 1, FLAGS$embeddingSize, 
                        input_length = FLAGS$sequenceLengthWords, 
                        mask_zero = TRUE, weights = list(embeddingMatrix)) %>%
        layer_lstm(FLAGS$nHiddenLayers, return_sequences = TRUE, 
                   dropout = FLAGS$dropout1, recurrent_dropout = FLAGS$dropout1,
                   activation = NULL) %>%
        layer_batch_normalization() %>%
        layer_activation("tanh") %>%
        layer_lstm(FLAGS$nHiddenLayers, 
                   dropout = FLAGS$dropout2, recurrent_dropout = FLAGS$dropout2,
                   activation = NULL) %>%
        layer_batch_normalization() %>%
        layer_activation("tanh") %>%
        layer_dense(length(vocab$id) + 1) %>%
        layer_batch_normalization() %>%
        layer_activation("softmax")
    
    model %>% compile(
        loss = "sparse_categorical_crossentropy", 
        optimizer = optimizer_nadam(lr = FLAGS$learningRate),
        metrics = c('accuracy')
    )
} else {
    model <- load_model_hdf5(FLAGS$continueFrom)
}

rm(embeddingMatrix)

validationGen <- inputGenerator(validation, vocab, FLAGS)
validationDataset <- validationGen()

superBatches <- floor(length(train) / FLAGS$sentencesPerBatch) 
for(i in 1:superBatches){
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
            callbacks = list(
                callback_model_checkpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only = TRUE),
                callback_reduce_lr_on_plateau(monitor = "val_loss",factor = FLAGS$lrDecay, patience = FLAGS$decreaseLrPatience, min_lr = FLAGS$lrMin),
                callback_tensorboard(log_dir = "log", embeddings_freq = 5, embeddings_metadata = 'vocab.tsv')
                #callback_early_stopping(monitor = "val_loss", patience = 10)
            ))
    
    save_model_hdf5(model, paste('keras_model', i, '.h5', sep = ""), include_optimizer = TRUE)
    saveRDS(history, paste('history', i, '.rds', sep = ""))
}

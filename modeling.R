source("modeling/loader.R")
library(data.table)
library(keras)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

FLAGS <- flags(
    flag_numeric("sequenceLengthWords", 7L),
    flag_numeric("strideStep", 1L),
    flag_numeric("embeddingSize", 200L),
    flag_numeric("nHiddenLayers", 200L),
    flag_numeric("learningRate", 0.002),
    flag_numeric("nEpochs", 5),
    flag_numeric("lrDecay", 0.9),
    flag_numeric("lrMin", 0.0001),
    flag_numeric("decreaseLrPatience", 10),
    #flag_numeric("dropout1", 0),
    flag_numeric("dropout2", 0),
    flag_numeric("sentencesPerBatch", 250000L),
    flag_numeric("inputSentences", 2500000L),
    flag_numeric("validationSentences", 2500L)
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
model <- keras_model_sequential()

model %>%
    layer_embedding(length(vocab$id) + 1, FLAGS$embeddingSize, 
                    input_length = FLAGS$sequenceLengthWords, 
                    mask_zero = TRUE, weights = list(embeddingMatrix)) %>%
    #layer_lstm(FLAGS$nHiddenLayers, return_sequences = TRUE, 
    #           dropout = FLAGS$dropout1) %>%
    layer_lstm(FLAGS$nHiddenLayers, 
               dropout = FLAGS$dropout2, recurrent_dropout = FLAGS$dropout2) %>%
    layer_dense(length(vocab$id) + 1) %>%
    layer_activation("softmax")

model %>% compile(
    loss = "sparse_categorical_crossentropy", 
    optimizer = optimizer_nadam(lr = FLAGS$learningRate),
    metrics = c('accuracy')
)

rm(embeddingMatrix)

validationGen <- inputGenerator(validation, vocab, FLAGS)
validationDataset <- validationGen()

superBatches <- floor(length(train) / FLAGS$sentencesPerBatch) - 1
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
            epochs=FLAGS$nEpochs, 
            validation_data = validationDataset,
            callbacks = list(
                callback_model_checkpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only = TRUE),
                #callback_reduce_lr_on_plateau(monitor = "val_loss",factor = FLAGS$lrDecay, patience = FLAGS$decreaseLrPatience, min_lr = FLAGS$lrMin),
                callback_tensorboard(log_dir = "log", embeddings_freq = 5, embeddings_metadata = 'vocab.tsv')
                #callback_early_stopping(monitor = "val_loss", patience = 10)
            ))
    
    save_model_hdf5(model, paste('keras_model', i, '.h5', sep = ""), include_optimizer = TRUE)
    saveRDS(history, paste('history', i, '.rds', sep = ""))
}

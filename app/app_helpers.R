library(data.table)

predictOnText <- function(model, vocab, text, samplingTemp = 0){
    text <- tolower(text)
    # replace punctuation and numbers:
    text <- gsub("[][\"#$%&()*+/:;<=>@\\^_`{|}~\u201C\u201D\u00AB\u00BB\u00BD\u00B7\u2026\u0093\u0094\u0092\u0096\u0097\u2032\u2033\u00B4\u00A3\u2013]", " ", text)
    text <- gsub("\\W['-]", " ", text)
    text <- gsub("['-]\\W", " ", text)
    text <- gsub("[+-]?[0-9]+[.,]?[0-9]*", " <num> ", text)
    text <- gsub(",", " ", text)
    text <- gsub("[.!?]+", " <eos> ", text)
    
    words <- unlist(strsplit(text, " ", fixed = TRUE))
    words <- trimws(tail(words, n=8))
    
    ids <- getWordIds(vocab, words)
    
    # If we don't have 5 words yet, pad with 0 for embedding layer mask in model:
    padLength = 8 - length(ids)
    if(padLength > 0){
        ids <- c(rep.int(0, padLength), ids)
    }
    
    predictions <- predictNextIds(model, vocab, ids, samplingTemp)
    predictions['word'] <- getWordForId(vocab, predictions[,'id'])
    
    predictions[predictions['word'] == '<eos>', 'word'] = "."
    predictions[predictions['word'] == '<unk>', 'word'] = "Out of vocabulary"
    predictions[predictions['word'] == '<num>', 'word'] = "A number"
    
    predictions
}

# Transforms a vector of words to a vector of the ids that represent those words.
getWordIds <- function(vocab, words){
    ids <- vocab[words, on='word']$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

predictNextIds <- function(model, vocab, ids, samplingTemp = 0){
    ids <- matrix(ids, nrow = 1)
    
    preds <- predict(model, ids)
    
    if (samplingTemp >= 0.1){
        # Add some minor randomization for funzies
        n = 5
        samples <- t(sampleMod(preds, temperature = 1 - samplingTemp, n))
        
        sorted <- as.data.frame(sort(samples, decreasing = TRUE, index.return = TRUE))
        tops <- sorted[sorted$x > 0,]
        probs <- tops['x'] / n
    } else {
        sorted <- as.data.frame(sort(preds, decreasing = TRUE, index.return = TRUE))
        tops <- head(sorted, n = 5)
        probs <- tops['x']
    }
  
    # Subtract 1 to offset masking
    indexes <- tops['ix'] - 1

    res <- data.frame(indexes, probs)
    colnames(res) <- c('id', 'prob')
    
    return(res)
}

getWordForId <- function(vocab, ids){
    sapply(ids, function(x){
        subset(vocab, vocab$id == x)$word
    })
}


sampleMod <- function(preds, temperature = 1, n = 5){
    preds <- log(preds)/temperature
    exp_preds <- exp(preds)
    preds <- exp_preds/sum(exp(preds))
    
    rmultinom(1, n, preds)
}

sparse_top_k_cat_acc <- function(y_pred, y_true){
    metric_sparse_top_k_categorical_accuracy(y_pred, y_true, k = 3)
}
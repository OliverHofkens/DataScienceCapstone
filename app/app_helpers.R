library(data.table)

predictOnText <- function(model, vocab, text){
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
    
    predictions <- predictNextIds(model, vocab, ids)
    predictions['word'] <- getWordForId(vocab, predictions[,'id'])
    
    predictions[predictions['word'] == '<eos>', 'word'] = "./!/?"
    predictions[predictions['word'] == '<unk>', 'word'] = "Out of Vocabulary"
    
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

predictNextIds <- function(model, vocab, ids){
    ids <- matrix(ids, nrow = 1)
    
    preds <- predict(model, ids)
    
    sorted <- as.data.frame(sort(preds, decreasing = TRUE, index.return = TRUE))
    
    # Subtract 1 to offset the masking 0 value
    tops <- head(sorted, n = 5)
    indexes <- tops['ix'] - 1
    probs <- tops['x']
    res <- data.frame(indexes, probs)
    colnames(res) <- c('id', 'prob')
    
    return(res)
}

getWordForId <- function(vocab, ids){
    sapply(ids, function(x){
        subset(vocab, vocab$id == x)$word
    })
}
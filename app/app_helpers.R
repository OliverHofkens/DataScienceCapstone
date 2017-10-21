library(data.table)

predictOnText <- function(model, vocab, word_vector){
    ids <- getWordIds(vocab, word_vector)
    
    # If we don't have 5 words yet, pad with 0 for embedding layer mask in model:
    padLength = 5 - length(ids)
    if(padLength > 0){
        ids <- c(ids, rep.int(0, padLength))
    }
    
    predictionIds <- predictNextIds(model, vocab, ids)
    getWordForId(vocab, predictionIds)
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
    #input <- sapply(vocab$id, function(x){
    #    as.integer(x == ids)
    #})
    ids <- matrix(ids, nrow = 1)
    
    preds <- predict(model, ids)
    
    #preds <- log(preds)
    #exp_preds <- exp(preds)
    #preds <- exp_preds / sum(exp_preds)
    
    sorted <- sort(preds, decreasing = TRUE, index.return = TRUE)
    
    # Subtract 1 to offset the masking 0 value
    indexes <- sorted[['ix']] - 1
    
    return(head(indexes, n=5))
}

getWordForId <- function(vocab, ids){
    sapply(ids, function(x){
        subset(vocab, vocab$id == x)$word
    })
}
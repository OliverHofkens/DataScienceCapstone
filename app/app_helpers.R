library(data.table)

predictOnText <- function(model, vocab, word_vector){
    ids <- getWordIds(vocab, word_vector)
    prediction_id <- predictNextId(model, vocab, ids)
    getWordForId(vocab, prediction_id)
}

# Transforms a vector of words to a vector of the ids that represent those words.
getWordIds <- function(vocab, words){
    ids <- vocab[.(words)]$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

predictNextId <- function(model, vocab, ids){
    #input <- sapply(vocab$id, function(x){
    #    as.integer(x == ids)
    #})
    ids <- matrix(ids, nrow = 1)
    
    preds <- predict(model, ids)
    
    preds <- log(preds)
    exp_preds <- exp(preds)
    preds <- exp_preds / sum(exp_preds)
    
    # Returns the INDEX of the prediction, not the ID!
    # So we subtract 1 (offset of 0-masking)
    rmultinom(1, 1, preds) %>% 
        as.integer() %>%
        which.max() - 1
}

getWordForId <- function(vocab, ids){
    sapply(ids, function(x){
        subset(vocab, vocab$id == x)$word
    })
}
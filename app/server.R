library(shiny)
library(tensorflow)
library(keras)
library(data.table)

# File names
model <- load_model_hdf5('model/keras_model.h5')

#Load Vocab
vocab <- readRDS(paste('model','vocab.rds', sep = "/"))

# Transforms a vector of words to a vector of the ids that represent those words.
getWordIds <- function(words){
    ids <- vocab[.(words)]$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

getWordForId <- function(id){
    select <- vocab$id == id
    vocab[select]$word
}

predict_next_ids <- function(ids){
    input <- sapply(vocab$id, function(x){
        as.integer(x == ids)
    })
    dim(input) <- c(1, dim(input))
    
    preds <- predict(model, input)
    
    preds <- log(preds)
    exp_preds <- exp(preds)
    preds = exp_preds / sum(exp_preds)
    
    rmultinom(1, 1, preds) %>% 
        as.integer() %>%
        which.max()
}

shinyServer(function(input, output) {
    output$predictions <- renderPrint({
        text <- tolower(input$text_input)
        words <- unlist(strsplit(text, " ", fixed = TRUE))
        words <- tail(words, n=5)
        ids <- getWordIds(words)
        
        prediction <- predict_next_ids(ids)
        print(getWordForId(prediction))
    })
})

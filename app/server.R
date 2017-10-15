library(shiny)
library(tensorflow)
library(data.table)

# File names
modelDir <- 'model'
checkpointDir <- 'log'
modelGraph <- 'model-2048540.meta'
modelValues <- 'model-2048540'

#Load Vocab
vocab <- readRDS(paste('model','vocab.rds', sep = "/"))

# Load RNN
tf$reset_default_graph()
sess <- tf$Session()
saver <- tf$train$import_meta_graph(paste(modelDir, modelGraph, sep = "/"), clear_devices = TRUE)
saver$restore(sess, paste(modelDir, modelValues, sep="/"))

# Transforms a vector of words to a vector of the ids that represent those words.
getWordIds <- function(words){
    ids <- vocab[.(words)]$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

predict <- function(inputs){
    graph <- tf$get_default_graph()
    fetch = "Test/Model/Reshape_1"
    outputs = sess$run(fetch, dict(inputs))
    outputs
}

shinyServer(function(input, output) {
    output$predictions <- renderPrint({
        text <- tolower(input$text_input)
        words <- unlist(strsplit(text, " ", fixed = TRUE))
        ids <- getWordIds(words)
        
        print(ids)
        
        results <- predict(ids)
    })
})

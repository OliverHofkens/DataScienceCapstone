library(shiny)
library(tensorflow)
library(data.table)

modelDir <- 'model'
checkpointDir <- 'log'
modelGraph <- 'model-2048540.meta'
modelValues <- 'model-2048540'

vocab <- readRDS(paste('model','vocab.rds', sep = "/"))

tf$reset_default_graph()

with(tf$Session() %as% sess, {
    model <- tf$train$import_meta_graph(paste(modelDir, modelGraph, sep = "/"), clear_devices = TRUE)
    model$restore(sess, paste(modelDir, modelValues, sep="/"))
})
#checkpointName <- 'checkpoint'
#session <- tf$Session()
#saver <- tf$train$import_meta_graph(paste(modelDir, modelName, sep = ""))
#saver$restore(session, modelDir)

getWordIds <- function(words){
    ids <- vocab[.(words)]$id
    
    unk <- vocab[vocab$word == "<unk>"]$id
    
    # remove words that weren't found with <unk>
    ids[is.na(ids)] <- unk
    
    ids
}

shinyServer(function(input, output) {
    
    output$predictions <- renderPrint({
        text <- tolower(input$text_input)
        words <- unlist(strsplit(text, " ", fixed = TRUE))
        ids <- getWordIds(words)
        
        print(ids)
    })
    
})

library(shiny)
library(tensorflow)
library(data.table)

modelDir <- 'model/'
vocab <- readRDS(paste('model','vocab.rds', sep = "/"))
#checkpointName <- 'checkpoint'
#modelName <- 'model-157668.meta'

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

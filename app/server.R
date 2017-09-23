library(shiny)
library(tensorflow)

vocab <- readRDS('vocab.rds')
modelDir <- 'model/'
checkpointName <- 'checkpoint'
modelName <- 'model-157668.meta'

session <- tf$Session()
saver <- tf$train$import_meta_graph(paste(modelDir, modelName, sep = ""))
saver$restore(session, modelDir)

shinyServer(function(input, output) {
    
    output$predictions <- renderText({
        paste("<p>",getwd(),"</p>")
    })
    
})

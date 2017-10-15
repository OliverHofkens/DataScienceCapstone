library(shiny)
library(tensorflow)
library(keras)
library(data.table)

source("app_helpers.R")

# File names
model <- load_model_hdf5('model/keras_model.h5')

#Load Vocab
vocab <- readRDS(paste('model','vocab.rds', sep = "/"))

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

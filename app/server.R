library(shiny)
library(tensorflow)
library(keras)
library(data.table)

source("app_helpers.R")

model <- load_model_hdf5('model/model.hdf5')
vocab <- readRDS('model/vocab.RDS')

shinyServer(function(input, output) {
    output$predictions <- renderDataTable({
        text <- tolower(input$text_input)
        words <- unlist(strsplit(text, " ", fixed = TRUE))
        words <- tail(words, n=10)
        
        prediction <- predictOnText(model, vocab, words)
        prediction
    })
})

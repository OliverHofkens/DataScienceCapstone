library(shiny)
library(tensorflow)
library(keras)
library(data.table)

source("app_helpers.R")

model <- load_model_hdf5('model/model.hdf5')
vocab <- readRDS('model/vocab.RDS')

shinyServer(function(input, output) {
    output$predictions <- renderText({
        text <- tolower(input$text_input)
        words <- unlist(strsplit(text, " ", fixed = TRUE))
        words <- tail(words, n=7)
        
        prediction <- predictOnText(model, vocab, words)
        
        apply(prediction, 1, function(pred){
            pct <- as.numeric(pred['prob']) * 100
            pctFormatted <- sprintf("%.2f", pct)
            opacity <- max(pred['prob'], 0.1)
            
            paste("<li style='opacity:", opacity ,";'>", pred['word'], " - ", pctFormatted, "%</li>", sep="")
        })
    })
})

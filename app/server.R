library(shiny)
library(tensorflow)
library(keras)
library(data.table)

source("app_helpers.R")

model <- load_model_hdf5('model/model.h5', custom_objects = c(top_3_acc=sparse_top_3_acc))
vocab <- readRDS('model/vocab.RDS')


shinyServer(function(input, output) {
    output$predictions <- renderText({
        prediction <- predictOnText(model, vocab, input$text_input, input$sampling_temp / 100)
        
        apply(prediction, 1, function(pred){
            pct <- as.numeric(pred['prob']) * 100
            pctFormatted <- sprintf("%.2f", pct)
            opacity <- max(pred['prob'], 0.1)
            
            paste("<li style='opacity:", opacity ,";'>", pred['word'], " - ", pctFormatted, "%</li>", sep="")
        })
    })
})

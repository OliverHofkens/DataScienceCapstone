library(shiny)

shinyUI(fluidPage(
  
  titlePanel("RNN Text Predictions"),
    
    mainPanel(
        textInput('text_input', NULL, placeholder = "Start typing here"),
        htmlOutput('predictions')
    )
  )
)

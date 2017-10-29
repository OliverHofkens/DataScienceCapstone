library(shiny)

shinyUI(fluidPage(
  
  titlePanel("RNN Text Predictions"),
  conditionalPanel(
      condition="!output.predictions",
      span("Running first time setup..."),
      br(),
      img(src="images/ajax-loader.gif")
  ),
    conditionalPanel(
        condition="output.predictions",
        textAreaInput('text_input', NULL, placeholder = "Start typing here"),
        dataTableOutput('predictions')
    )
  )
)

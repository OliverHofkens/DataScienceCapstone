library(shiny)

shinyUI(fluidPage(theme = "style.css",
  div(
    titlePanel("NEURAL NET WORD PREDICTIONS"),
    verticalLayout(
      conditionalPanel(
          condition="!output.predictions",
          br(),
          h3("Running first time setup..."),
          br(),
          img(src="images/ajax-loader.gif", class = 'center-block')
      ),
      conditionalPanel(
          condition="output.predictions",
          textAreaInput('text_input', NULL),
          htmlOutput('predictions', container = tags$ul, class = "pred-list")
      )
    ),
    class='col-md-6 col-md-offset-3'
  )
  )
)

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
          sliderInput('sampling_temp', "CREATIVE FREEDOM BOOSTER", 0, 100, value = 0, step = 10, post = "%"),
          htmlOutput('predictions', container = tags$ul, class = "pred-list")
      )
    ),
    class='col-md-6 col-md-offset-3'
  )
  )
)

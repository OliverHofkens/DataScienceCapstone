source("modeling/batcher.R")
library(tensorflow)

LanguageModelInput <- setClass(
    "LanguageModelInput",
    
    slots = c(
        batchSize = "integer",
        numSteps = "integer",
        epochSize = "integer",
        inputData = "ANY",
        targetData = "ANY"
    )
)

LanguageModelInput <- function(config, data, name){
    # Epoch size is the amount of complete batches of data divided by the amount of steps.
    epochSize <- ((length(data) %/% config$batchSize) - 1) %/% config$numSteps
    epochSize <- as.integer(epochSize)
    
    batches <- batchData(data, config$batchSize, config$numSteps, name)
    inputData <- batches[[1]]
    targetData <- batches[[2]]
    
    new("LanguageModelInput", batchSize = config$batchSize, 
        numSteps = config$numSteps, epochSize = epochSize, 
        inputData = inputData, targetData = targetData)
}


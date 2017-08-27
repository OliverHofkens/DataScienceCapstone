source("modeling/loader.R")
source("modeling/input.R")
source("modeling/model.R")
source("modeling/runner.R")
library(tensorflow)

modelConfig <- list(
    initScale = 0.1,
    learningRate = 1,
    maxGradientNorm = 5L,
    numLayers = 2L,
    hiddenSize = 200L,
    maxEpoch = 4L,
    maxMaxEpoch = 13L,
    keepProb = 1,
    learningRateDecay = 0.5,
    vocabSize = 10001L,
    batchSize = 20L,
    numSteps = 20L
)

evalConfig <- modelConfig
evalConfig$batchSize <- 1L
evalConfig$numSteps <- 1L

rawData <- loadModelInputs()
detach('package:quanteda')

with(tf$Graph()$as_default(), {
    initializer = tf$random_uniform_initializer(-1 * modelConfig$initScale, modelConfig$initScale)
    
    with(tf$name_scope("Train"), {
        trainInput <- LanguageModelInput(modelConfig, rawData$train, "TrainInput")
        with(tf$variable_scope("Model", reuse=FALSE, initializer=initializer),{
            m <- LanguageModel(TRUE, modelConfig, trainInput)
            tf$summary$scalar("Training Loss", m@cost)
            tf$summary$scalar("Learning Rate", m@learningRate)
        })
    })
    
    with(tf$name_scope("Valid"), {
        validInput <- LanguageModelInput(modelConfig, rawData$validation, "ValidationInput")
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            mValid <- LanguageModel(FALSE, modelConfig, validInput)
            tf$summary$scalar("Validation Loss", mValid@cost)
        })
    })
    
    with(tf$name_scope("Test"), {
        testInput <- LanguageModelInput(evalConfig, rawData$test, "TestInput")
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            mTest <- LanguageModel(FALSE, evalConfig, testInput)
        })
    })
    
    # dir.create("log", showWarnings = FALSE)
    # supervisor <- tf$train$Supervisor(logdir=paste(getwd(), 'log', sep = "/"))
    # 
    # with(supervisor$managed_session() %as% sess, {
    #     for(i in seq.int(from = 0L, to = (modelConfig$maxMaxEpoch - 1L))){
    #         lrDecay = modelConfig$learningRateDecay ** max(i + 1 - modelConfig$maxEpoch, 0.0)
    #         assignLearningRate(m, sess, modelConfig$learningRate * lrDecay)
    #         
    #         cat(sprintf("Epoch: %d Learning rate: %.3f\n", i + 1, sess$run(m@learningRate))
    #         trainPerplexity = runEpoch(sess, m, evalOp=m@trainOp, verbose=TRUE)
    #         cat(sprintf("Epoch: %d Train Perplexity: %.3f\n",  i + 1, trainPerplexity))
    #         
    #         validerplexity = runEpoch(sess, mValid)
    #         cat(sprintf("Epoch: %d Valid Perplexity: %.3f\n", i + 1, validPerplexity))
    #     }
    #     
    #     testPerplexity = runEpoch(sess, mTest)
    #     cat(sprintf("Test Perplexity: %.3f\n", testPerplexity))
    #         
    #     print("Saving model\n")
    #     supervisor$saver$save(sess, paste(getwd(), 'model', sep = "/"), global_step=supervisor$global_step)
    # })
})

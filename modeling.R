source("modeling/loader.R")
library(tensorflow)
use_virtualenv("~/.virtualenvs/r-tensorflow/")

# Actual config:
modelConfig <- list(
    numberOfWords = 5L, # Number of input words to base a prediction on.
    learningRate = 0.001,
    trainingIterations = 500L,
    printInfoAfterXIterations = 100L,
    nHiddenLayers = 200L,
    vocabSize = 10001L
)

rawData <- loadModelInputs()
detach('package:quanteda')

tf$reset_default_graph()

# Tensor that takes the input words
inputTensor <- tf$placeholder(tf$float32, 
                              shape(NULL, modelConfig$numberOfWords, 1L))
# Tensor that holds the expected prediction
rightPredictionTensor <- tf$placeholder(tf$float32, 
                                        shape(NULL, modelConfig$vocabSize))

# RNN output node weights and biases
weights = dict(
    'out' = tf$Variable(tf$random_normal(
        shape(modelConfig$nHiddenLayers, modelConfig$vocabSize)
        ))
    )

biases = dict(
    'out' = tf$Variable(tf$random_normal(
        shape(modelConfig$vocabSize)
        ))
)

with(tf$variable_scope("RNN"), {
    # Reshape the input to length-5 blocks
    inputs <- tf$reshape(inputTensor, shape(-1L, modelConfig$numberOfWords))
    
    # Split the inputs into subtensors of length 5
    inputs <- tf$split(inputs, modelConfig$numberOfWords, 1L)
    
    # Stack 2 LSTM cells
    rnnCell = tf$contrib$rnn$MultiRNNCell(list(
        tf$contrib$rnn$BasicLSTMCell(modelConfig$nHiddenLayers),
        tf$contrib$rnn$BasicLSTMCell(modelConfig$nHiddenLayers)
    ))
    
    # Generates predictions
    results = tf$contrib$rnn$static_rnn(rnnCell, inputs, dtype=tf$float32)
    outputs <- results[[1]]
    state <- results[[2]]
    
    # there are numberOfWords outputs but we only want the last output
    prediction <- tf$matmul(outputs[length(outputs)], weights[['out']]) + biases[['out']]
})

# Loss and optimizer
cost <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(
    logits=prediction, labels=rightPredictionTensor)
    )
optimizer <- tf$train$RMSPropOptimizer(learning_rate=modelConfig$learningRate)$minimize(cost)

# Compare the predictions with the correct values:
correctPredictions <- tf$equal(tf$argmax(prediction, 1), tf$argmax(rightPredictionTensor,1))
accuracy <- tf$reduce_mean(tf$cast(correctPredictions, tf$float32))

# Initializer
init <- tf$global_variables_initializer()

# with(tf$Session %as% sess, {
#     sess$run(init)
#     
#     step = 0L
#     offset = random.randint(0,n_input+1)
#     end_offset = n_input + 1
#     totalAccuracy = 0
#     totalLoss = 0
#     
#     while(step < modelConfig$trainingIterations){
#         # Take numberOfWords and the correct next word:
#         inputWords <- rawData
#     }
# })

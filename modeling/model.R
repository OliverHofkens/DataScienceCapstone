source("modeling/input.R")

LanguageModel <- setClass(
    "LanguageModel",
    
    slots = c(
        input = "LanguageModelInput",
        initialState = "ANY",
        finalState = "ANY",
        cost = "ANY",
        learningRate = "ANY",
        trainOp = "ANY",
        newLearningRate = "ANY",
        learningRateUpdate = "ANY"
    )
)

LanguageModel <- function(isTraining, config, input){
    model <- new("LanguageModel", input = input)
    
    # Create an LSTM cell;
    makeLstmCell <- function(){
        tf$contrib$rnn$BasicLSTMCell(config$hiddenSize, forget_bias=0.0)
    }
    
    # An attention cell is just a LSTM cell, unless we're in training and a keepProb below 100% was set:
    makeAttnCell <- makeLstmCell
    if(isTraining && config$keepProb < 1){
        makeAttnCell <- function(){
            tf$contrib$rnn$DropoutWrapper(cell, output_keep_prob = config$keepProb)
        }
    }
    
    # Create the multiple layered RNN cells
    cell <- tf$contrib$rnn$MultiRNNCell(replicate(config$numLayers, makeAttnCell()), state_is_tuple=TRUE)
    model@initialState <- cell$zero_state(config$batchSize, tf$float32)
    
    # Create the embedding variable and lookup
    embedding <- tf$get_variable("embedding", shape = shape(config$vocabSize, config$hiddenSize), dtype = tf$float32)
    inputs <- tf$nn$embedding_lookup(embedding, input@inputData)
    
    # Add dropout if configured
    if (isTraining && config$keepProb < 1){
        inputs = tf$nn$dropout(inputs, config$keepProb)
    }
    
    # Build the actual RNN layers
    outputs <- list()
    state <- model@initialState
    
    with(tf$variable_scope("RNN"), {
        for(timeStep in seq.int(from = 0L, to = (config$numSteps - 1L))){
            if(timeStep > 0L){
                tf$get_variable_scope()$reuse_variables()
            } 
            results = cell(inputs[, timeStep, ], state)
            outputs = c(outputs, results[1])
            state = results[[2]]
        }
    })
    
    # Outputs
    output = tf$reshape(tf$stack(axis=1L, values=outputs), list(-1L, config$hiddenSize))
    softmaxW = tf$get_variable("softmax_w", c(config$hiddenSize, config$vocabSize), dtype=tf$float32)
    softmaxB = tf$get_variable("softmax_b", c(config$vocabSize), dtype=tf$float32)
    logits = tf$matmul(output, softmaxW) + softmaxB
    
    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf$reshape(logits, c(config$batchSize, config$numSteps, config$vocabSize))
    
    # use the contrib sequence loss and average over the batches
    loss = tf$contrib$seq2seq$sequence_loss(
        logits,
        input@targetData,
        tf$ones(list(config$batchSize, config$numSteps), dtype=tf$float32),
        average_across_timesteps=FALSE,
        average_across_batch=TRUE
    )
    
    # Update the cost variables
    model@cost <- tf$reduce_sum(loss)
    model@finalState <- state
    
    if(!isTraining){
        return(model)
    }
    
    model@learningRate <- tf$Variable(0.0, trainable=FALSE)
    tvars = tf$trainable_variables()
    grads = tf$clip_by_global_norm(tf$gradients(model@cost, tvars), config$maxGradientNorm)
    grads = grads[[1]]
    optimizer = tf$train$GradientDescentOptimizer(model@learningRate)
    zipped = mapply(c, grads, tvars, SIMPLIFY = FALSE)
    model@trainOp <- optimizer$apply_gradients(zipped, global_step=tf$contrib$framework$get_or_create_global_step())
    
    model@newLearningRate <- tf$placeholder(tf$float32, shape=c(), name="new_learning_rate")
    model@learningRateUpdate <- tf$assign(model@learningRate, model@newLearningRate)
    
    model
}

# Define the generic method:
setGeneric(name="assignLearningRate",
          def=function(model, session, value)
          {
              standardGeneric("assignLearningRate")
          }
)

# Implement it for our languageModel:
setMethod(f="assignLearningRate",
          signature="LanguageModel",
          definition=function(model, session, value)
          {
              newLearningRate <- model@newLearningRate
              session$run(model@learningRateUpdate, feed_dict=dict(newLearningRate = value))
          }
)

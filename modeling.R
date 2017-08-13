library(tensorflow)

languageModel <- function(){
    # Adjustable parameters:
    init_scale <- 0.1
    learning_rate <- 1.0
    max_grad_norm <- 5L
    num_layers <- 2L
    hidden_size <- 200L
    max_epoch <- 4L
    max_max_epoch <- 13L
    keep_prob <- 1.0
    lr_decay <- 0.5
    vocab_size <- 10000L
    
    epochSize <- 0L
    cellLayers <- 0L
    initialState <- 0L
    finalState <- 0L
    cost <- 0L
    learningRate <- 0L
    lrUpdate <- 0L
    newLr <- 0L
    lr <- 0L
    num_steps <- 0L
    trainOp <- 0L
    
    # Create an LSTM cell;
    makeLstmCell <- function(){
        tf$contrib$rnn$BasicLSTMCell(hidden_size, forget_bias=0.0)
    }
    
    # Create an attention cell:
    makeAttnCell <- function(isTraining){
        cell <- makeLstmCell()
        
        if(isTraining && keep_prob < 1){
            cell <- tf$contrib$rnn$DropoutWrapper(cell, output_keep_prob = keep_prob)
        }
        
        cell
    }
    
    # Layer multiple RNN cells together
    makeLayeredCells <- function(isTraining, batchSize){
        cellLayers <<- tf$contrib$rnn$MultiRNNCell(replicate(num_layers, makeAttnCell(isTraining)))
        initialState <<- cellLayers$zero_state(batchSize, tf$float32)
    }
    
    runModel <- function(isTraining, data, targets, batchSize = 20L, numSteps = 20L){
        num_steps <<- numSteps
        epochSize <<- ((length(data) %/% batchSize) - 1) %/% numSteps
        
        makeLayeredCells(isTraining, batchSize)
        
        # Create the embedding variable
        embedding <- tf$get_variable("embedding", shape = shape(vocab_size, hidden_size), dtype = tf$float32)
        # Lookup inputs:
        inputs <- tf$nn$embedding_lookup(embedding, data)
        
        if (isTraining && keep_prob < 1){
            inputs = tf$nn$dropout(inputs, keep_prob)
        }
           
        outputs <- list()
        state <- initialState
        
        with(tf$variable_scope("RNN"), {
            for(time_step in seq.int(from = 0L, to = (numSteps - 1L))){
                if(time_step > 0L){
                    tf$get_variable_scope()$reuse_variables()
                } 
                results = cellLayers(inputs[, time_step, ], state)
                outputs = c(outputs, results[1])
                state = results[[2]]
            }
        })
        
        output = tf$reshape(tf$stack(axis=1L, values=outputs), list(-1L, hidden_size))
        softmax_w = tf$get_variable("softmax_w", c(hidden_size, vocab_size), dtype=tf$float32)
        softmax_b = tf$get_variable("softmax_b", c(vocab_size), dtype=tf$float32)
        logits = tf$matmul(output, softmax_w) + softmax_b
            
        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf$reshape(logits, c(batchSize, numSteps, vocab_size))
        
        # use the contrib sequence loss and average over the batches
        loss = tf$contrib$seq2seq$sequence_loss(
            logits,
            targets,
            tf$ones(list(batchSize, numSteps), dtype=tf$float32),
            average_across_timesteps=FALSE,
            average_across_batch=TRUE
        )
        
        # update the cost variables
        cost <<- tf$reduce_sum(loss)
        finalState <<- state
        
        if(!isTraining){
            return(list(cost = cost))
        }
        
        lr <<- tf$Variable(0.0, trainable=FALSE)
        tvars = tf$trainable_variables()
        grads = tf$clip_by_global_norm(tf$gradients(cost, tvars), max_grad_norm)
        grads = grads[[1]]
        optimizer = tf$train$GradientDescentOptimizer(lr)
        zipped = mapply(c, grads, tvars, SIMPLIFY = FALSE)
        trainOp <<- optimizer$apply_gradients(zipped, global_step=tf$contrib$framework$get_or_create_global_step())
        
        newLr <<- tf$placeholder(tf$float32, shape=c(), name="new_learning_rate")
        lrUpdate <<- tf$assign(lr, newLr)
        
        list(cost = cost, lr = lr)
    }
    
    assignLr <- function(session, value){
        session$run(lrUpdate, feed_dict=dict(newLr = value))
    }
    
    getLr <- function(){
        lr
    }
    
    getCost <- function(){
        cost
    }
    
    getInitialState <- function(){
        initialState
    }
    
    getFinalState <- function(){
        finalState
    }
    
    getEpochSize <- function(){
        epochSize
    }
    
    getNumSteps <- function(){
        num_steps
    }
    
    getTrainOp <- function(){
        trainOp
    }
    
    list(
        runModel = runModel, 
        assignLr = assignLr, 
        getLr = getLr, 
        getInitialState = getInitialState, 
        getCost = getCost,
        getFinalState = getFinalState,
        getEpochSize = getEpochSize,
        getNumSteps = getNumSteps,
        getTrainOp = getTrainOp
        )
}

runEpoch <- function(session, model, eval_op = NA, verbose = FALSE){
    costs = 0.0
    iters = 0L
    state = session$run(model$getInitialState())
    
    fetches = list(cost = model$getCost(), finalState = model$getFinalState())
    if(!is.na(eval_op)){
        fetches$evalOp <- eval_op
    }
    
    for(i in seq.int(from = 0L, to = model$getEpochSize())){
        feed_dict = dict()
        
        index = 1L
        initialState <- model$getInitialState()
        for(item in initialState){
            c <- item$c
            h <- item$h
            feed_dict[[c]] = state[[index]]$c
            feed_dict[[h]] = state[[index]]$h
            index = index + 1
        }
        
        vals = session$run(fetches, feed_dict)
        cost = vals$cost
        state = vals$finalState
        
        costs = costs + cost
        iters = iters + model$getNumSteps()
        
        if(verbose && step %% (model$getEpochSize() %/% 10) == 10){
            sprintf("%.3f perplexity: %.3f", step * 1.0 / model$getEpochSize(), exp(costs / iters))
        }
    }
       
    return(exp(costs/iters))
}

with(tf$Graph()$as_default(), {
    initializer = tf$random_uniform_initializer(-1 * init_scale, init_scale)
    
    trainingInput <- batchData(train, 20L, 20L)
    with(tf$name_scope("Train"), {
        with(tf$variable_scope("Model", reuse=FALSE, initializer=initializer),{
            mTrain = languageModel()
            results <- mTrain$runModel(isTraining = TRUE, data = trainingInput[[1]], targets = trainingInput[[2]])
            tf$summary$scalar("Training Loss", results$cost)
            tf$summary$scalar("Learning Rate", results$lr)
        })
    })
    
    validationInput <- batchData(validation, 20L, 20L)
    with(tf$name_scope("Valid"), {
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            mValid = languageModel()
            results <- mValid$runModel(isTraining = FALSE, data = validationInput[[1]], targets = validationInput[[2]])
            tf$summary$scalar("Validation Loss", results$cost)
        })
    })
    
    testInput <- batchData(test, 1L, 1L)
    with(tf$name_scope("Test"), {
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            mTest = languageModel()
            results <- mTest$runModel(isTraining = FALSE, data = testInput[[1]], targets = testInput[[2]], 1L, 1L)
        })
    })
    
    supervisor <- tf$train$Supervisor(logdir=getwd())
    
    with(supervisor$managed_session() %as% sess, {
        for(i in seq.int(from = 0L, to = 12L)){
            lr_decay = 0.5 ** max(i + 1 - 4, 0.0)
            mTrain$assignLr(sess, 1 * lr_decay)
            
            sprintf("Epoch: %d Learning rate: %.3f", i + 1, sess$run(mTrain$getLr()))
            train_perplexity = runEpoch(sess, mTrain, eval_op=mTrain$getTrainOp(), verbose=TRUE)
            sprintf("Epoch: %d Train Perplexity: %.3f",  i + 1, train_perplexity)
            
            valid_perplexity = runEpoch(sess, mValid)
            sprintf("Epoch: %d Valid Perplexity: %.3f", i + 1, valid_perplexity)
        }
           
        test_perplexity = runEpoch(sess, mTest)
        sprintf("Test Perplexity: %.3f", test_perplexity)
            
        print("Saving model")
        supervisor$saver$save(sess, getwd(), global_step=supervisor.global_step)
    })
})












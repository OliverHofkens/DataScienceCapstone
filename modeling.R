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
    
    cellLayers <- 0L
    initialState <- 0L
    
    cost <- 0L
    learningRate <- 0L
    
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
    makeLayeredCells <- function(isTraining){
        cellLayers <<- tf$contrib$rnn$MultiRNNCell(replicate(num_layers, makeAttnCell(isTraining)))
        initialState <<- cellLayers$zero_state(batch_size, tf$float32)
    }
    
    runModel <- function(isTraining, data, targets, batch_size = 20L, num_steps = 20L){
        makeLayeredCells(isTraining)
        
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
            for(time_step in seq.int(from = 0L, to = (num_steps - 1L))){
                if(time_step > 0L){
                    tf$get_variable_scope()$reuse_variables()
                } 
                # Line below changed:
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
        logits = tf$reshape(logits, c(batch_size, num_steps, vocab_size))
        
        # use the contrib sequence loss and average over the batches
        loss = tf$contrib$seq2seq$sequence_loss(
            logits,
            targets,
            tf$ones(list(batch_size, num_steps), dtype=tf$float32),
            average_across_timesteps=FALSE,
            average_across_batch=TRUE
        )
        
        # update the cost variables
        cost = tf$reduce_sum(loss)
        final_state = state
        
        if(!isTraining){
            return(list(cost = cost))
        }
        
        lr = tf$Variable(0.0, trainable=FALSE)
        tvars = tf$trainable_variables()
        grads = tf$clip_by_global_norm(tf$gradients(cost, tvars), max_grad_norm)
        grads = grads[[1]]
        optimizer = tf$train$GradientDescentOptimizer(lr)
        zipped = mapply(c, grads, tvars, SIMPLIFY = FALSE)
        train_op = optimizer$apply_gradients(zipped, global_step=tf$contrib$framework$get_or_create_global_step())
        
        new_lr = tf$placeholder(tf$float32, shape=c(), name="new_learning_rate")
        lr_update = tf$assign(lr, new_lr)
        
        list(cost = cost, lr = lr)
    }
    
    list(runModel = runModel)
}

with(tf$Graph()$as_default(), {
    initializer = tf$random_uniform_initializer(-1 * init_scale, init_scale)
    
    trainingInput <- batchData(train, 20L, 20L)
    with(tf$name_scope("Train"), {
        with(tf$variable_scope("Model", reuse=FALSE, initializer=initializer),{
            m = languageModel()
            results <- m$runModel(isTraining = TRUE, data = trainingInput[[1]], targets = trainingInput[[2]])
            tf$summary$scalar("Training Loss", results$cost)
            tf$summary$scalar("Learning Rate", results$lr)
        })
    })
    
    validationInput <- batchData(validation, 20L, 20L)
    with(tf$name_scope("Valid"), {
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            m = languageModel()
            results <- m$runModel(isTraining = FALSE, data = validationInput[[1]], targets = validationInput[[2]])
            tf$summary$scalar("Validation Loss", results$cost)
        })
    })
    
    testInput <- batchData(test, 1L, 1L)
    with(tf$name_scope("Test"), {
        with(tf$variable_scope("Model", reuse=TRUE, initializer=initializer),{
            m = languageModel()
            results <- m$runModel(isTraining = FALSE, data = testInput[[1]], targets = testInput[[2]], 1L, 1L)
        })
    })
})












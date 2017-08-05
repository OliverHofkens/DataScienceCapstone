library(tensorflow)

languageModel <- function(){
    # Adjustable parameters:
    init_scale <- 0.1
    learning_rate <- 1.0
    max_grad_norm <- 5
    num_layers <- 2
    num_steps <- 20
    hidden_size <- 200
    max_epoch <- 4
    max_max_epoch <- 13
    keep_prob <- 1.0
    lr_decay <- 0.5
    batch_size <- 20
    vocab_size <- 10000
    
    cellLayers <- 0L
    initialState <- 0L
    
    cost <- 0L
    learningRate <- 0L
    
    # Create an LSTM cell;
    makeLstmCell <- function(){
        tf$contrib$rnn$BasicLSTMCell(hidden_size, forget_bias=0.0)
    }
    
    # Create an attention cell:
    makeAttnCell <- function(is_training){
        cell <- makeLstmCell()
        
        if(is_training && keep_prob < 1){
            cell <- tf$contrib$rnn$DropoutWrapper(cell, output_keep_prob = keep_prob)
        }
        
        cell
    }
    
    # Layer multiple RNN cells together
    makeLayeredCells <- function(is_training){
        cellLayers <<- tf$contrib$rnn$MultiRNNCell(replicate(num_layers, makeAttnCell(is_training)))
        initialState <<- cellLayers$zero_state(batch_size, tf$float32)
    }
    
    runModel <- function(isTraining, data, targets){
        makeLayeredCells(TRUE)
        
        # Create the embedding variable
        embedding <- tf$get_variable("embedding", shape = shape(vocab_size, hidden_size), dtype = tf$float32)
        # Lookup inputs:
        inputs <- tf$nn$embedding_lookup(embedding, data)
        
        if (isTraining && keep_prob < 1){
            inputs = tf$nn$dropout(inputs, keep_prob)
        }
           
        outputs <- c()
        state <- initialState
        
        with(tf$variable_scope("RNN"), {
            for(time_step in seq(0L, num_steps)){
                if(time_step > 0L){
                    tf$get_variable_scope()$reuse_variables()
                } 
                results = cellLayers(inputs[, time_step, ], state)
                outputs = c(outputs, results$cell_output)
                stats = results$state
            }
        })
        
        output = tf$reshape(tf$stack(axis=1, values=outputs), c(-1L, size))
        softmax_w = tf$get_variable("softmax_w", c(size, vocab_size), dtype=tf$float32)
        softmax_b = tf$get_variable("softmax_b", c(vocab_size), dtype=tf$float32)
        logits = tf$matmul(output, softmax_w) + softmax_b
            
        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf$reshape(logits, c(batch_size, num_steps, vocab_size))
        
        # use the contrib sequence loss and average over the batches
        loss = tf$contrib$seq2seq$sequence_loss(
            logits,
            targets,
            tf$ones(c(batch_size, num_steps), dtype=tf$float32),
            average_across_timesteps=FALSE,
            average_across_batch=TRUE
        )
        
        # update the cost variables
        cost = tf$reduce_sum(loss)
        final_state = state
        
        if(!is_training){
            return()
        }
        
        lr = tf$Variable(0.0, trainable=FALSE)
        tvars = tf$trainable_variables()
        grads = tf$clip_by_global_norm(tf$gradients(cost, tvars), max_grad_norm)
        optimizer = tf$train$GradientDescentOptimizer(lr)
        train_op = optimizer$apply_gradients(zip(grads, tvars), global_step=tf$contrib$framework$get_or_create_global_step())
        
        new_lr = tf$placeholder(tf$float32, shape=c(), name="new_learning_rate")
        lr_update = tf$assign(lr, new_lr)
        
        list(cost = cost, lr = lr)
    }
    
    list(runModel = runModel)
}

with(tf$Graph()$as_default(), {
    initializer = tf$random_uniform_initializer(-1 * init_scale, init_scale)
    
    trainingTest <- batchData(train, 20L, 20L)
    with(tf$name_scope("Train"), {
        with(tf$variable_scope("Model", reuse=FALSE, initializer=initializer),{
            m = languageModel()
            results <- m$runModel(isTraining = TRUE, data = trainingTest, targets = trainingTest)
            tf$summary$scalar("Training Loss", m$cost)
            tf$summary$scalar("Learning Rate", m$lr)
        })
    })
})












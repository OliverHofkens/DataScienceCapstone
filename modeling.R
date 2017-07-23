library(tensorflow)

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

makeLayeredCells(TRUE)

# Create the embedding variable
embedding <- tf$get_variable("embedding", shape = c(vocab_size, hidden_size), dtype = tf$float32)
# Lookup inputs:
inputs <- tf$nn$embedding_lookup(embedding, input_.input_data)

library(tensorflow)

# Creates batches of data from the complete input set.
batchData <- function(data, batchSize, steps, name){
    batches <- floor(length(data) / batchSize)
    lastBatchedElement <- (batches * batchSize) - 1
    
    with(tf$name_scope(name, 'batcher', list(data, batchSize, steps)), {
        # Load the data
        data = tf$convert_to_tensor(data, name="raw_data", dtype=tf$int32)
        
        # Calculate how many batches there are, and reshape the vector to a table of batches.
        dataLen = tf$size(data)
        batchLen = dataLen %/% batchSize
        data = tf$reshape(data[0L : lastBatchedElement], list(batchSize, batchLen))
        
        # The last batch is incomplete because data is not exactly a multiple of batchSize
        epochSize = (batchLen - 1L) %/% steps
        assertion = tf$assert_positive(epochSize, message="epochSize == 0, decrease batchSize or numSteps")
        with(tf$control_dependencies(list(assertion)), {
            epochSize = tf$identity(epochSize, name="epoch_size")
        })
        
        # Create the actual batches
        i = tf$train$range_input_producer(epochSize, shuffle=FALSE)$dequeue()
        
        x = tf$strided_slice(data, list(0L, i * steps), list(batchSize, (i + 1L) * steps))
        x$set_shape(shape(batchSize, steps))
        
        y = tf$strided_slice(data, list(0L, i * steps + 1L), list(batchSize, (i + 1L) * steps + 1L))
        y$set_shape(shape(batchSize, steps))
        
        return(c(x,y))
    })
}

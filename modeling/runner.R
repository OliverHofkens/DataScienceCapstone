runEpoch <- function(session, model, evalOp = NA, verbose = FALSE){
    costs = 0.0
    iters = 0L
    state = session$run(model@initialState)
    
    fetches = dict(cost = model@cost, finalState = model@finalState)
    if(!is.na(evalOp)){
        fetches$evalOp <- evalOp
    }
    
    for(i in seq.int(from = 0L, to = (model@input@epochSize - 1))){
        feedDict = dict()
        
        index = 1L
        initialState <- model@initialState
        for(item in initialState){
            c <- item$c
            h <- item$h
            feedDict[[c]] = state[[index]]$c
            feedDict[[h]] = state[[index]]$h
            index = index + 1
        }
        
        vals = session$run(fetches, feedDict)
        cost = vals$cost
        state = vals$finalState
        
        costs = costs + cost
        iters = iters + model@input@numSteps
        
        if(verbose && i %% (model@input@epochSize %/% 10) == 10){
            cat(sprintf("%.3f perplexity: %.3f\n", i * 1.0 / model@input@epochSize, exp(costs / iters)))
        }
    }
    
    return(exp(costs/iters))
}
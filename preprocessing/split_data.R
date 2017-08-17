splitData <- function(directory){
    # Read all lines of all files:
    files <- as.character(list.files(path=directory))
    
    allText <- vector(mode = "character")
    for(file in files){
        lines <- readLines(paste(directory,"/",file,sep=""))
        allText <- c(allText, lines)
    }
    
    set.seed(1)
    
    # Split total in train vs non-train
    inTrain <- as.logical(rbinom(length(allText), 1, 0.6))
    train <- allText[inTrain]
    notTrain <- allText[!inTrain]
    
    # split non-train in validation vs test
    inValidation <- as.logical(rbinom(length(notTrain), 1, 0.5))
    validation <- notTrain[inValidation]
    test <- notTrain[!inValidation]
    
    target = 'data/split'
    dir.create(target, showWarnings = FALSE)
    
    # Write out definite files for model training:
    writeLines(train, paste(target, 'train.txt', sep="/"))
    writeLines(validation, paste(target, 'validation.txt', sep="/"))
    writeLines(test, paste(target, 'test.txt', sep="/"))
}

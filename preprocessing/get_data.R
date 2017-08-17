getRawData <- function(language){
    dataDir <- 'data'
    dataSrc <- paste(dataDir, 'Coursera-SwiftKey.zip', sep = "/")
    dataTarget <- paste(dataDir,'raw', sep = "/")
    
    dir.create(dataDir, recursive = TRUE, showWarnings = FALSE)
    
    if(!dir.exists(dataTarget)){
        if(!file.exists(dataSrc)){
            download.file('https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip', dataSrc)
        }
        unzip(dataSrc, exdir = dataTarget)
    }
    
    paste(dataTarget, 'final' ,language, sep="/")
}
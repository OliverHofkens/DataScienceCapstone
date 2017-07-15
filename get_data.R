getRawData <- function(language){
    datadir <- 'data'
    datasource <- 'data/Coursera-SwiftKey.zip'
    dataset <- 'data/final'
    dir.create(datadir, showWarnings = FALSE)
    
    
    if(!dir.exists(dataset)){
        if(!file.exists(datasource)){
            download.file('https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip', datasource)
        }
        unzip(datasource, exdir = 'data')
    }
    
    paste(dataset,language, sep="/")
}
library(tm)
library(quanteda)

# Replace rare words with '<unk>'
replaceWithUnk <- function(x, commons){
    words <- unlist(strsplit(x, split = " "))
    selection = !(words %in% commons)
    words[selection] <- "<unk>"
    paste(words, collapse = " ")
}

# Vectorize the above function
replaceWithUnkVector <- function(x, commons){
    sapply(x, replaceWithUnk, commons = commons)
}

transformData <- function(data_dir){
    corp <- VCorpus(DirSource(data_dir, encoding="UTF-8"), readerControl = list(reader = readPlain))
    
    quantCorp <-  corpus(corp)
    
    # Get the most common words from the training set to establish a vocabulary:
    train <- corpus_subset(quantCorp, id == 'training.txt')
    trainTokens <- tokens(train, what = "fasterword")
    trainDfm <- dfm(trainTokens)
    commonWords <- names(topfeatures(trainDfm, 10000))
    
    # Apply the filter to the corpus
    newCorpus <- tm_map(corp, content_transformer(replaceWithUnkVector), commons = commonWords, mc.cores=7)
    
    # Write the new corpus to txt files:
    newDir <- 'data/model_input'
    dir.create(newDir, showWarnings = FALSE, recursive = TRUE)
    writeCorpus(newCorpus, newDir, filenames = c('test.txt','training.txt', 'validation.txt'))
}

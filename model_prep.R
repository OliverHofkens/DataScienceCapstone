source('clean_data.R')
library(quanteda)

tmCorpus <- corpus
rm(corpus)
quantCorp <-  corpus(tmCorpus)

# Get the most common words:
quantCorpDfm <- dfm(quantCorp)
commonWords <- names(topfeatures(quantCorpDfm, 10000))

# Replace rare words with '<unk>'
replaceWithUnk <- function(x, commons){
    words <- unlist(strsplit(x, split = " "))
    selection = !(words %in% commons)
    words[selection] <- "<unk>"
    words
    paste(words, collapse = " ")
}

replaceWithUnkVector <- function(x, commons){
    sapply(x, replaceWithUnk, commons = commons)
}

newCorpus <- tm_map(tmCorpus, content_transformer(replaceWithUnkVector), commons = commonWords, mc.cores=7)

directory <- 'data/modelprep'
dir.create(directory, showWarnings = FALSE)
writeCorpus(newCorpus, directory)

# Read all lines of all files:
files <- as.character(list.files(path=directory))
allText <- vector(mode = "character")
for(file in files){
    lines <- readLines(paste(directory,"/",file,sep=""))
    allText <- c(allText, lines)
}
rm(lines)
 
# Split in training/validation/test
set.seed(1)

inTrain <- as.logical(rbinom(length(allText), 1, 0.6))
train <- allText[inTrain]
notTrain <- allText[!inTrain]
rm(inTrain, allText)
inValidation <- as.logical(rbinom(length(notTrain), 1, 0.5))
validation <- notTrain[inValidation]
test <- notTrain[!inValidation]
rm(notTrain, inValidation)

# Write out definite files for model training:
writeLines(train, paste(directory, "/", 'train.txt', sep=""))
writeLines(validation, paste(directory, "/", 'validation.txt', sep=""))
writeLines(test, paste(directory, "/", 'test.txt', sep=""))

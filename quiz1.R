source('clean_data.R')

englishCorpus <- getCorpus('en_US')
#englishCorpus <- tm_map(englishCorpus, stripWhitespace)

### 3
getLongestLineLength <- function(corpus, document){
    longest <- 0
    
    for(line in as.character(corpus[[document]])){
        count <- nchar(line)
        
        if(count > longest){
            longest <- count
        }
    }
    
    longest
}

longestBlog <- getLongestLineLength(englishCorpus, 'en_US.blogs.txt')
longestNews <- getLongestLineLength(englishCorpus, 'en_US.news.txt')

### 4
getWordRatio <- function(word1, word2, corpus, document){
    lines <- as.character(corpus[[document]])
  
    word1Count <- sum(grepl(word1, lines))
    word2Count <- sum(grepl(word2, lines))
   
    word1Count / word2Count
}

loveHate <- getWordRatio('love', 'hate', englishCorpus, 'en_US.twitter.txt')

### 5 & 6
findInCorpus <- function(search, corpus, document){
    lines = as.character(corpus[[document]])
    matches = grepl(search, lines)
    
    lines[matches]
}

biostats <- findInCorpus('biostats', englishCorpus, 'en_US.twitter.txt')
computerTweets <- findInCorpus('A computer once beat me at chess, but it was no match for me at kickboxing', englishCorpus, 'en_US.twitter.txt')
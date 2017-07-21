source('clean_data.R')
library(quanteda)

corp <-  corpus(corpus)
rm(corpus)

summary(corp)

# With stopwords still included:
features <- dfm(corp)
topfeatures(features, 100)

set.seed(100)
textplot_wordcloud(features, min.freq = 100000, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))

rm(features)

# Without stopwords:
featuresNoStop <- dfm(corp, remove = stopwords("english"), remove_punct = TRUE)
topfeatures(featuresNoStop, 100)

textplot_wordcloud(featuresNoStop, min.freq = 50000, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))
rm(featuresNoStop)

# Checking out bi-grams
bigrams <- tokens(corp, ngrams = 2)
featuresBiGrams <- dfm(bigrams) 
topfeatures(featuresBiGrams, 100)

rm(bigrams)
rm(featuresBiGrams)

# Checking out tri-grams
trigrams <- tokens(corp, ngrams = 3)
featuresTriGrams <- dfm(trigrams)
topfeatures(featuresTriGrams, 100)

source("modeling/loader.R")

#inputs <- loadModelInputs()
#vocab <-inputs$vocabulary

#vectors <- read.table("data/glove.6B/glove.6B.200d.txt", quote="", comment.char="")
embeddingMatrix <- matrix(nrow=10001, ncol = 200)

merged <- merge(vocab, vectors, by.x = "word", by.y = "V1", all.x = TRUE)
merged <- merged[order(id)]
merged[is.na(merged)] <- runif(sum(is.na(merged)), -1, 1)

embeddingMatrix <- merged[,3:202]
colnames(embeddingMatrix) <- NULL

zeros <- matrix(0, nrow=1, ncol=200)

embeddingMatrix <- rbind(zeros, embeddingMatrix)
embeddingMatrix <- as.matrix(embeddingMatrix)
saveRDS(embeddingMatrix, "matrix.RDS")

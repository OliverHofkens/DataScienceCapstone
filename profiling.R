gen <- inputGenerator(train, vocab, config)

Rprof(tmp <- tempfile(), append=TRUE)
res <- replicate(n = 100, gen())
Rprof()

summaryRprof(tmp)
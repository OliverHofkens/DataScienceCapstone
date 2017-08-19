source("preprocessing/get_data.R")
source("preprocessing/split_data.R")
source("preprocessing/clean_data.R")
source("preprocessing/transform_data.R")

language <- 'en_US'
Sys.setlocale("LC_ALL", "nl_BE.UTF-8")

# Downloads the raw data
rawEnglish <- getRawData(language)

# Splits the data into training/validation/test
splitData(rawEnglish)

# Cleans the data
cleanEnglish <- getCleanData(language)

# Do the final transformations for model training
transformData(cleanEnglish)

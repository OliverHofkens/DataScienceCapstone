source("preprocessing/get_data.R")
source("preprocessing/split_data.R")
source("preprocessing/clean_data.R")

language <- 'en_US'

# Downloads the raw data
rawEnglish <- getRawData(language)

# Splits the data into training/validation/test
splitData(rawEnglish)

# Cleans the data
# ! TODO: This messes up all quotes and other signs:
cleanEnglish <- getCleanData(language)

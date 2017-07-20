# 16NepaliNews Corpus
The 16 Nepali News data set is a collection of approximately 14,364 Nepali language news documents, partitioned (unevenly) across 16 different newsgroup: Auto, Bank, Blog, Business Interview, Economy, Employment, Entertainment, Interview, Literature, National News, Opinion, Sports, Technology, Tourism, and World. 

## Loading the Corpus
```python
MLCOMPDIR = r'LOCATION OF CORPUS'

trainNews = load_mlcomp('16NepaliNews', 'train', mlcomp_root= MLCOMPDIR)
testNews = load_mlcomp('16NepaliNews', 'test', mlcomp_root= MLCOMPDIR)
```
### Or Manually Preparing Training and Test Set
```python
news = load_mlcomp('16NepaliNews', 'raw', mlcomp_root= MLCOMPDIR)

''' Testing and Training Data '''
SPLIT_PERCENT = 0.9

splitSize = int(len(news.data) * SPLIT_PERCENT)
print(splitSize)
xTrain = news.data[:splitSize]
xTest = news.data[splitSize:]
yTrain = news.target[:splitSize]
yTest = news.target[splitSize:]

```

## This Corpus was developed by parsing and scrapping contents published from 2015 of different online news portals. All the news contents ## belong to their respective owners. 

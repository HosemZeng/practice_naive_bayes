# 引用数据集链接
# http://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# 引用代码链接
# https://blog.csdn.net/weixin_44613063/article/details/105896576
# https://blog.csdn.net/mingzhiqing/article/details/82971672
import nltk
#nltk.download() #下载必要文件
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#预处理
def preproccesser(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stops = stopwords.words('english')
    tokens = [token for token in tokens if token not in stops]

    tokens = [token.lower() for token in tokens if len(token)>=3]
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(token) for token in tokens]
    preproccessed_text = ' '.join(tokens)
    return preproccessed_text

#读取数据集
sms_data = open('./SMSSpamCollection','r',encoding='utf-8')
x=[]
y=[]
for line in sms_data.readlines(): #分行
    data = line.split('\t')
    y.append(data[0])
    x.append(preproccesser(data[1]))
sms_data.close()

#拆分
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#特征处理
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words='english', strip_accents='unicode', norm='l2')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test) 

#选择并训练模型
model = MultinomialNB()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print('分数为%.6f' % (score))
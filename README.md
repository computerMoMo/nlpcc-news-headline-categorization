# NLPCC Task2——新闻标题分类

## 如何运行

### Clone

	git clone http://git.oschina.net/eshijia/nlpcc-news-headline-categorization
	cd nlpcc-news-headline-categorization

### 生成pre-train词向量

	python word2vec.py

### 开始训练

	python train.py

### Baseline结果
NBOW: 0.783

#### Word Level

Results Summary
********************
Model Name: S_CNN
Best Accuracy: 0.7805
********************
********************
Model Name: M_CNN
Best Accuracy: 0.780361111111
********************
********************
Model Name: U_LSTM
Best Accuracy: 0.784083333333
********************
********************
Model Name: B_LSTM
Best Accuracy: 0.779555555556
********************
********************
Model Name: CNN_LSTM
Best Accuracy: 0.78075
********************

#### Character Level

Results Summary
********************
Model Name: S_CNN
Best Accuracy: 0.775361111111
********************
********************
Model Name: M_CNN
Best Accuracy: 0.775333333333
********************
********************
Model Name: U_LSTM
Best Accuracy: 0.759194444444
********************
********************
Model Name: B_LSTM
Best Accuracy: 0.757305555556
********************
********************
Model Name: CNN_LSTM
Best Accuracy: 0.789611111111
********************

#### Mixed Model
- SGD：0.7884
- Adam:0.797
- Adadelta:0.8028
- RMSprop: 0.8046
- Adagrad: 0.8131
- Adamax: 0.8019
- Nadam: 0.7892



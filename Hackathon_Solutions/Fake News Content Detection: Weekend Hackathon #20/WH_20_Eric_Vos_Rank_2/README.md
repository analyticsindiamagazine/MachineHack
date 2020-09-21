# MH20_Rank_2

## Problem Overview
Welcome to another weekend hackathon, this weekend we are providing a great opportunity to the machinehackers to flex their NLP muscles again by building a fake content detection algorithm. Fake contents are everywhere from social media platforms, news platforms and there is a big list. Considering the advancement in NLP research institutes are putting a lot of sweat, blood, and tears to detect the fake content generated across the platforms.

Fake news, defined by the New York Times as “a made-up story with an intention to deceive”, often for a secondary gain, is arguably one of the most serious challenges facing the news industry today. In a December Pew Research poll, 64% of US adults said that “made-up news” has caused a “great deal of confusion” about the facts of current events

In this hackathon, your goal as a data scientist is to create an NLP model, to combat fake content problems. We believe that these AI technologies hold promise for significantly automating parts of the procedure human fact-checkers use today to determine if a story is real or a hoax.

 

Dataset Description:

Train.csv - 10240 rows x 3 columns (Inlcudes Labels Columns as Target)
Test.csv - 1267 rows x 2 columns
Sample Submission.csv - Please check the Evaluation section for more details on how to generate a valid submission
 

Attribute Description:

Text - Raw content from social media/ new platforms
Text_Tag - Different types of content tags
Labels - Represents various classes of Labels
Half-True - 2
False - 1
Mostly-True - 3
True - 5
Barely-True - 0
Not-Known - 4
Skills:

NLP, Sentiment Analysis
Feature extraction from raw text using TF-IDF, CountVectorizer
Using Word Embedding to represent words as vectors
Using Pretrained models like Transformers, BERT
Optimizing multi-class log loss to generalize well on unseen data



# Solution Overview :

[1]mh20-basline-v1 : Initial baseline using autoviml

[2]mh20-fastai-v1 : Fastai with Transformers (RoBERTa) Version 1²

[3]Blend_v1f : Layer 1 blender (39%[1]-61%[2])

[4]mh20-fastai-v2 : Fastai with Transformers (RoBERTa) Version 2²

[5]Blend_v4d : Layer 2 blender (65%[3]-35%[4])


² : Deeply inspired by https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2

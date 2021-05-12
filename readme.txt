Code of Rethinking Label-Wise Cross-Modal Retrieval from A Sematic Sharing Perspective.
Enjoy the code.

**************************** Requirement ****************************
#requirement python 3.6,pytorch 1.3,cuda 10.0,cudnn 7.6

******************************* USAGE *******************************
data_m.py ----- The code of dataloader(Flickr,COCO,NUS-Wide).
evalution.py  ----- The code of evalution 
tools.py  ----- The code of save models.
Model.py  ----- The code of model
main.py  ----- The main code of the algorithm. It takes training data/label and parameters as input.

--the data file should contain:
#'train_text': train text data
#'dev_text': dev text data
#'test_text': test text data
#'train_label': train data label
#'dev_label': dev data label
#'test_label': test data label
#'train_preim': train image data
#'devl_preim': dev image data
#'test_preim': test image data

--the parameters
In the main
#'lr': the learning rate
#'img_dim': the embedding size of image
#'embed_size': the embedding size of text
#'data_name': the prepare dataset name
#'bi_gru': the parameter of use bidirectional GRU

--demo:
data/: dataset of Flickr, there are 2 modalities including image modality and text modality
i.e.
python main.py

***************************** REFERENCE *****************************
If you use this code in scientific work, please cite:
Yang Yang, Chu-Bing Zhang, Dian-Hai Yu, Jian Yang. Rethinking Label-Wise Cross-Modal Retrieval from A Sematic Sharing Perspective. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI'21).
*********************************************************************

# Microsoft Azure Notebooks UMass Amherst
This repository contains end-to-end machine learning notebooks made by students in the 696DS course that showcase different functionalities with Microsoft Azure Machine Learning Services. 

### [`commercial-blocks-classification`](https://github.com/katiehouse3/microsoft-azure-ml-notebooks/blob/master/commercial-blocks-classification/classifying-commercial-blocks.ipynb)
* **Task**: Classifying commericial blocks on telvision News Channels
* **Methods**: Comparing standard Scikit Learn models with Azure Automated Machine Learning 
* **Azure Functionalities**: Running an Experiment, Automated Machine Learning, and Logging Metrics
* **Results**: Increased testing accuracy by 3.25%!

### [`CIFAR-notebook`](https://github.com/katiehouse3/microsoft-azure-ml-notebooks/blob/master/CIFAR-notebook/Azure-Cifar10_FcNet.ipynb)
* **Task**: Image Classification on the CIFAR 10 dataset
* **Methods**: Fully connected neural network model
* **Azure Functionalities**: Hyperdrive run, Metric logging
* **Results**: Increased model classification accuracy by ~16%!

### [`telecom-churn`](https://github.com/katiehouse3/microsoft-azure-ml-notebooks/blob/master/telecom-churn/Telecom%20Churn.ipynb)
* **Task**: Classifying customers for customer retention problem. 
* **Methods**: Model with highest accuracy- RandomForest
* **Azure Functionalities**: Model explanation
* **Results**: Understood the features responsible for someone's churn. 

### [`machine-translation`](https://github.com/katiehouse3/microsoft-azure-ml-notebooks/blob/master/machine-translation/train_wrapper.ipynb)
* **Task**: Translating sentences from German to English
* **Methods**: Using pre-trained BERT representations in Transformer Model
* **Azure Functionalities**: Register DataStore, AML Compute, Submitting and Cancelling Experiment Runs
* **Results**: 26.3
* _Note_: Implementation closely follows this [tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

### [`mashable-news-popularity`](https://github.com/katiehouse3/microsoft-azure-ml-notebooks/blob/master/mashable-news-popularity/mashable_news_popularity.ipynb)
* **Task**: Predicting popularity of online news articles based on number of shares
* **Methods**: Classification
* **Azure Functionalities**: Auto Machine Learning (AML), AutoMLExplainer, Register Model
* **Results**: Iterated over 10 models using AML module. Achieved 67% accuracy over 5 different classes of labels (much higher than random!).

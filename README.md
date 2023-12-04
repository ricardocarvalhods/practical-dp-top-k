## Differentially Private Top-k Selection

The script in this folder is part of the supplementary material for the following paper:
- **Title**: A Framework for Practical Differentially Private Top-𝑘 Selection
- **Authors**: [Ricardo Carvalho](https://ricardocarvalho.ca), [Ke Wang](http://www.cs.sfu.ca/~wangk/), [Lovedeep Gondara](https://lovedeepgondara.com/)

---

### Goal

- Our goal is to perform differentially private top-k selection of elements in a dataset where records contain users' contributions to elements.
- To allow selection in practice on real-world systems with massive datasets, the mechanisms evaluated select top-k from the restricted domain of top-<img src="https://render.githubusercontent.com/render/math?math=\bar{k}"> elements, for a given <img src="https://render.githubusercontent.com/render/math?math=\bar{k} \geq k">.
- For the datasets used below, elements will be locations/venues. In this context, each user may contribute a value of 1 if the location was ever visited (irrespective of how many times) and 0 if the user never visited the location.
- Therefore, we aim to select the top-k most visited locations without compromising the privacy of users.

---

### Datasets

Datasets need to be downloaded to run the experiments, as follows:

- D1: Yelp
  - Our source is the file **yelp_academic_dataset_tip.json**, available at [this link](https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_tip.json).
  - To run experiments below on D1, simply download the json file to the **datasets** folder.
- D2: Foursquare
  - Our sources are the files **dataset_TSMC2014_NYC.txt** and **dataset_TSMC2014_TKY.txt** inside the folder **dataset_tsmc2014** obtained from the zip file available at [this link](http://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip).
  - To run experiments below on D2, download the zip file, unzip it and copy the two txt files mentioned above to the **datasets** folder.
- D3: Gowalla
  - Our source is the file **loc-gowalla_totalCheckins.txt**, available at [this link](https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz).
  - To run experiments below on D3, download the zip file and unzip it to the **datasets** folder.

---

### Mechanisms

- We compare two methods:
  - **RestrictedGumbel (RG)**: novel mechanism from the framework proposed in our paper.
  - **LimitDomain (LD)**: algorithm from [Durfee and Rogers, '19](https://arxiv.org/pdf/1905.04273.pdf).
- These two mechanisms consider a **user may contribute to an <u>unknown</u> number of elements**.
- We experiment various settings of privacy budget, such that each individual execution of a mechanism has overall privacy guarantee as (<img src="https://render.githubusercontent.com/render/math?math=\varepsilon_{total},\,\delta_{total}">)-DP.

---

### Metrics

- To compare the mechanisms we use the following metrics:
    - *Number of elements returned*, i.e. the output size of each mechanism. Since restricted domain mechanisms are not guaranteed to output k elements, this metric is an efficient measurement of the utility of such mechanisms.
    - *SR*: Score Ratio, defined as the ratio between the "sum of the scores of the elements returned" and the "sum of the scores of the true top-k elements". *SR* gives an idea of the quality of the scores by comparing the scores of the elements each mechanism outputs to the best possible, which are true top-k scores.

---

### How to run experiments

- Unfortunately, this anonymized repository does not allow download of files. 
- So to be able to run the experiments, you have to copy the contents of the files listed.
- After that, simply download the datasets and run the following commands:
   - `pip3 install numpy pandas matplotlib`
   - `python3 run_experiments.py`


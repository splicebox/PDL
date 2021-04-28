# DL
This is a probabilistic deep learning model for pridicting the splicing ratios with the inputs of sequence elements (optionally splicing factor in the form of position weight matrix) and training inputs of splice junction read counts

```
Copyright (C) 2021, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea
```
### <a name="model-architecture " /> Model architecture
![alt text](https://github.com/splicebox/DL/blob/main/figures/PDL_architecture.png)


### <a name="dnn-models " /> DNN models
We build two models for training and evaluation, which largely share the same architecture, but with a key difference. For the first model, herein named DNN1, we initialize the filters of the RBP PWMs CNN Layer by RBP PWMs downloaded from the ATtRACT database. A quality value of 0.1 was used to filter the PWMs, and 1041 PWMs for 151 human RBPs were selected. For the second model, named DNN2, we randomly initialized 1024 (4 x 1024) filters for the RBP PWMs CNN Layer.

### <a name="support" /> Support
Contact: gyang22@jhu.edu

### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

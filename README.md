# DL
This is a computational model aimed at revealing the splicing regulation. More specifically, we describe a probabilistic deep learning model to predict and quantify alternative splicing events from the information on cis-regulatory sequence elements and trans-splicing factors in different tissues.

```
Copyright (C) 2021, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea
```
### <a name="model-architecture " /> Model architecture
![alt text](https://github.com/splicebox/DL/blob/main/figures/PDL_architecture.png)


### <a name="input-and-output" /> Input and Output
To allow investigating how the sequence elements and associated trans-splicing factors affect the selection of introns, the model implements three types of inputs. The first input is the RNA sequence. The RNA sequence around the splice site contains the exonic splicing enhancers (ESEs), exonic splicing silencers (ESSs), intronic splicing enhancers (ISEs), and  intronic splicing silencers (ISSs), which are the binding motifs for RBPs. The second type of input consists of the splice junction read counts to represent the `expression' level of an intron. The third type of input is the RBP RNA recognition motif (RRM), the RNA binding domain of the RBP.

![alt text](https://github.com/splicebox/DL/blob/main/figures/input_output.png)

### <a name="dnn-models" /> DNN models
We build two models for training and evaluation, which largely share the same architecture, but with a key difference. For the first model, herein named DNN1, we initialize the filters of the RBP PWMs CNN Layer by RBP PWMs downloaded from the ATtRACT database. A quality value of 0.1 was used to filter the PWMs, and 1041 PWMs for 151 human RBPs were selected. For the second model, named DNN2, we randomly initialized 1024 (4 x 1024) filters for the RBP PWMs CNN Layer.

### <a name="support" /> Support
Contact: gyang22@jhu.edu

### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

# Hiformer: Hybrid Frequency Feature Enhancement Inverted Transformer for Long-Term Wind Power Prediction
# Introduction
Unlike traditional power station, the output of wind power plants is highly dependent on weather parameters such as wind speed, wind direction and temperature. This dependence introduces significant volatility and intermittency in wind power production, making grid systems highly fragile and challenging to manage
<p align="center">
  <img src="figure/sample.png" width="50%">
</p>

# Overall Architecture
Hiformer introduces a unique structure that integrates signal decomposition technology with weather feature extraction technique to enhance the modeling of correlations between meteorological conditions and wind power generation. Additionally, Hiformer employs an encoder-only architecture, which reduces the computational complexity associated with long-term wind power forecasting.
<p align="center">
  <img src="figure/model.png" width="70%">
</p>

# Usage
1.Install Pytorch and necessary dependencies
```bash
pip install -r requirements.txt
```
2.The datasets can be obtained from SDWPF and GEFcom2014
# Results of Long-Term Wind Power Prediction
## Performance comparison of different models for wind power prediction on SDWPF and GEFcom datasets.
We evaluate the Hiformer on two real-world datasets. Achieving best performance (MAE/MSE)
![image](figure/table.png)
## Normalized forecasting results from Aug-7 to Aug-9 across the entire wind farm by all methods for (a) SDWPF dataset and (b) GEFcom dataset
![image](figure/result.png)
## Computational time
By desiging the proposed framework, Hiformer achieves significant performance improvement in computational time
<p align="center">
  <img src="figure/Computational time.png" width="50%">
</p>

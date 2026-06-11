# References

The bibliography for the PM2.5 GNN-TL project. Organized by topic. All citations follow the standard et-al author/year convention.

---

## A. Transfer learning for environmental / cross-city time series

Sanjeev, C. T., Prakash, B. B., & Maitra, B. (2025). *Transfer Learning Framework for PM2.5 Forecasting in Indian Cities.* B.Tech thesis, Indian Institute of Information Technology Sricity. Supervised by Dr. Mainak Thakur. Submitted 05 January 2025.

Khan, S. N., Li, D., & Maimaitijiang, M. (2024). Using gross primary production data and deep transfer learning for crop yield prediction in the US Corn Belt. *International Journal of Applied Earth Observation and Geoinformation, 131*, 103965.

Sangiorgio, M., & Guariso, G. (2024). Transfer learning in environmental data-driven models: A study of ozone forecast in the Alpine region. *Environmental Modelling & Software, 177*, 106048.

Ye, R., & Dai, Q. (2018). A novel transfer learning framework for time series forecasting. *Knowledge-Based Systems, 156*, 74–99.

---

## B. Spatio-Temporal GNN backbones

Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. In *Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18)*, pp. 3634–3640. (STGCN.)

Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. In *International Conference on Learning Representations (ICLR)*. (DCRNN.)

Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 1907–1913.

Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD-20)*. (MTGNN.)

Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. In *Advances in Neural Information Processing Systems 33 (NeurIPS-20)*. (AGCRN.)

Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019). Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence, 33(01)*, 922–929. (ASTGCN.)

Zheng, C., Fan, X., Wang, C., & Qi, J. (2020). GMAN: A Graph Multi-Attention Network for Traffic Prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence, 34(01)*, 1234–1241.

---

## C. Air-quality forecasting (graph-based and GNN)

Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network for PM2.5 Forecasting. In *Proceedings of the 28th International Conference on Advances in Geographic Information Systems (SIGSPATIAL '20)*. (PM2.5-GNN.)

Chen, L., Xu, J., Wu, B., Qian, Y., Du, Z., Li, Y., & Zhang, Y. (2021). Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting. arXiv:2108.12238. (GAGNN.)

Shao, X., Zhang, C., Yan, T., & Li, Z. (2021). HighAir: A Hierarchical Graph Neural Network-Based Air Quality Forecasting Method. arXiv:2101.04264.

Liang, Y., Xia, Y., Ke, S., Wang, Y., Wen, Q., Zhang, J., Zheng, Y., & Zimmermann, R. (2023). AirFormer: Predicting Nationwide Air Quality in China with Transformers. In *Proceedings of the AAAI Conference on Artificial Intelligence, 37(12)*, 14329–14337.

Han, J., Liu, H., Zhu, H., Xiong, H., & Dou, D. (2021). Joint Air Quality and Weather Prediction Based on Multi-Adversarial Spatiotemporal Networks. In *Proceedings of the AAAI Conference on Artificial Intelligence, 35(5)*, 4061–4069.

Zheng, Y., Yi, X., Li, M., Li, R., Shan, Z., Chang, E., & Li, T. (2015). Forecasting Fine-Grained Air Quality Based on Big Data. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-15)*.

Han, J., Yang, S., Wang, Q., Zhou, J., & Lin, Y. (2021). A New Benchmark of Graph Learning for PM2.5 Forecasting under Distribution Shift. In *NeurIPS 2021 Workshop on Graph Learning Benchmarks (GLB)*. (PM2.5-GLB.)

Bedi, S., Katiyar, A., Krishnan, N. A., & Kota, S. H. (2024). Utilizing LSTM models to predict PM2.5 levels during critical episodes in Delhi. *Urban Climate, 53*, 101835.

---

## D. Inductive GNN and pre-training

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. In *Advances in Neural Information Processing Systems 30 (NeurIPS-17)*. (GraphSAGE.)

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. In *International Conference on Learning Representations (ICLR)*. (GAT.)

Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? In *International Conference on Learning Representations (ICLR)*. (GATv2 — diagnoses a static-attention limitation of the original GAT and proposes a swap-then-LeakyReLU fix; relevant motivation for the single-head edge-weighted variant used here.)

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? In *International Conference on Learning Representations (ICLR)*. (GIN; basis for the mean⊕max graph-readout in §2.6.)

Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In *International Conference on Learning Representations (ICLR)*.

Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for Pre-training Graph Neural Networks. In *International Conference on Learning Representations (ICLR)*.

Rong, Y., Huang, W., Xu, T., & Huang, J. (2020). DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. In *International Conference on Learning Representations (ICLR)*. (Edge-dropping regularizer; relevant to oversmoothing prevention — not adopted here because the GNN is only 2 layers deep, but cited as a deeper-architecture remedy.)

You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). Graph Contrastive Learning with Augmentations. In *Advances in Neural Information Processing Systems 33 (NeurIPS-20)*. (GraphCL; complementary self-supervised GNN pre-training approach.)

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271. (TCN; basis for the dilated 1-D causal conv used in §2.4.)

---

## E. GNN domain adaptation / transfer

Wu, M., Pan, S., Zhou, C., Chang, X., & Zhu, X. (2020). Unsupervised Domain Adaptive Graph Convolutional Networks. In *Proceedings of The Web Conference 2020 (WWW '20)*. (UDA-GCN.)

Zhu, Q., Yang, C., Xu, Y., Wang, H., Zhang, C., & Han, J. (2021). Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization. In *Advances in Neural Information Processing Systems 34 (NeurIPS-21)*. (EGI.)

Shi, B., Wang, Y., Guo, F., Shao, J., Shen, H., & Cheng, X. (2024). A Survey on Graph Domain Adaptation. arXiv:2402.00904.

Ruiz, L., Chamon, L. F. O., & Ribeiro, A. (2020). Graphon Neural Networks and the Transferability of Graph Neural Networks. In *Advances in Neural Information Processing Systems 33 (NeurIPS-20)*.

Levie, R., Huang, W., Bucci, L., Bronstein, M., & Kutyniok, G. (2021). Transferability of Spectral Graph Convolutional Neural Networks. *Journal of Machine Learning Research, 22(272)*, 1–59.

---

## F. Cross-city spatio-temporal transfer

Wei, Y., Zheng, Y., & Yang, Q. (2016). Transfer Knowledge between Cities. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-16)*.

Wang, L., Geng, X., Ma, X., Liu, F., & Yang, Q. (2018). Cross-City Transfer Learning for Deep Spatio-Temporal Prediction. arXiv:1802.00386. (RegionTrans precursor.)

Wang, L., Geng, X., Ma, X., Liu, F., & Yang, Q. (2019). Cross-City Transfer Learning for Deep Spatio-Temporal Prediction. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)*.

Yao, H., Liu, Y., Wei, Y., Tang, X., & Li, Z. (2019). Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction. In *Proceedings of The Web Conference 2019 (WWW '19)*. (MetaST.)

Lu, B., Gan, X., Zhang, W., Yao, H., Fu, L., & Wang, X. (2022). Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD-22)*. (ST-GFSL.)

Jin, Y., Chen, K., & Yang, Q. (2022). Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD-22)*. (CrossTReS.)

Jin, Y., Chen, K., & Yang, Q. (2023). Transferable Graph Structure Learning for Graph-Based Traffic Forecasting Across Cities. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD-23)*. (TransGTR.)

Tang, Y., Qu, A., Chow, A. H. F., Lam, W. H. K., Wong, S. C., & Ma, W. (2022). Domain Adversarial Spatial-Temporal Network: A Transferable Framework for Short-term Traffic Forecasting across Cities. In *Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM-22)*. (DASTNet.)

---

## G. Temporal distribution adaptation

Du, Y., Wang, J., Feng, W., Pan, S., Qin, T., Xu, R., & Wang, C. (2021). AdaRNN: Adaptive Learning and Forecasting for Time Series. In *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM-21)*.

---

## H. Graph normalization and training stability

Cai, T., Luo, S., Xu, K., He, D., Liu, T.-Y., & Wang, L. (2021). GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training. In *Proceedings of the 38th International Conference on Machine Learning (ICML-21)*.

Zhao, L., & Akoglu, L. (2020). PairNorm: Tackling Oversmoothing in GNNs. In *International Conference on Learning Representations (ICLR)*.

Lin, L., Chen, J., & Wang, H. (2024). Unleash Graph Neural Networks from Heavy Tuning. arXiv:2405.12521.

---

## I. Domain-adversarial foundations

Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. In *Proceedings of the 32nd International Conference on Machine Learning (ICML-15)*. (DANN; gradient-reversal layer that this project's Variant B is built on.)

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks. *Journal of Machine Learning Research, 17(59)*, 1–35. (Foundational domain-adaptation reference; cited as related work.)

Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). Adversarial Discriminative Domain Adaptation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR-17)*. (ADDA; adversarial domain adaptation, cited as related work.)

de Mathelin, A., Atiq, M., Richard, G., de la Concha, A., Yachouti, M., Deheeger, F., Mougeot, M., & Vayatis, N. (2020). Adversarial Weighting for Domain Adaptation in Regression. arXiv:2006.08251. (Diagnoses why naïve adversarial domain adaptation underperforms on regression; cited as related work.)

Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML-17)*. (MAML.)

Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. arXiv:1803.02999. (Reptile.)

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? In *Advances in Neural Information Processing Systems 27 (NeurIPS-14)*. (Empirical foundation for the freeze-low-layers / fine-tune-high-layers TL recipe used in both Variant A and Variant B's fine-tune phase.)

---

## J. Surveys

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). A Comprehensive Survey on Graph Neural Networks. *IEEE Transactions on Neural Networks and Learning Systems, 32(1)*, 4–24.

Jin, M., Koh, H. Y., Wen, Q., Zambon, D., Alippi, C., Webb, G. I., King, I., & Pan, S. (2023). A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection. arXiv:2307.03759.

Sahili, Z. A., & Awad, M. (2023). Spatio-Temporal Graph Neural Networks: A Survey. arXiv:2301.10569.

Wang, S., Cao, J., & Yu, P. S. (2022). Deep Learning for Spatio-Temporal Data Mining: A Survey. *IEEE Transactions on Knowledge and Data Engineering, 34(8)*, 3681–3700.

---

## K. Air quality policy and Indian context

Centre for Science and Environment (CSE). (2024). *2023 – the crossroad: Yearend analysis of PM2.5 pollution in Delhi.* CSE Report, January 2024. Retrieved from https://www.cseindia.org/Report-Jan3-4-End-of-2023-state-of-air-pollution.pdf

IQAir. (2024). *2023 World Air Quality Report.* IQAir AirVisual, Goldach, Switzerland.

Central Pollution Control Board (CPCB), Government of India. *Continuous Ambient Air Quality Monitoring (CAAQM)* — station-level air quality data portal. https://cpcb.nic.in/

Mor, S., Singh, N., Verma, M., Pankaj, A., Singh, P., & Ravindra, K. (2024). PM2.5 trends across cities of India: a comprehensive review of monitoring, modeling, and mitigation strategies. *Environment, Development and Sustainability*. Springer.

Awasthi, A., Pandey, S. K., & Verma, V. (2023). Air pollution monitoring stations in India: density, distribution, and urban–rural disparities. *Atmospheric Environment, 314*, 120103.

---

## L. PM2.5 forecasting with distribution shift

Han, J., Yang, S., Wang, Q., Zhou, J., & Lin, Y. (2023). PM2.5 forecasting under distribution shift: A graph learning approach. *Machine Learning with Applications, 12*, 100467. (Same group as PM2.5-GLB benchmark.)

Mahmoodvand, S., Aliyari, M., & Hashemi, S. (2024). Enhancing PM2.5 prediction by mitigating annual data drift using wrapped loss and neural networks. *PLOS ONE, 19(4)*, e0314327.

Pant, M., Sharma, R., & Bansal, A. (2025). Temporally boosting neural network for improving dynamic prediction of PM2.5 concentration with changing and unbalanced distribution. *Journal of Cleaner Production*, advance online publication.

Verma, S., Roy, A., & Khan, M. (2024). An adaptation Koopman model for predicting PM2.5 with distribution drift. *Atmospheric Environment, 319*, 120355.

---

## M. Deep TL applied to PM2.5 in India

Yadav, P., Kumar, A., Sharma, V., & Verma, A. (2024). Deep transfer learning and attention based PM2.5 forecasting in Delhi using a decade of winter season data. *Environmental Modelling & Software, 175*, 105987. (Multi-year, frozen-layer fine-tune within Delhi; cited as evidence for the climatology-residual / year-transfer approach.)

---

## N. Statistical testing

Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics, 13(3)*, 253–263. (DM test, used for the eventual paper's pairwise method comparison.)

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting, 13(2)*, 281–291. (Small-sample correction to the DM test.)

---

## P. Time-series cross-validation and data-leakage prevention

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40(8)*, 913–929. (Authoritative treatment of how to design CV folds that respect temporal/spatial autocorrelation; motivates the chronological-block control split discussed in [AUDIT.md](AUDIT.md) §3.2.)

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. (Foundational analysis of why naïve k-fold CV inflates error estimates on autocorrelated time series, and when it is — and isn't — defensible.)

Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis, 120*, 70–83. (Shows that for stationary AR processes, k-fold CV is asymptotically valid — relevant defence for the interleaved-split protocol used in `--fixed` mode, with caveats discussed in [AUDIT.md](AUDIT.md) §3.2.)

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data, 6(4)*. (Taxonomy of leakage modes — used in [AUDIT.md](AUDIT.md) to categorize each potential leakage source identified in this codebase.)

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning, 109*, 1997–2028. (Empirical comparison of holdout, k-fold, blocked-k-fold, and rolling-origin CV for time-series forecasting — informs the recommendation block in [AUDIT.md](AUDIT.md) §6.)

---

## O. WHO / global air-quality standards

World Health Organization. (2021). *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.* Geneva: World Health Organization. https://www.who.int/publications/i/item/9789240034228

---

*End of References. Cross-references in [reports/MAIN_REPORT.md](MAIN_REPORT.md), [reports/MATH_AUDIT.md](MATH_AUDIT.md), [reports/PINN_PHYSICS.md](PINN_PHYSICS.md), and [reports/SENSOR_EFFICIENCY_PINN_LITERATURE.md](SENSOR_EFFICIENCY_PINN_LITERATURE.md) draw from this bibliography.*

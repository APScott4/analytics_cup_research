**Introduction**  
This project explores the use of graph neural networks (GNNs) to predict the outcome of play-out-from-the-back attempts in football. Building on the methodology of A Graph Neural Network deep-dive into successful counterattacks , Sahasrabudhe and Bekkers, we adapt their graph-based approach to model phases where a team builds play from deep defensive areas.

The model is trained on 33,263 frames from nine A-League matches, with one match held out for testing. Tracking and event data are processed to isolate relevant play-out phases, which are represented as dynamic player graphs capturing spatial and contextual information. Current work is exploratory, focusing on refining phase definitions, expanding feature sets, and tuning model parameters to improve predictive performance.

We also investigate feature importance using a dataset-shuffling approach to identify which spatial and contextual factors most influence successful outcomes. Preliminary results demonstrate the potential of GNNs to model complex, multi-agent interactions in structured football phases, providing a foundation for future tactical and performance analyses.

**Methods**  
Play-out-from-the-back attempts are identified by following a team’s possession from deep defensive phases, treating build-up as a continuous sequence. Only subsequent phases involving the same team are included to capture the full progression. Success is defined as the ball reaching the middle or attacking third or leading to a shot/goal; failure is defined as lost possession or chaotic/direct play. Phases are converted to frame-level labels indicating active attempts and outcomes.

Using the unravelsports package, tracking data are converted into graph datasets. Node features include positions, velocity, distances to goal and ball, attacking/GK/ball flags, and angles to goal and ball. Edge features include player distance, speed difference, and positional/velocity angles.

#### **Results**

| Loss | AUC | Binary Accuracy |
| :---- | :---- | :---- |
| 0.2818 | 0.9501 | 0.8723 |

These results suggest strong overall performance at first glance. However, further analysis indicates that these metrics may overestimate true predictive ability, as the model still struggles with accurately classifying certain play-out-from-the-back scenarios. This highlights that while the model captures general patterns, it requires additional refinement to reliably predict outcomes across all situations.

#### **Conclusion**

This research demonstrates the potential of graph-based modelling approaches to capture and evaluate complex tactical sequences in football, such as playing out from the back. While the current model remains exploratory, the results highlight how structured representations of player interactions can offer new perspectives for analysing both offensive build-up and defensive pressure. With additional data, refined phase definitions, and further tuning of model parameters, this approach has the capacity to produce more reliable and practically useful insights for coaches and analysts.

Future work will focus on improving the identification and labelling of play-out-from-the-back sequences and extending the modelling framework to more advanced architectures, such as attention-based transformers. Incorporating longer temporal context through these methods would allow the model to better learn patterns across sequences of play, further enhancing its ability to evaluate decision-making and tactical effectiveness in build-up situations.  

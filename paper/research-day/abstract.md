#	Problembeschreibung
Reinforcement Learning has gathered more and more attention in recent years. This is mostly due to the fact, that they are currently able to beat World Champion Human Players at board games like Chess and Go, and also videogames like Dota 2. These algorithms solve problems dot not solve problems using a predefined set of instructions like in traditional programming, but instead learn by interacting with an environment, much like humans do. With reinforcement learning algorithms becoming more efficient and computer hardware becoming increasingly powerful new challenges are needed to test their boundaries. To that effect a group of Blizzard and Deepmind developers have recently released the StarCraft II Machine Learning and PySC2 APIs. They allow StarCraft II, often described as one of the most complex video games of all time, to be accessible by reinforcement learning algorithms. PySC2 in particular is specifically designed for usage with rl algorithms. The challenge is to integrate this new technology with standard rl algorithms and overcoming difficulties like StarCraft II's vast action- and observation spaces.

#	Zielstellung
The accompanying project to this article aims to explore StarCraft II as an evironment and benchmark for reinforcement learning algorithms. Both the StarCraft II environment itself is evaluated for it's suitability and also a number of specific algorithms. For interfacing with the StarCraft II client the PySC2 API will be used. Additionally, the project provides both abstract scenarios for training purposes and scenarios modelled after scenarios commonly found in the competitive 1 vs 1 mode of the game. These are custom built with the StarCraft II Galaxy editor. Using these Scenarios models are trained that achieve results comparable to human players and, to the best of our knowledge for the first time, even manage to score higher rewards than expert players in a competitive scenario. Lastly, this project provides a framework that makes it easier for future users of the PySC2 API to interface standard reinforcement learning algorithms with StarCraft II.

#	Lösungsansätze und methodische Herangehensweise
The StarCraft II Environment is explored using 3 different algorithms. A custom implementation of the A3C algorithm, and also two of the OpenAI Baselines algorithms, ACKTR and A2C, which are adapted and integrated into this project. As these algorithms can not interface with PySC2 or StarCraft II directly, a set of wrappers is developed, one for each scenario, which connect them. These wrappers also allow fine tuning of some of the parameters of the difficulty and can restrict access to only a fraction of the observation and action spaces. Additionally they are designed in a way that they can easily be adapted to any number of different scenarios. Lastly, the algorithms are augmented beyond their base implementation, in an attempt to improve training results. Among these augmentations are using a state history, which includes both current and past observations in the current state representation, and an attempt to implement a hybrid training strategy, that mixes in prerecorded human experiences into normal training.
#	evtl. verwendete Datengrundlage/Literatur

Denny Britz (2017). Hype or Not? Some Perspective on OpenAI’s DotA 2 Bot. http:
//www.wildml.com/2017/08/hype-or-not-some-perspective-onopenais-
dota-2-bot/. Online; accessed 29 January 2014.
OpenAI (2017). Dota 2. https://blog.openai.com/dota-2/. Online; accessed
29 January 2014.
YuhuaiWu and Elman Mansimov and Shun Liao and Alec Radford and John Schulman
(2015). OpenAI Baselines A2c and ACKTR Blog. https://blog.openai.
com/baselines-acktr-a2c/. Online; accessed 29 January 2014.
Adam Heinermann (2015). C++ API for Starcraft Broodwar. http://bwapi.github.
io/. Online; accessed 29 January 2014.
Babaeizadeh, Mohammad et al. (2016). “GA3C: GPU-based A3C for Deep Reinforcement
Learning”. In: CoRR abs/1611.06256. URL: http://arxiv.org/abs/
1611.06256.
Bellman, Richard (1957). “A Markovian Decision Process”. In: Indiana Univ. Math. J.
6 (4), pp. 679–684. ISSN: 0022-2518.
Blizzard Corporation (2015a). C++ API for Starcraft II. https://github.com/
Blizzard/s2client-proto. Online; accessed 29 January 2014.
– (2015b). pysc2 Python API for Starcraft II. https://github.com/deepmind/
pysc2. Online; accessed 29 January 2014.
– (2015c). Starcraft II. http://eu.battle.net/sc2/en/. Online; accessed 29
January 2014.
Cerda, Carlos B. Ramirez and Armando J. Espinosa de los Monteros F. (1997). “Evaluation
of a (R, s, Q, c) Multi-Item Inventory Replenishment Policy Through Simulation”.
In: Proceedings of the 29th conference onWinter simulation, WSC 1997, Atlanta,
GA, USA, December 7-10, 1997. Ed. by Sigrún Andradóttir et al. ACM, pp. 825–831.
DOI: 10.1109/WSC.1997.640959. URL: http://doi.ieeecomputersociety.
org/10.1109/WSC.1997.640959.
Churchill, D., A. Saffidine, and M. Buro (2012). “Fast Heuristic Search for RTS Game
Combat Scenarios”. In: AAAI AIIDE. URL: http : / / musicweb . ucsd . edu /
~sdubnov/Mu270d/AIIDE12/01/AIIDE12-027.pdf.
Churchill, David et al. (2015). “StarCraft Bots and Competitions”. In: URL: http://
www.cs.mun.ca/~dchurchill/pdf/ecgg15_chapter-competitions.
pdf.
David Silver (2015). UCL Course on RL. http://www0.cs.ucl.ac.uk/staff/
d.silver/web/Teaching.html.
Finn, Chelsea et al. (2016). “Generalizing Skills with Semi-Supervised Reinforcement
Learning”. In: CoRR abs/1612.00429. arXiv: 1612.00429. URL: http://arxiv.
org/abs/1612.00429.
Gers, Felix A., Jürgen Schmidhuber, and Fred A. Cummins (2000). “Learning to Forget:
Continual Prediction with LSTM”. In: Neural Computation 12.10, pp. 2451–
90 BIBLIOGRAPHY
2471. DOI: 10.1162/089976600300015015. URL: https://doi.org/10.
1162/089976600300015015.
Glynn, Peter W. (1990). “Likelihood Ratio Gradient Estimation for Stochastic Systems”.
In: Commun. ACM 33.10, pp. 75–84. DOI: 10.1145/84537.84552. URL:
http://doi.acm.org/10.1145/84537.84552.
Gullapalli, Vijaykumar (1990). “A stochastic reinforcement learning algorithm for
learning real-valued functions”. In: Neural Networks 3.6, pp. 671–692. DOI: 10 .
1016/0893-6080(90)90056-Q. URL: https://doi.org/10.1016/0893-
6080(90)90056-Q.
Hasselt, Hado van, Arthur Guez, and David Silver (2016). “Deep Reinforcement
Learning with Double Q-Learning”. In: pp. 2094–2100. URL: http://www.aaai.
org/ocs/index.php/AAAI/AAAI16/paper/view/12389.
He, Kaiming et al. (2015). “Deep Residual Learning for Image Recognition”. In: CoRR
abs/1512.03385. arXiv: 1512.03385. URL: http://arxiv.org/abs/1512.
03385.
Hesterberg, Tim (2004). “Introduction to Stochastic Search and Optimization: Estimation,
Simulation, and Control”. In: Technometrics 46.3, pp. 368–369. DOI: 10.
1198/tech.2004.s206. URL: https://doi.org/10.1198/tech.2004.
s206.
Hochreiter, Sepp and Jürgen Schmidhuber (1997). “Long Short-term Memory”. In: 9,
pp. 1735–80.
Jaromír Jaara (2017). A3C implementation for the Cartpole Environment. https : / /
github.com/jaara/AI-blog/blob/master/CartPole-A3C.py. "Github
repository of Jaromír Jaara ".
Justesen, Niels and Sebastian Risi (2017). “Continual Online Evolutionary Planning
for In-Game Build Order Adaptation in StarCraft”. In: URL: http://sebastianrisi.
com/wp-content/uploads/justesen_gecco17.pdf.
Kaelbling, Leslie Pack, Michael L. Littman, and Anthony R. Cassandra (1998). “Planning
and Acting in Partially Observable Stochastic Domains”. In: Artif. Intell. 101.1-
2, pp. 99–134. DOI: 10.1016/S0004-3702(98)00023-X. URL: https://doi.
org/10.1016/S0004-3702(98)00023-X.
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton (2012). “ImageNet Classification
with Deep Convolutional Neural Networks”. In: URL: https://papers.
nips.cc/paper/4824-imagenet-classification-with-deep-convolutionalneural-
networks.pdf.
Lecun, Yann et al. (1998). “Gradient-Based Learning Applied to Document Recognition”.
In: URL: http://yann.lecun.com/exdb/publis/pdf/lecun-
01a.pdf.
Martens, James and Roger B. Grosse (2015). “Optimizing Neural Networks with
Kronecker-factored Approximate Curvature”. In: CoRR abs/1503.05671. arXiv: 1503.
05671. URL: http://arxiv.org/abs/1503.05671.
Mnih, Volodymyr et al. (2015). “Human-level control through deep reinforcement
learning”. In: Nature 518.7540, pp. 529–533. DOI: 10.1038/nature14236. URL:
https://doi.org/10.1038/nature14236.
Mnih, Volodymyr et al. (2016). “Asynchronous Methods for Deep Reinforcement
Learning”. In: CoRR abs/1602.01783. URL: http://arxiv.org/abs/1602.
01783.
Rummery, G. A. and M. Niranjan (1994). On-Line Q-Learning Using Connectionist Systems.
Tech. rep.
Schulman, John et al. (2015). “Trust Region Policy Optimization”. In: CoRR abs/1502.05477.
arXiv: 1502.05477. URL: http://arxiv.org/abs/1502.05477.
BIBLIOGRAPHY 91
Shantia, A., E. Begue, and M. Wiering (2011). “Connectionist Reinforcement Learning
for Intelligent Unit Micro Management in StarCraft”. In: IEEE. URL: http:
//www.ai.rug.nl/~mwiering/GROUP/ARTICLES/StarCraft.pdf.
Silver, David et al. (2016). “Mastering the game of Go with deep neural networks
and tree search”. In: Nature 529.7587, pp. 484–489. DOI: 10.1038/nature16961.
URL: https://doi.org/10.1038/nature16961.
Silver, David et al. (2017). “Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm”. In: CoRR abs/1712.01815. arXiv: 1712.
01815. URL: http://arxiv.org/abs/1712.01815.
Smallwood, Richard D. and Edward J. Sondik (1973). “The Optimal Control of Partially
Observable Markov Processes over a Finite Horizon”. In: Operations Research
21.5, pp. 1071–1088. DOI: 10.1287/opre.21.5.1071. URL: https://doi.
org/10.1287/opre.21.5.1071.
SSCAIT. Student StarCraft AI Tournament and LadderWebsite. https://sscaitournament.
com/.
Stadie, Bradly C., Pieter Abbeel, and Ilya Sutskever (2017). “Third-Person Imitation
Learning”. In: CoRR abs/1703.01703. arXiv: 1703.01703. URL: http://arxiv.
org/abs/1703.01703.
Sutton, Richard S. (1988). “Learning to Predict by the Methods of Temporal Differences”.
In: Machine Learning 3, pp. 9–44. DOI: 10.1007/BF00115009. URL:
https://doi.org/10.1007/BF00115009.
Szegedy, Christian et al. (2014). “Going Deeper with Convolutions”. In: CoRR abs/1409.4842.
URL: http://arxiv.org/abs/1409.4842.
Tieleman Tijmen and Hinton Geoffrey (2012). Lecture 6.5-rmsprop: Divide the gradient
by a running average of its recent magnitude. "COURSERA: Neural Networks for
Machine Learning".
Vinyals, Oriol et al. (2017). “StarCraft II: A New Challenge for Reinforcement Learning”.
In: CoRR abs/1708.04782. URL: http://arxiv.org/abs/1708.04782.
Wang, Ziyu, Nando de Freitas, and Marc Lanctot (2015). “Dueling Network Architectures
for Deep Reinforcement Learning”. In: CoRR abs/1511.06581. arXiv:
1511.06581. URL: http://arxiv.org/abs/1511.06581.
Watkins, Christopher J. C. H. and Peter Dayan (1992). “Technical Note Q-Learning”.
In: Machine Learning 8, pp. 279–292. DOI: 10.1007/BF00992698. URL: https:
//doi.org/10.1007/BF00992698.
Wender, Stefan and Ian Watson (2012). “Applying Reinforcement Learning to Small
Scale Combat in the Real-Time Strategy Game StarCraft:Broodwar”. In: IEEE. URL:
http://geneura.ugr.es/cig2012/papers/paper44.pdf.
Williams, Ronald J. (1992). “Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning”. In: Machine Learning 8, pp. 229–256. DOI:
10.1007/BF00992696. URL: https://doi.org/10.1007/BF00992696.
Wu, Yuhuai et al. (2017). “Scalable trust-region method for deep reinforcement learning
using Kronecker-factored approximation”. In: CoRR abs/1708.05144. URL: http:
//arxiv.org/abs/1708.05144.
Zhang, Jian, Yuxin Peng, and Mingkuan Yuan (2018). “Semi-supervised Cross-modal
Hashing by Generative Adversarial Network”. In: CoRR. arXiv: 1802 . 02488.
URL: https://arxiv.org/abs/1802.02488.
Zhang, Shangtong and Richard S. Sutton (2017). “A Deeper Look at Experience Replay”.
In: CoRR abs/1712.01275. arXiv: 1712.01275. URL: http://arxiv.
org/abs/1712.01275.
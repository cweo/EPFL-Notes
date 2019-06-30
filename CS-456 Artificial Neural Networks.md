## Quiz

### Lecture 1: Simple Perceptrons for Classification

Quiz: biological neural networks

[x] Neurons in the brain have a threshold.

[  ] Learning means a change in the threshold.

[x] Learning means a change of the connection weights

[x] The total input to a neuron is the weighted sum of individual inputs

[  ] The neuronal network in the brain is feedforward: it has no recurrent connections



Quiz: Classification versus Reinforcement Learning

[x] Classification aims at predicting the correct category such as ‘car’ or ‘dog’

[  ] Classification is based on rewards

[x] Reinforcement learning is based on rewards

[x] Reinforcement learning aims at optimal action choices



Quiz: Perceptron algorithm

The input vector has N dimensions and we apply a perceptron algorithm.

[  ] A change of parameters corresponds always to a rotation of the separating hyperplane in N dimensions.

[x] A change of the separating hyperplane implies a rotation of the hyperplane in N+1 dimensions.

[  ] An increase of the length of the weight vector implies an increase of the distance of the hyperplane from the origin in N dimensions.

[  ] An increase of the length of the weight vector implies that the hyperplane does not change in N dimensions

[x] An increase of the length of the weight vector implies that the hyperplane does not change in N+1 dimensions



### Lecture 2: Backprop and multilayer perceptrons

Backprop: Quiz

[x] BackProp is nothing else than the chain rule, handled well.

[  ] BackProp is just numerical differentiation

[x] BackProp is a special case of automatic algorithmic differentiation

[x] BackProp is an order of magnitude faster than numerical differentiation



Curve fitting: Quiz

[x] 20 data points can always be perfectly well fit by a polynomial with 20 parameters

[  ] The prediction for future data is best if the past data is perfectly fit

[  ] A sin-function on [0,2p] can be well approximated by a polynomial with 10 parameters



Regularization: Quiz

If we increase the penalty parameter

[  ] the flexibility of the fitting procedure increases

[x] the flexibility of the fitting procedure decreases

[x] the ‘effective’ number of free parameters decreases

[  ]  the ‘effective’ number of free parameters remains the same

[x] the ‘explicit’ number of parameters remains the same



### Lecture 3: Statistical classification by deep networks

QUIZ: Maximum likelihood solution means

[  ] find the unique set of parameters that generated the data

[x] find the unique set of parameters that best explains the data

[x] find the best set of parameters such that your model could have generated the data



Miminization of the cross-entropy error function for single class output

[x] is consistent with the idea that the output $\hat{y}_1$of your network can be interpreted as $\hat{y}_1 = \Pr(C_1|x)$

[  ] guarantees that the output $\hat{y}_1$of your network can be interpreted as $\hat{y}_1 = \Pr(C_1|x)$ 



QUIZ: Modern Neural Networks

[  ] piecewise linear units should be used in all layers

[x] piecewise linear units should be used in the hidden layers

[x] softmax unit should be used for exclusive multi-class in an output layer in problems with 1-hot coding

[x] sigmoidal unit should be used for single-class problems

[x] two-class problems (mutually exclusive) are the same as single-class problems

[x] multiple-attribute-class problems are treated as multiple-single-class

[  ] In neural nets we can interpret the output as a probability, $\hat{y}_1 = \Pr(C_1|x)$ 

[x] if we are careful in the model design, we may interpret the output as a probability that the data belongs to the class



### Lecture 4: Regularization and Tricks of the Trade in deep networks

[x] If you want to win a machine learning competition, it is better to average the prediction on new data over ten different models, rather than just using the model that is best on your validation data.

[x] If you want to win a machine learning competition,  it is better to hand in 10 contributions (using different author names) rather than a single contribution.



[x] forward propagation with ReLu leaves only a few active paths

[x] back propagation with ReLu leaves only a few active paths

[x] a non-zero weight update step of ReLu shifts most often the mean

[x] forward propagation with ReLu is always linear on the active paths

[  ] in a ReLu network all patterns are processed with the same linear filter

[x] in a sigmoidal network with small weights (and normalized inputs) all patterns are processed with the same linear filter

[x] in a sigmoidal network with big weights, there are active units in the forward pass that contribute a vanishing gradient in the backward path

[  ] in a network with SELU, there are active units in the forward path which contribute a vanishing gradient in the backward path????

[  ] a non-zero the weight update step of SELU shifts the mean



### Lecture 5: Error landscape and optimization methods for deep networks

Quiz: Strengthen your intuitions in high dimensions

1. A deep neural network with 9 layers of 10 neurons each

[  ] has typically between 1 and 1000 minima (global or local)

[x] has typically more than 1000 minima (global or local)

2. A deep neural network with 9 layers of 10 neurons each

[ ] has many minima and in addition a few saddle points

[ ] has many minima and about as many saddle points

[x] has many minima and even many more saddle points



Quiz: Strengthen your intuitions in high dimensions

A deep neural network with many neurons

[  ] has many minima and a few saddle points

[  ] has many minima and about as many saddle points

[x] has many minima and even many more saddle points

[x] gradient descent is slow close to a saddle point

[  ] close to a saddle point there is only one direction to go down

[x] has typically many equivalent ‘optimal’ solutions

[x] has typically many near-optimal solutions



Quiz: Momentum

[x]  momentum speeds up gradient descent in ‘boring’ directions

[x] momentum suppresses oscillations

[  ] with a momentum parameter a=0.9 the maximal speed-up is a factor 1.9

[x] with a momentum parameter a=0.9 the maximal speed-up is a factor 10

[  ] Nesterov momentum needs twice as many gradient evaluations as standard momentum



Quiz: No Free Lunch (NFL) Theorems

Take neural networks with many layers, optimized by Backprop as an example of deep learning

[x] Deep learning performs better than most other algorithms on real world problems.

[x] Deep learning can fit everything.

[  ] Deep learning performs better than other algorithms on all problems.





### Lecture 8: Reinforcement Learning and SARSA

Quiz: Rewards in Reinforcement Learning

[x] Reinforcement learning is based on rewards

[x] Reinforcement learning aims at optimal action choices

[  ] In chess, the player gets an external reward after every move

[x] In table tennis, the player gets a reward when he makes a point

[x] A dog can learn to do tricks if you give it rewards at appropriate moments



Quiz: Exploration – Exploitation dilemma

We use an iterative method and update Q-values with eta=0.1

[  ] With a greedy policy the agent uses the best possible action

[x] Using an epsilon-greedy method with epsilon = 0.1 means that, even after convergence of Q-values, in at least 10 percent of cases a suboptimal action is chosen.

[x] If the rewards in the system are between 0 and 1 and Q-values are initialized with Q=2, then each action is played at least 5 times before exploitation starts.



Quiz: Exploration – Exploitation dilemma

All Q values are initialized with the same value Q=0.1. Rewards in the system are r =0.5 for action 1 (always) and r=1.0 for action 2 (always)

We use an iterative method and update Q-values with eta=0.1

[x] if we use softmax with beta = 10, then, after 100 steps, action 2 is chosen almost always

[  ] if we use softmax with beta = 0.1, then action 2 is taken about twice as often as action 1.



Quiz: Bellman equation with policy $\pi$

[  ] The Bellman equation is linear in the variables $Q(s’,a’)$

[  ] The set of variables $Q(s’,a’)$ that solve the Bellman equation is unique and does not depend on the policy



SARSA algorithm

We have initialized SARSA and played for n>2 steps. Is the following true?

[x] in SARSA, updates are applied after each move.

[  ] in SARSA, the agent updates the Q-value $Q(s(t),a(t))$related to the current state $s(t)$

[x] in SARSA, the agent updates the Q-value $Q(s(t-1),a(t-1))$  related to the previous state, when it is in state $s(t)$

[x] in SARSA, the agent moves in the environment using the policy $\pi(s,a)$

[x] SARSA is an online algorithm



### Lecture 9: Variants of TD-learning methods and continuous space

Quiz: TD methods in Reinforcement Learning

[x] SARSA is a TD method

[x] expected SARSA is a TD method

[x] Q-learning is a TD method

[x] TD learning is an on-policy TD method

[  ] Q-learning is an on-policy TD method

[x] SARSA is an on-policy TD method



Quiz: Monte Carlo methods

We have a network with 1000 states and 4 action choices in each state. There is a single terminal state.

We do Monte-Carlo estimates of total return to estimate Q-values.

Our episode starts with $(s,a)$ that is 400 steps away from the terminal state. How many return $R(s,a)$ variables do I have to open in this episode?

[  ] one, i.e. the one for the starting configuration $(s,a)$

[  ] about 100 to 400

[x] about 400 to 4000

[  ] potentially even more than 4000



Quiz: Eligibility Traces

[x] Eligibility traces keep information of past state-action pairs.

[x] For each Q-value $Q(s,a)$, the algorithm keeps one eligibility trace $e(s,a)$, i.e., if we have 200 Q-values we need 200 eligibility traces

[x] Eligibility traces enable information to travel rapidly backwards into the graph

[x] The update of $Q(s,a)$ is proportional to $[r-(Q(s,a)-Q(s’,a’))]$

[x] In each time step all Q-values are updated



### Lecture 10: Policy Gradient

Quiz: Policy Gradient and Reinforcement learning

[  ] All reinforcement learning algorithms work either with Q-values or V-values

[  ] The transition from batch to online is always easy: you just drop the summation signs and bingo!

[X] All reinforcement learning algorithms try to optimize the expected total reward (potentially discounted if there are multiple time steps)

[  ] The derivative of the log-policy is some abstract quantity that has no intuitive meaning.



Quiz: Policy Gradient and Reinforcement learning

[x] Even some policy gradient algorithms use V-values

[x] V-values for policy gradient are calculated in a separate network (but some parameters can be shared with the actor network)

[x] REINFORCE with baseline uses Monte-Carlo estimates of V-values



### Lecture 11: Reinforcement Learning and the Brain

Quiz: Policy Gradient and Reinforcement learning

[x] Even some policy gradient algorithms use V-values

[x] V-values for policy gradient are calculated in a separate network (but some parameters can be shared with the actor network)

[x] The actor-critic network has basically the same architecture as REINFORCE with baseline

[x] While actor-critic uses ideas from TD learning, REINFORCE with baseline uses Markov estimates of V-values

[x] Eligibility traces are ‘shadow’ variables for each parameter

[x] Eligibility traces appear naturally in policy gradient algos.







## Learning outcomes

- apply learning in deep networks to real data

- assess/evaluate performance of learning algorithms

- Elaborate relations between different mathematical concepts of learning

- judge limitations of learning algorithms
- propose models for learning in deep networks



## Lecture 1: Simple Perceptrons for Classification

- understand classification as a geometrical problem
- discriminant function of classification
- linear versus nonlinear discriminant function
- perceptron algorithm
- gradient descent for simple perceptrons



A generic data base for supervised learning requires a nonlinear discriminant function

A simple perceptron can only implement a linear discriminant function: the separating hyperplane

The perceptron algorithm turns the separating hyperplane in N+1 dimensions



Three factors for changing a connection

- activity of neuron j

- activity of neurone i

- success

To implement a change of synaptic weights, three factors are needed:

- the activity of the sending neuron;
- the activity of the receiving neuron;
- and a broadcast signal that transmit the information: this was successful (because it led to a reward).

In biology a basic idea is that joint activity of two neurons can change the weight that connects those two neurons (Hebb rule).



Network for choosing action: action: Advance king

2nd output for value of action: probability to win

- The main outputs are the actions such as ‘move king to the right’

- An auxiliary output predicts the ‘value’ of each state. It can be used to explore possible next positions so as to pick the one with the highest value.

- The value can be interpreted as the probability to win (given the position)



## Lecture 2: Backprop and multilayer perceptrons

- XOR problem and the need for multiple layers

- understand backprop as a smart algorithmic implementation of the chain rule

- hidden neurons add flexibility, but flexibility is not always good: the problem of generalization

- training base, validation and test base: the need to predict well for future data

For a network consisting of a single neuron, the discriminant function is a hyperplane.

An error on the validation base that is much larger than the error on the training base is a signature of overfitting.

Convergence

- To local minimum
- No guarantee to find global minimum
- Learning rate needs to be sufficiently small
- Learning rate can be further decreased once you are close to convergence

Flexibility = ‘effective’ number of free parameters

assigns an ‘error’ to flexible solutions

check ‘normal’ error on separate data (validation set)

The logic is that ‘curvy’ separating surface requires big positive and negative weights, whereas zero weights or tiny weights enable no or very litte curvature only.

Early stopping:

network ‘uses’ its total flexibility only after lengthy optimization

go back to ‘earlier’ solution

maximal flexibility not exploited

Therefore the flexibility of a network increases over training. Early stopping means stopping before the maximal flexibility is exploited.

go back later to the ‘earlier’ solution.



While validation on the validation set is performed using the NORMAL error function, training is done on the training set using an error function that includes a penalty term.

The sum in the penalty term runs over all weights, but not the thresholds!!!.



There could be a potential problem: we added lambda as one of the parameters (sometimes called a hyper-parameter). To optimize lambda we use the validation base. But this logically implies that we can not consider the validation based as ‘future data’!

In fact, the same logic also applies to the earlier scheme where we changed the explicit number of neurons - the neuron number is also a hyperparameter which is optimized by exploiting the validation base.

network ‘uses’ its total flexibility only after lengthy optimization

In order to implement a flexible surface with high curvature, big weights are needed. Therefore the flexibility of a network increases over training.



## Lecture 3: Statistical classification by deep networks

- The cross-entropy error is the optimal loss function for classification tasks

- The sigmoidal (softmax) is the optimal output unit for classification tasks

- Multi-class problems and ‘1-hot coding’

- Under certain conditions we may interpret the output as a probability

- The rectified linear unit (RELU) for hidden layers (Piecewise linear units are preferable for hidden layers)

What is the likelihood that my set of P data points could have been generated by my model?



In the exercises this week, you will show that the conditions that

(i) outputs are probabilities

(ii) probabilities add up to one,

imply the ‘softmax’ output function



Can we be sure that the output will represent the probability?

A We will need enough data for training (not just 10 data points for a complex task)

B We need a sufficiently flexible network (not a simple perceptron for XOR task)



The calculations with discrete bins ∆𝑥 show that the notions ‘enough examples’ and ‘flexible enough’ are linked to each other: we need enough data samples in each bin to reliably estimate the fraction of positive examples in a bin; and the network must have enough flexibility to output for each bin a different value.





## Lecture 4: Regularization and Tricks of the Trade in deep networks

- Bagging
- Dropout
- What are good units for hidden layers?
- Rectified linear unit (RELU)
- Shifted exponential linear (ELU and SELU)
- BackProp: Initialization
- Linearity problem, vanishing gradient problem, bias problem
- Batch normalization



To answer this question, we will look at the BackProp algorithm and focus on values

x = ±𝜀 where epsilon is a small number. 

But also at values x = ±𝛼 with alpha of order one.



Dropout two interpretations:

1. An approximate, but practical implementation of bagging

2. A tool to enforce representation sharing in the hidden neurons



the validation loss could make (together with the training loss) a big jump downward a long time after having passed through a first minimum.



- initialization is important so as to exploit nonlinearities

- choice of hidden unit is important in initial phase of training

- ReLu has disadvantages in keeping the mean: batch normalization

- Tanh has problems with vanishing gradient
- Sigmoidal has problems with vanishing gradient and mean
- SELU solves all problems and is currently best choice



- Bagging: multiple models help always to improve results!
- Dropout: two interpretations
  - (i) a practical implementation of bagging
  - (ii) forced feature sharing
- BackProp: Initialization, nonlinearity, and symmetry
- What are good units for hidden layers?
  - problems of vanishing gradient and shift of mean
  - solved by Shifted exponential linear (SELU)
- Batch normalization: necessary for ReLu



For testing you use the full network with all hidden units.

However, since there are now twice as many hidden units as during training, you need to multiply the output weights by factor ½, so that the typical input to a unit in the next layers is roughly the same as during training.







## Lecture 5: Error landscape and optimization methods for deep networks

- Error function landscape: minima and saddle points
  - Random matrix?
- Momentum
- Adam
- No Free Lunch
- Shallow versus Deep Networks



A momentum term suppresses these oscillations while giving rise to a ‘speed-up’ in the directions where the gradient does not change



Reading for this lecture:

Goodfellow et al.,2016 Deep Learning

- Ch. 8.2, Ch. 8.5
- Ch. 4.3
- Ch. 5.11, 6.4, Ch. 15.4, 15.5

Idea: estimate mean and variance from k=1/𝛼 samples

Note that a momentum term with weight a can be seen as a running average of the gradient of roughly 1/a examples (see Exercises).

Note that second moment and variance are not exactly the same (see also exercises). For variance, you subtract the mean before you square.

Result: parameter movement slower in uncertain directions (ex1)

The idea is that (as we have seen for the momentum term earlier) evaluating a constant gradient using a momentum term with parameter r gives effectively rise to a factor 1/[1-r]

However, since it takes some time to build up this factor, one could artificially introduce this factor in the first few time steps – and this is what is done in this algorithm.



- Error function landscape: there are many good minima and even more saddle points

- Momentum: gives a faster effective learning rate in boring directions

- Adam: gives a faster effective learning rate in low-noise directions

- No Free Lunch: no algo is better than others
- Deep Networks: are better than shallow ones on real-world problems due to feature sharing



Sec.5 distributed representation??





## Lecture 6: Sequences and Recurrent Networks

- Why are sequences important?

- Long-term dependencies in sequence data

- Sequence processing with feedforward models
- Sequence processing with recurrent models
- Vanishing Gradient Problem
- Long-Short-Term Memory (LSTM)
- Application: Music generation



Objectives for today:

- Why are sequences important? 
  - they are everywhere; labeling is (mostly) for free

- Long-term dependencies in sequence data
  - unknown time scales, fast and slow

- Sequence processing with feedforward models
  - corresponds to n-gram=finite memory
- Sequence processing with recurrent models
  - potentially unlimited memory, but:

- Vanishing Gradient Problem
  - error information does not travel back beyond a few steps

- Long-Short-Term Memory (LSTM)
  - explicit memory units keep information beyond a few steps

- Application: Music generation



## Lecture 7: Convolutional Neural Networks

-  Review: No free-lunch theorem and inductive biases
-  Convolution Layers
-  Max Pooling Layers
-  Inductive bias of ConvNets
-  ImageNet competition and modern ConvNets
-  Training ConvNets with AutoDiff
-  Applications beyond object recognition



Inductive bias of a convolutional layer

1)  Equivariance to translation

2)  Only local interactions matter

A conv layer is like a standard dense layer with

- many neurons having the same weights
-  many weights zero (except in a small neighborhood)



## Lecture 8: Reinforcement Learning and SARSA

- Reinforcement Learning is learning by rewards
- Agents and actions
- Exploration vs Exploitation
- Bellman equation
- SARSA algorithm



Reward information is available in the brain

Neuromodulator dopamine: Signals reward minus expected reward

(ii) If the learning rate $\eta$ decreases, fluctuations around $\mathbb{E}[\hat{Q}(s,a)] = Q(s,a)$ decrease.



- Reinforcement Learning is learning by rewards
  - world is full of rewards (but not full of labels)

- Agents and actions
  - agent learns by interacting with the environment
  - state s, action a, reward r

- Exploration vs Exploitation
  - optimal actions are easy if we know reward probabilities
  - since we don’t know the probabilities we need to explore

- Bellman equation
  - self-consistency condition for Q-values

- SARSA algorithm: state-action-reward-state-action
  - update while exploring environment with current policy



## Lecture 9: Variants of TD-learning methods and continuous space

- TD learning refers to a whole class of algorithms
- There are many Variations of SARSA
- All set up to iteratively solve the Bellman equation
- Eligibility traces and n-step Q-learning to extend over time
- Continuous space and ANN models
- Models of actions and models of value



reward must account for the difference in Q-values between neighboring states.

Backup diagrams

- Expected SARSA
- Q-learning
  - Run with exploration
  - Update with greedy
  - it is as if you turn-off the current policy during the update.
  - Neighbours are one time step away

Neighboring states: neighboring time steps

The averaging step in TD methods (‘bootstrap’) is more efficient (compared to Monte Carlo methods) to propagate information back into the graph, since information from different starting states is combined and compressed in a Q-value or V-value.

therefore we can keep the flow constant even if the discretization changes by readjusting $\lambda$.

TD algorithms do not scale correctly if the discretization is changed

either Introduce eligibility traces (temporal smoothing)

or Switch from 1-step TD to n-step TD (temporal coarse graining)



2. Variations of SARSA
3. TD – learning (Temporal Difference)
4. Monte-Carlo methods
5. Eligibility traces and n-step methods
6. Modeling the input space



Basis of all:

iterative solution of Bellman equation



- TD – learning (Temporal Difference)
  - work with V-values, rather than Q-values

- Variations of SARSA

  - off-policy Q-learning (greedy update)

  - Monte-Carlo

  - n-step Bellman equation

- Eligibility traces

  - allows rescaling of states, smoothes over time

  - similar to n-step SARSA
- Continuous space
  - use neural network to model and generalize



Q(s,a) continuous case



## Lecture 10: Policy Gradient Methods

- basic idea of policy gradient: learn actions, not Q-values
- log-likelihood trick: getting the correct statistical weight
- policy gradient algorithms
- why subtract the mean reward?
- reinforce with baseline (see actor critic)



Eligibility traces enable to connect the reward at the end to states several steps before.

Q-values as a SMOOTH function of the input enables generalization. Hidden layers of neural networks are able to extract compressed representations of the input space that introduce heuristic but useful notion of what it means that two states are ‘similar’ or ‘neighbors.

Similar states



The family of functions can be defined by the parameters of a Neural Network or by the parameters of a linear superposition of basis functions.



3. Policy Gradient methods: Batch-to-Online



All updates done AT THE END of the episode

Algorithm maximizes expected discounted rewards starting at S0



(Unfortunately, the minimal noise is not exactly the situation where one subtracts the mean, but it is close to it).



- basic idea of policy gradient: learn actions, not Q-values
  - gradient ascent of total expected discounted reward

- log-likelihood trick: getting the correct statistical weight
  - enables transition from batch to online

- policy gradient algorithms
  - updates of parameter propto

- why subtract the mean reward?
  - reduces noise of the online stochastic gradient

- Reinforce with baseline
  - a further output to subtract the mean reward



## Lecture 11: Reinforcement Learning and the Brain

1. Review Policy gradient

2. Review Subtracting the mean via the value function
3. Actor-Critic
4. Eligibility traces for policy gradient
5. Actor-Critic in the Brain
6. Application: Rat navigation
7. Model-based versus Model-free



- Actor-critic

- three-factor learning rules can be implemented by the brain
- eligibility traces as ‘candidate parameter updates’
- the dopamine signal has signature of the TD error
- model-based versus model-free



Subtracting the baseline:

(i) Subtract the mean return (=value V) in a multistep-horizon algorithm (or the mean reward in a one step-horizon algorithm).

This is what we consider here in this section

(ii) Subtract mean expected reward PER TIME STEP (related to the delta-error) in a multi-step horizon algorithm.

This is what we will consider in section 3 under the term Actor-Critic.

**Eligibility traces appear naturally in policy gradient algos.**



We just learn ‘arrows’: what is the next step (optimal next action), given the current state?



- policy gradient algorithms
  -  updates of parameter propto

- why subtract the mean reward?
  - reduces noise of the online stochastic gradient

- actor-critic framework
  - combines TD with policy gradient

- eligibility traces as ‘candidate parameter updates’
  - true online algorithm, no need to wait for end of episode

- Differences of model-based vs model-free RL
  - play out consequences in your mind by running the state transition model wait

- three-factor learning rules can be implemented by the brain
  - weight changes need presynaptic factor, postsynaptic factor and a neuromodulator (3rd factor)
  - actor-critic and other policy gradient methods give rise to very similar three-factor rules
- eligibility traces as ‘candidate parameter updates’
  -  set by joint activation of pre- and postsynaptic factor
  - decays over time
  - transformed in weight update if dopamine signal comes
- the dopamine signal has signature of the TD error
  - responds to reward minus expected reward
  - responds to unexpected events that predict reward



## Lecture 12: Use Cases of Deep Reinforcement Learning

- A3C, DQN and decorrelation for deep RL

- RL in the ATARI domain

- Replay Memory and Backward Planning in tabular environments

- Forward Planning in model-based RL (board games): Minimax vs. Monte Carlo Tree Search
- Alpha Zero
- Limitations of deep RL



with replay memory

-Ineﬃcient updates

-Needs more memory/computation

+Sample eﬃciency



standard Q-Learning

+Convergence

+Minimal memory/computation

-Sample ineﬃciency



prioritized sweeping

Organize memory in table

Prioritize backups cleverly

+Convergence

-Needs more memory/computation

+Highest sample eﬃciency



Prioritized sweeping is a backward-focusing planning method.

dyna


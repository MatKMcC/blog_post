# Introduction

Plans in healthcare have historically been priced in bulk, which is done to alleviate risk to individuals by putting money into pools. When predicting for this cost, it is typical to use aggregated features such as historical account cost and account demographics.

Growing capabilities to handle big and unstructured data as well as advances in modeling make individual-level predictions increasingly feasible and an attractive alternative to modeling at the aggregate level. Individual-level models can utilize medical data including lab values, diagnoses, prescriptions, or even doctors' notes to predict the cost of a single patient. However, there are risks in naively aggregating individual-level predictions to account-level predictions without due considerations to potentially exaggerating model error.

## Advantages of individual predictions

An account-level view is an extreme simplification of a very complex cost relationship between patient health, hospitals, and providers. A more detailed look at patient medical claims, for example, could provide more accurate predictions of cost. Claims data consist of thousands of possible medical codes, each of which represents an event in a patient's medical history. Patient-level models can discover complex interactions between these codes which are predictive of cost. For this purpose, Lumiata has built a robust patient tagging process capable of translating claims data into ingestible patient timelines. We leverage this in three ways: 

- **Depth** - Patient timelines are often independent of group timelines. A patient-level view allows for flexible aggregation, which is not restricted by the lifetime of the group. This helps with the cold start problem (what to predict for new groups) and enables us to use the full patient timeline. 

- **Reactivity** - It is known that patients with specific diseases can represent a disproportionate cost of medical care, but these patients are obscured at the group level. Identifying high cost disease states before they drive up costs can be a huge advantage.

- **Data, Data, Data** - A patient-level view allows use of more features, more complex models (e.g., deep learning and boosted trees), and more training examples. Due to aggregation group-level models are limited to only a few thousand training examples, reducing the ability to detect complex patterns.

## Concerns with individual predictions

But with great power comes great responsibility. Approaching cost prediction at the patient-level introduces complexity that must be tightly controlled. There are several characteristics of healthcare cost that makes this especially relevant:

- **Extreme Outliers** - High cost patients can exert disproportionate weight in cost functions.
- **Sparse Data** - The high dimensionality of the feature space implies that most features are null most of the time.
- **Noisy Signals** - Even for routine medical practice, diagnosis, treatment, and cost can vary from doctor to doctor and patient to patient.

More complex models (with low bias, see below) can adapt to these characteristics, but risk over-fitting (high variance). Below I'll review the bias-variance tradeoff and how we can use hyperparameters to control model complexity.


# Bias-variance review

For a more in-depth review of the bias-variance trade-off, I would suggest David Dalpiaz's great online resource, [R for Statistical Learning](https://daviddalpiaz.github.io/r4sl/biasvariance-tradeoff.html). The examples I am providing here are adapted from this resource.

Assume some random vector $(X,Y)$ with values in $\mathbb{R}^p \times \mathbb{R}$ and define $f(x)$ for ${E}(Y|X=x)$. Notice that this form of $f(x)$ minimizes the expected squared error, representing the best possible prediction we can make. Since $f(x)$ is unknown, we approximate it with $\hat f(x)$ using some training data $\mathcal{D}$, and our favorite machine learning algorithm. Note that when I refer to "algorithm," I am referencing the method used to learn a specific model.

Using these definitions and conditioning on $X$, observe that the expected value of prediction can be expanded into two separate components **reducible error** and **irreducible error**:

$$\mathbb{E}[(Y - \hat f(x))^2|X=x] = \underbrace{\mathbb{E}[(f(x) - \hat f(x))^2]}_\textrm{reducible error} + \underbrace{\sigma^2}_\textrm{irreducible error}$$

Reducible error is what we strive to \**drumroll** reduce as it is a measure of our approximation of $f(x)$ with $\hat f(x)$. Irreducible error, which is equal to $V(Y|X=x$), on the other hand, is simply not a learnable function of $X$ and should be recognized as noise. From reducible error, we can further derive **bias** and **variance**.

$$\mathbb{E}[(Y - \hat f(x))^2|X=x] = \underbrace{(f(x) - \mathbb{E}[\hat f(x)])^2}_{bias^2} + \underbrace{\mathbb{E}[(\hat f(x) - \mathbb{E}[\hat f(x)])^2]}_{variance} + \underbrace{\sigma^2}_\textrm{irreducible error}$$

Bias is a measure of the deviation of the expected form of our models and $f(x)$. The word "expected" means that the model is a function of the underlying data that an algorithm is trained on, which is it itself a random variable. Variance on the other hand measures the expected deviance of $\hat f(x)$ from the expected fit of  $\hat f(x)$.

It is always possible to have completely unbiased models with high variance by perfectly fitting the training data---but it would change significantly depending on the input, and so would generalize poorly (i.e., overfit). In order to lower the variance, the model must make certain generalizing *assumptions*. The more such assumptions it makes, the lower the variance---but at the cost of bias, if our assumptions turn out to be incorrect. For example, if we fit linear regression for expected true values that are not linear in the features, then that's a bad assumption that leads to bias; it does, however, decrease variance.

## Visualizing bias and variance

To demonstrate the bias-variance tradeoff, I repeatedly fit polynomial models to simulated data (normally distributed random points with mean $x^2$) as defined in the following code snippet. 

```python
import numpy as np

def f(x):
    return x ** 2

def get_sim_data(f, var, sample_size=100):
    x = np.random.uniform(size=sample_size, low=0, high=1.5)
    y = np.random.normal(size=sample_size, loc=f(x), scale=var)
    return x, y
```

Below I used 3 algorithms: $\hat f_k(x) = \ W_0 + \sum_{i=1}^kW_k x^k$ for $k \in \{1,2,10\}$, representing a biased algorithm (k=1), an unbiased and low variance algorithm (k=2), and an unbiased high-variance algorithm (k=10).

![](https://i.imgur.com/8pE41Mw.jpg)

In the above plot $\hat f_1$ seems quite consistent with varying training data, even if it is missing the true form of $f$ (defined above as $x^2$). In contrast observe that $\hat f_{10}$, while following the trend of the data, seems to vary a large deal from simulation to simulation.

Below I repeat the above simulations 100 times, and for each model calculate $f(x) - \hat f_{k}(x)$, where $x=.8$. Notice that the center of each resulting distribution indicates the bias introduced by the algorithm and the width of distribution indicates the variance of the fit:

![](https://i.imgur.com/BzXHZ0A.png)

There are several things to notice here: for the algorithm with bias, $\hat f_1$, we see that the distribution of fit is not centered around 0; however, it is relatively tight. As $k$ increases, bias is reduced but the dispersion of fit also increases. Additionally, notice that even for the well-conditioned algorithm, there is inherent randomness in the fitting process.

# The risks of aggregation

What happens to error with aggregated models? A naive approach to predictions at the group level is to train a patient-level model, and then for each group set the aggregate prediction to the sum of patient  predictions for every group member. The question then arises: does this procedure optimize for the group-level error? As it turns out, the answer is no!

To see this, consider the following heuristic. Let a group of size $N$ consist of individuals with feature values $\{x_1,\ldots,x_N\}$, and true costs $\{y_1,\ldots,y_N\}$, and further assume that costs are independent across patients. For the sake of simplicity, assume that $x_1=\ldots=x_N:= x$. (If naive aggregation fails with even this assumption, there is no hope in general.) Then:
$$y_i = f(x) + \varepsilon_i$$

where $f(x):=E(Y|X=x)$, and the $\varepsilon_i$'s are independent identically distributed variables with mean 0.

As before, the patient-level error for $X=x$ has a decomposition:
$$
 \mathbb{E}[(Y_i - \hat f(x))^2|X=x] = \underbrace{\mathbb{E}[(f(x) - \hat f(x))^2]}_\textrm{reducible error} + \underbrace{\sigma^2}_\textrm{irreducible error}$$
 
where $V(\varepsilon_i)=\sigma^2$ for all $i$.

Let's look now at the group-level error:

$$
\begin{align}
\mathbb{E}[(Y_1 + ... +Y_N - N\hat f(x))^2|X=x] &= \underbrace{\mathbb{E}[(Nf(x) - N\hat f(x))^2]}_\textrm{reducible error} + \underbrace{V(\varepsilon_1 + ...+\varepsilon_N)}_\textrm{irreducible error}\\\\
&=\underbrace{N^2\mathbb{E}[(f(x)-\hat f(x))^2]}_\textrm{reducible error} + \underbrace{N\sigma^2}_\textrm{irreducible error}
\end{align}
$$

because of the linearity of expectation, and that fact that variance of a sum of independent variables is the sum of the variances of those variables.

Thus the error of the aggregation is *not* a linear multiple of the sum of the errors of the individual predictions! At the aggregated level, reducible error is multiplied by the square of $N$ (in fact, both bias and variance are multiplied by the square of $N$), but the reducible error is multiplied only by $N$.


# Discussion

In the above analysis we can notice, somewhat  trivially, that bias compounds. Someone without much experience in data science might think that we may underpredict a little here and overpredict a little there..but hey! it all adds up to a null sum! This brings to mind the old joke "We lose money on every sale, but make it up on volume!". That is, when you add up many small losses you get a big loss!

In healthcare, we are even more likely to compound our model error because accounts tend to be more homogenous than the general population; that is, members within accounts are more similar than members without. For example, consider a logging company (the most accident prone job in America). In this case members have higher expected cost than the general population. Since our predicted cost $\hat f(x)$ is less than the expected cost $f(x)$, we have introduced bias. Furthermore, since we are predicting in aggregate for every member of that account, we will be compounding our bias for each prediction! We can generalize this to any account that has characteristics that are not reflected in the general population. 


# Making Healthcare Smarter

Optimizing patient predictions for aggregate performance is an area of active research at Lumiata. Below are some strategies we employ when predicting at the group-level:

-  Optimize patient predictions for group level performance.
-  Evaluate group predictions for signs of aggregated error.
-  Build additional corrective models on aggregated predictions using group-level features.

Following this roadmap we can leverage patient-level data while avoiding the pitfalls that would accompany naive aggregation. 

If you are interested in building and scaling cool models with healthcare data, [Lumiata is hiring](https://www.lumiata.com/careers.html)!

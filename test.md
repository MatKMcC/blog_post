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

Assume some random vector ![]https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/1c58aaaffa6c79566e4f46c7bf98769f.svg?invert_in_darkmode&sanitize=true) with values in <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/2915c66371f4d195940257920ebb9ddc.svg?invert_in_darkmode&sanitize=true" align=middle width=51.43392539999999pt height=22.648391699999998pt/> and define <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> for <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/47b07613dce14a6d5d9f06d4f34889d1.svg?invert_in_darkmode&sanitize=true" align=middle width=89.85149909999998pt height=24.65753399999998pt/>. Notice that this form of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> minimizes the expected squared error, representing the best possible prediction we can make. Since <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> is unknown, we approximate it with <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/101b25d063b532a19ae95b5c5b11acd7.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=31.50689519999998pt/> using some training data <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode&sanitize=true" align=middle width=13.13706569999999pt height=22.465723500000017pt/>, and our favorite machine learning algorithm. Note that when I refer to "algorithm," I am referencing the method used to learn a specific model.

Using these definitions and conditioning on <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>, observe that the expected value of prediction can be expanded into two separate components **reducible error** and **irreducible error**:

<p align="center"><img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/d796fca0292d71f8f1ebd54e239c3aee.svg?invert_in_darkmode&sanitize=true" align=middle width=413.83603965pt height=42.429156pt/></p>

Reducible error is what we strive to \**drumroll** reduce as it is a measure of our approximation of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/> with <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/101b25d063b532a19ae95b5c5b11acd7.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=31.50689519999998pt/>. Irreducible error, which is equal to <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/193a361d9be57260a7709eb40d720c13.svg?invert_in_darkmode&sanitize=true" align=middle width=83.61863729999999pt height=24.65753399999998pt/>), on the other hand, is simply not a learnable function of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> and should be recognized as noise. From reducible error, we can further derive **bias** and **variance**.

<p align="center"><img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/24f7ff2aca23230fdc48419fcf590ed6.svg?invert_in_darkmode&sanitize=true" align=middle width=576.5302471499999pt height=43.0227435pt/></p>

Bias is a measure of the deviation of the expected form of our models and <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>. The word "expected" means that the model is a function of the underlying data that an algorithm is trained on, which is it itself a random variable. Variance on the other hand measures the expected deviance of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/101b25d063b532a19ae95b5c5b11acd7.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=31.50689519999998pt/> from the expected fit of  <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/101b25d063b532a19ae95b5c5b11acd7.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=31.50689519999998pt/>.

It is always possible to have completely unbiased models with high variance by perfectly fitting the training data---but it would change significantly depending on the input, and so would generalize poorly (i.e., overfit). In order to lower the variance, the model must make certain generalizing *assumptions*. The more such assumptions it makes, the lower the variance---but at the cost of bias, if our assumptions turn out to be incorrect. For example, if we fit linear regression for expected true values that are not linear in the features, then that's a bad assumption that leads to bias; it does, however, decrease variance.

## Visualizing bias and variance

To demonstrate the bias-variance tradeoff, I repeatedly fit polynomial models to simulated data (normally distributed random points with mean <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/6177db6fc70d94fdb9dbe1907695fce6.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/>) as defined in the following code snippet. 

```python
import numpy as np

def f(x):
    return x ** 2

def get_sim_data(f, var, sample_size=100):
    x = np.random.uniform(size=sample_size, low=0, high=1.5)
    y = np.random.normal(size=sample_size, loc=f(x), scale=var)
    return x, y
```

Below I used 3 algorithms: <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/70b7c1e20d9aa45a257e4019135eb721.svg?invert_in_darkmode&sanitize=true" align=middle width=191.18632334999998pt height=32.51169900000002pt/> for <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7947617baa774db435d3b696f8e8fa37.svg?invert_in_darkmode&sanitize=true" align=middle width=93.09352304999999pt height=24.65753399999998pt/>, representing a biased algorithm (k=1), an unbiased and low variance algorithm (k=2), and an unbiased high-variance algorithm (k=10).

![](https://i.imgur.com/8pE41Mw.jpg)

In the above plot <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/c735bad34f0454065c4026d0906f210e.svg?invert_in_darkmode&sanitize=true" align=middle width=14.60053319999999pt height=31.50689519999998pt/> seems quite consistent with varying training data, even if it is missing the true form of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> (defined above as <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/6177db6fc70d94fdb9dbe1907695fce6.svg?invert_in_darkmode&sanitize=true" align=middle width=15.94753544999999pt height=26.76175259999998pt/>). In contrast observe that <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/713f327ecb6bda019aa1157c31153b91.svg?invert_in_darkmode&sanitize=true" align=middle width=21.153080849999988pt height=31.50689519999998pt/>, while following the trend of the data, seems to vary a large deal from simulation to simulation.

Below I repeat the above simulations 100 times, and for each model calculate <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/465a59166cae7d42cbeb469bded9a75c.svg?invert_in_darkmode&sanitize=true" align=middle width=90.40537769999999pt height=31.50689519999998pt/>, where <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/30ba872b472565d24ca62e6302d1b18e.svg?invert_in_darkmode&sanitize=true" align=middle width=44.098051799999986pt height=21.18721440000001pt/>. Notice that the center of each resulting distribution indicates the bias introduced by the algorithm and the width of distribution indicates the variance of the fit:

![](https://i.imgur.com/BzXHZ0A.png)

There are several things to notice here: for the algorithm with bias, <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/c735bad34f0454065c4026d0906f210e.svg?invert_in_darkmode&sanitize=true" align=middle width=14.60053319999999pt height=31.50689519999998pt/>, we see that the distribution of fit is not centered around 0; however, it is relatively tight. As <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> increases, bias is reduced but the dispersion of fit also increases. Additionally, notice that even for the well-conditioned algorithm, there is inherent randomness in the fitting process.

# The risks of aggregation

What happens to error with aggregated models? A naive approach to predictions at the group level is to train a patient-level model, and then for each group set the aggregate prediction to the sum of patient  predictions for every group member. The question then arises: does this procedure optimize for the group-level error? As it turns out, the answer is no!

To see this, consider the following heuristic. Let a group of size <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> consist of individuals with feature values <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/e983c99a4b986d4906499192736845a7.svg?invert_in_darkmode&sanitize=true" align=middle width=91.60032254999999pt height=24.65753399999998pt/>, and true costs <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/0d6b154f8e17e94f2a3f6b925911725a.svg?invert_in_darkmode&sanitize=true" align=middle width=88.92921014999999pt height=24.65753399999998pt/>, and further assume that costs are independent across patients. For the sake of simplicity, assume that <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/6ce73a5fe4e8b3a5797979ebac0a68fc.svg?invert_in_darkmode&sanitize=true" align=middle width=137.5245828pt height=14.15524440000002pt/>. (If naive aggregation fails with even this assumption, there is no hope in general.) Then:
<p align="center"><img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/949c178521b56b44be49bc5475e98eba.svg?invert_in_darkmode&sanitize=true" align=middle width=99.8552874pt height=16.438356pt/></p>

where <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/cd67e1c35da5bf8ff5e228c366eef8f7.svg?invert_in_darkmode&sanitize=true" align=middle width=148.3331883pt height=24.65753399999998pt/>, and the <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/e76eead5067c5c0fdf70614a08ab0c95.svg?invert_in_darkmode&sanitize=true" align=middle width=12.316403549999992pt height=14.15524440000002pt/>'s are independent identically distributed variables with mean 0.

As before, the patient-level error for <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/e626d8b8793579845f6923033fea5fcc.svg?invert_in_darkmode&sanitize=true" align=middle width=46.22128499999999pt height=22.465723500000017pt/> has a decomposition:
<p align="center"><img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/905880aaa7611da1cea5f4482d4ddc80.svg?invert_in_darkmode&sanitize=true" align=middle width=415.65585765pt height=42.429156pt/></p>
 
where <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/230cc08e43ff219cdfad277dc475e164.svg?invert_in_darkmode&sanitize=true" align=middle width=77.61882314999998pt height=26.76175259999998pt/> for all <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>.

Let's look now at the group-level error:

<p align="center"><img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/39257603a9dbd02fc49deb9cad10d5ea.svg?invert_in_darkmode&sanitize=true" align=middle width=560.5403826pt height=119.37885299999999pt/></p>

because of the linearity of expectation, and that fact that variance of a sum of independent variables is the sum of the variances of those variables.

Thus the error of the aggregation is *not* a linear multiple of the sum of the errors of the individual predictions! At the aggregated level, reducible error is multiplied by the square of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> (in fact, both bias and variance are multiplied by the square of <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>), but the reducible error is multiplied only by <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>.


# Discussion

In the above analysis we can notice, somewhat  trivially, that bias compounds. Someone without much experience in data science might think that we may underpredict a little here and overpredict a little there..but hey! it all adds up to a null sum! This brings to mind the old joke "We lose money on every sale, but make it up on volume!". That is, when you add up many small losses you get a big loss!

In healthcare, we are even more likely to compound our model error because accounts tend to be more homogenous than the general population; that is, members within accounts are more similar than members without. For example, consider a logging company (the most accident prone job in America). In this case members have higher expected cost than the general population. Since our predicted cost <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/101b25d063b532a19ae95b5c5b11acd7.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=31.50689519999998pt/> is less than the expected cost <img src="https://raw.githubusercontent.com/MatKMcC/blog_post/master/tex/7997339883ac20f551e7f35efff0a2b9.svg?invert_in_darkmode&sanitize=true" align=middle width=31.99783454999999pt height=24.65753399999998pt/>, we have introduced bias. Furthermore, since we are predicting in aggregate for every member of that account, we will be compounding our bias for each prediction! We can generalize this to any account that has characteristics that are not reflected in the general population. 


# Making Healthcare Smarter

Optimizing patient predictions for aggregate performance is an area of active research at Lumiata. Below are some strategies we employ when predicting at the group-level:

-  Optimize patient predictions for group level performance.
-  Evaluate group predictions for signs of aggregated error.
-  Build additional corrective models on aggregated predictions using group-level features.

Following this roadmap we can leverage patient-level data while avoiding the pitfalls that would accompany naive aggregation. 

If you are interested in building and scaling cool models with healthcare data, [Lumiata is hiring](https://www.lumiata.com/careers.html)!

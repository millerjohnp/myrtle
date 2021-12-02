"""Example of how to use the myrtle code."""
import numpy as np
from testbed.learning_rule import LearningRule
import torch


# Simple synthetic data (same shape as CIFAR10)
num_samples = 256
X = np.random.randn(num_samples, 32, 32, 3)
y = np.random.choice(10, size=num_samples)

# Construct the myrtlenet learning rule
rule = LearningRule(
    family='myrtlenet',
    params=dict(
        batch_size=64,
        # Specify total number of gradient steps
        # e.g. num_epochs = total_examples / len(X)
        total_examples=len(X) * 3,
    )
)
# Train the myrtle net
model = rule.learn(
    examples=X,
    labels=y,
    half_precision=torch.cuda.is_available(),
    verbose=True,
)

# Evaluate the model on the dataset
ev = model.predict(examples=X, labels=y)

predictions = ev.predictions

# Compute model accuracy
accuracy = ev.accuracy
print(f"Accuracy: {accuracy:.2f}")

# Grab the prediction vector
print("Predictions")
print(ev.predictions)


# Train another model on a potentially new dataset.
# (Don't need to create a new learning rule)
#model2 = rule.learn(
#    examples=X,
#    labels=y,
#    half_precision=torch.cuda.is_available(),
#    verbose=True,
#)

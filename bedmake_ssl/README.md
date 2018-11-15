# Self-Supervised Learning (SSL)

(These results will eventually move to a different repo.)

See `tentative_results.md` for some tentative results.

## Mean and Std

The original data from Ryan's script (with a few samples removed).

```
Just loaded: ssldata/rollout.pkl  (len: 81)
skipping 0
skipping 5
skipping 20
skipping 30
skipping 40
skipping 60
skipping 69
done loading data, train 55 & valid 18 (total 73)
numbers.shape: (3, 22425600)  (for channel mean/std)
mean(numbers): [107.04831643 102.66479564 105.59662141]
std(numbers):  [109.82162089 112.29766834 114.25086572]

But, use this for actual mean/std because we want them in [0,256) ...
mean(scaled): [0.41979732 0.40260704 0.4141044 ]
std(scaled):  [0.43067302 0.44038301 0.44804261]
```

This is the first data collection he did.

## Options

We have several options for the architecture and loss, called `model_type` in
the command line args; here `model` is taken up with the ResNet stem stuff.

All of these use the pre-trained ResNet 18-stem, where we remove the last layer
(which was a (512-2) dense layer) and replace it with a 200-dim output. Then,
for both parallel streams, we concatenate them, and feed into *two more* FC
layers of size 200 each. Then, and only then, do we get to these types:

- Type 1: we do a MSE on the location, and Cross Entropy on the angle class. For
  this we use separate branches of the network.

- Type 2: we do a MSE on the location, and Cross Entropy on the angle class (as
  in type 1). But we rely on an autoregressive architecture where we do the
  angle first, then feed the position.

- Type 3: same as type 3, except we do position first, then feed that into the
  angle.

We are not dealing with length for now.

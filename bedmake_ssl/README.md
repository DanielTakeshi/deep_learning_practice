# Self-Supervised Learning (SSL)

I need to adjust the transforms to deal with pairs of images and
with the action parameterization we use.

With the appropriate data, `python prepare_data.py` results in:

```
Just loaded: ssldata/rollout.pkl  (len: 81)
skipping 0
skipping 20
skipping 40
skipping 60
done loading data, train 57 & valid 19 (total 76)
numbers.shape: (3, 23347200)  (for channel mean/std)
mean(numbers): [106.96605379 102.65406233 105.63056662]
std(numbers):  [109.67399139 112.08692714 114.09877422]

But, use this for actual mean/std because we want them in [0,256) ...
mean(scaled): [0.41947472 0.40256495 0.41423752]
std(scaled):  [0.43009408 0.43955658 0.44744617]
```

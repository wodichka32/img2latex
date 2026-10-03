## Test metrics

| model     |   bleu |   edit_score |   exact_match |    n |
|:----------|-------:|-------------:|--------------:|-----:|
| hybrid_tf | 0.9332 |       0.9524 |        0.5924 | 6806 |
| pix2tex   | 0.8799 |       0.9026 |        0.408  | 6806 |

## BLEU by formula length

| length         |   hybrid_tf |   pix2tex |
|:---------------|------------:|----------:|
| long (>70)     |       0.909 |     0.846 |
| medium (31-70) |       0.939 |     0.889 |
| short (<=30)   |       0.921 |     0.862 |

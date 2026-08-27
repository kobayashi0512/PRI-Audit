| Audit | Dataset | Primary estimate | Uncertainty / diagnostic | Interpretation |
|---|---|---|---|---|
| Raw reconstruction | GSE151177; GSE173706; GSE206391 | 3/3 count matrices exact | metadata and canonical count contents exact | formal v21 analyses start from verified public inputs |
| Random cell split | GSE151177 | median AUROC 0.999 | 2.5–97.5% range 0.998–0.999; 100% test rows from seen sources | invalid within-source measurement-interpolation estimand |
| Random spot split | GSE206391 | median AUROC 0.970 | 2.5–97.5% range 0.966–0.974; 100% test rows from seen sources | invalid within-source measurement-interpolation estimand |
| Global vs fully nested neighbourhood | GSE151177 | K=8 gap +0.071 | 95% CI +0.000 to +0.222; K=16 +0.000, K=32 -0.054 | global–strict difference is resolution-sensitive, not universally positive |
| Mismatched-reference sensitivity | GSE206391 | apparent PCA gain +0.082 | 95% CI +0.002 to +0.219; unmatched 16-HVG baseline | a weakened comparator can manufacture an apparent gain |
| Coordinate shuffle | GSE206391 | spatial graph observed 0.870 | 500 within-sample shuffles; null mean 0.870; tail P=0.563; combined observed 0.875, P=0.677 | no demonstrated coordinate-specific patient-level increment for this graph |
| Independent high-resolution boundary | GSE202011 | expression = PCA = AUROC 1.000 | five paired patients; increment 0.000 (0.000, 0.000) | saturated expression/PCA boundary replication; no coordinate archive |

# Maneuver-demand proxy

This observable is an intentionally simple geometric policy proxy. An encounter is actionable when
the fragment characteristic size is at least the assumed trackability threshold and the miss
distance is no greater than the assumed action threshold. Multiple numerical records for the same
debris-target TCA are clustered within an explicit time tolerance and the smallest-distance record
is retained.

The outputs are the retained encounter records, affected target indices, per-target event counts,
and event rate. The probability of at least one event uses the homogeneous Poisson-rate proxy
`1 - exp(-rate * duration)` and must be labelled as a model assumption, not an empirical frequency.

This calculation does not use covariance, calculate operational collision probability, optimize
delta-v, insert maneuvers, or repropagate trajectories. It estimates gross warning-policy demand,
not actual operator decisions.

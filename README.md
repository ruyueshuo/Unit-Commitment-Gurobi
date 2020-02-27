# Unit-Commitment_Gurobi
Python version of unit commitment solver using gurobi-python api.

This example formulates and solves the following simple UC model:
 minimize
       objective: min f(x)
 subject to
       constraints: AX<=b
 with
       variables: x0,x1,...,xn

This is a deterministic model solver and just a baseline.
If you want a stochastic programme model solver, try scenario generation method.

In this model, the variables are:
on-off status of thermal unit, power output of thermal unit, hydro unit, wind farm, solar station and battery station.
The following constraints are included:
Power balance constraints, upper and lower outputs constraints, ramp constraints, reservoir volume constraints,
on-off minimum time constraints, and reserve capacity constraints.
The objectives are:
Operation cost of thermal units and punishment for abandoning renewable energy.

There are three simplifications:
First one is the hydro power output has no relationship with reservoir volume or water-head height,
and the upcoming and up-flowing water volumes are the same or linear relationship during the whole 24h.
You can also set the reservoir volumes by yourself.
Second one is that the range of battery station power output is defined as [-Ps,Pu].
But if you consider pumped storage power station rather than battery station, the output range should be {-Ps,[0,Pu]}.
Third one is the start up cost of thermal unit is not concluded in the objective.


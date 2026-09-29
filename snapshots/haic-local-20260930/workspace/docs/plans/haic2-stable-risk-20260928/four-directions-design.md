# Four stable-base directions: TRAIN activation

Common control is the selected full-road-guard with its exact checkpoint. Four independent one-change directions use that stable actor: (1) asphalt-clearance and bend-growth gated gas floor, (2) short throttle pulse only after a bend opens, (3) earlier brake when a centered obstacle is visibly approaching, and (4) modest outward steering at mild bend entry. None uses map geometry or simulator state at inference. Each has one fixed setting; there is no threshold sweep.

Run consumed TRAIN 1/43 and 2/102, five profiles × two cells, 2,000 decisions each, Linux CPU, 1,800-second registered limit. Require action changes and no invalid actions for each direction. Reject a direction that loses a completion the control earns. These cells establish only mechanism activation; a later fresh tune comparison needs its own plan and split.

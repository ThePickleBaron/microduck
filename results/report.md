# Deployment-readiness results

## Site: held_out

| Condition | Scenario | Falls / 10 min | Speed error (m/s) | Yaw-rate error (rad/s) | Stalled (%) |
| --- | --- | --- | --- | --- | --- |
| c1 | slippery_floor | 0.0 | 0.141 | 0.118 | 21 |
| c1 | heavy_payload | 0.0 | 0.131 | 0.117 | 0 |
| c1 | rough_floor | 0.0 | 0.136 | 0.171 | 9 |
| c1 | worn_gears | 0.0 | 0.127 | 0.157 | 0 |
| c1 | low_battery | 0.0 | 0.144 | 0.155 | 5 |
| c1 | slow_link | 0.0 | 0.117 | 0.120 | 0 |
| c1 | bumped | 3.3 | 0.133 | 0.186 | 5 |
| c1 | site_combined | 0.0 | 0.137 | 0.161 | 9 |

## Site: nominal

| Condition | Scenario | Falls / 10 min | Speed error (m/s) | Yaw-rate error (rad/s) | Stalled (%) |
| --- | --- | --- | --- | --- | --- |
| c1 | nominal | 0.0 | 0.127 | 0.142 | 1 |

## Readiness gap (held-out vs nominal)

Positive = worse on the held-out site.

| Condition | Scenario | Tracking gap (%) | Extra falls / 10 min | Extra stall (pts) |
| --- | --- | --- | --- | --- |
| c1 | slippery_floor | +12 | +0.0 | +20 |
| c1 | heavy_payload | +3 | +0.0 | -1 |
| c1 | rough_floor | +7 | +0.0 | +8 |
| c1 | worn_gears | +1 | +0.0 | -1 |
| c1 | low_battery | +14 | +0.0 | +4 |
| c1 | slow_link | -8 | +0.0 | -1 |
| c1 | bumped | +5 | +3.3 | +4 |
| c1 | site_combined | +8 | +0.0 | +8 |

### Mean gap per condition

| Condition | Mean tracking gap (%) | Mean extra falls / 10 min |
| --- | --- | --- |
| c1 | +5 | +0.4 |

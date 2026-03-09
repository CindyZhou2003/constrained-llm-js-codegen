# Code generation and test results
## MultiPL-E
| Dataset | Temperature | Pass@k | Estimate | NumProblems | MinCompletions | MaxCompletions |
|---------|--------|--------|----------|-------------|----------------|----------------|
| mbpp-js-microsoft_phi_2-0.0-syncode | 0.0 |1| 0.38287153652392947| 397|1|1 |
| mbpp-js-microsoft_phi_2-0.0-unconstrained | 0.0 |1| 0.05037783375314862|397|1|1|
| mbpp-js-microsoft_phi_2-0.2-syncode | 0.2 | 1 | 0.3702770780856423|397|1|1|
| mbpp-js-microsoft_phi_2-0.2-unconstrained| 0.2| 1 | 0.05289672544080604|397|1|1|


| Mode | Temperature | Score |
| unconstrained | 0.0 | 0.05037783375314862 |
| unconstrained | 0.2 | 0.05289672544080604 |
| syncode | 0.0 | 0.38287153652392947 |
| syncode | 0.2 | 0.3702770780856423 |
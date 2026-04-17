# Patience Sensitivity Study

Configuration:
- 5,000 custom proton-only images
- Optimal proton-only filter configuration (Table 1, converted to metres)
- MAX_EPOCHS = 10,000

| Patience | Val Loss | Stopped at Epoch |
|----------|----------|-----------------|
| 5        | 0.0077   | 39              |
| 10       | 0.0064   | 75              |
| 15       | 0.0043   | 168             |
| 20       | 0.0042   | 203             |
| 25       | 0.0044   | 151             |
| 30       | 0.0042   | 216             |

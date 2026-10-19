# heart-trabeculae

Testing the claim that trabeculae carneae keep ventricular walls from sticking together by suction, framed as a squeeze film (Stefan adhesion) problem.

* `squeeze_film/reynolds.py`: Tier 1 Reynolds lubrication solver for smooth, grooved and fractal endocardial surfaces
* `squeeze_film/experiments.py`: physiological sweeps and figures
* `tests/`: verification against analytic solutions
* `RESULTS.md`: findings, verdict and limitations

```
pip install -r requirements.txt
pytest
python -m squeeze_film.experiments
```

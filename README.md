# heart-trabeculae

Testing the claim that trabeculae carneae keep ventricular walls from sticking together by suction, framed as a squeeze film (Stefan adhesion) problem.

* `squeeze_film/reynolds.py`: Tier 1 Reynolds lubrication solver for smooth, grooved and fractal endocardial surfaces
* `squeeze_film/experiments.py`: physiological sweeps and figures
* `tests/`: verification against analytic solutions
* `RESULTS.md`: Tier 1 findings, verdict and limitations
* `ventricle/`: Tier 2 2D long axis ventricle (penalized Navier Stokes, prescribed kinematics, valves, Tier 1 film closure)
* `RESULTS_TIER2.md`: Tier 2 findings, verdict and limitations
* `apex3d/`: Tier 3 3D resolved apposed apical walls (coupled penalization solve, Brinkman walls, two way wall coupling) and the calibrated reduced film model
* `RESULTS_TIER3.md`: Tier 3 findings, verdict and limitations

```
pip install -r requirements.txt
pytest
python -m squeeze_film.experiments
for k in trab smooth_crest smooth_vol; do python -m ventricle.run --kind $k --cycles 3; done
python -m ventricle.analyze
bash results/tier3/lane1.sh & bash results/tier3/lane2.sh
python -m apex3d.analyze
```

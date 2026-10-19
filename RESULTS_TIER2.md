# Tier 2 results: a 2D ventricle with trabeculae, valves and a subgrid film

## Question

Tier 1 showed that trabecular grooving removes the squeeze film suction between apposed apical walls, but a smooth wall holding the same blood volume did even better. So the real question moved to the scale of the whole ventricle: **does a trabeculated ventricle that empties to near apposition fill and wash out better than a smooth ventricle, either one emptying to the same crest gap or one holding the same volumes?** Tier 2 answers that in 2D with a full cardiac cycle.

## Model

**Flow solver** (`ventricle/solver.py`). Incompressible Navier Stokes is solved pseudo spectrally on a periodic 75 by 125 mm box at 0.25 mm resolution (300 by 500 cells), with $\mu = 3.5$ mPa·s and $\rho = 1060$ kg/m³. Myocardium and valve leaflets enter through Brinkman volume penalization, which drives the fluid velocity toward the prescribed solid velocity inside bodies. Advection uses second order Adams Bashforth with 2/3 dealiasing, viscosity an exact integrating factor, and the projection is exact in Fourier space. One subtlety mattered. A single penalize then project pass lets thin bodies leak fluid at roughly $\Delta t \, |\nabla p| / \rho$, which cut the resolved squeeze pressure to 27% of the lubrication value in a validation case. Repeating the penalize and project pair and keeping $\Delta t$ at or below $10^{-4}$ s fixed it. Old blood is tracked by a bounded semi Lagrangian tracer.

**Ventricle** (`ventricle/geometry.py`). This is a long axis slice whose crest surface morphs from an elliptical end diastolic shape (85 mm long, 44 mm wide at the annulus) to an end systolic shape whose apical quarter collapses into a slit. The cycle lasts 0.8 s, with these phases:

* ejection: 300 ms
* isovolumic relaxation: 80 ms
* E wave: 170 ms, carrying 75% of the filling
* diastasis: 110 ms
* A wave: 120 ms
* isovolumic contraction: 20 ms

A mitral orifice (28 mm) and an aortic orifice (14.5 mm) sit in the base. Their leaflets open and close on schedule, so inflow and outflow take different paths. Compact wall thickness follows from conserving myocardial area, which holds exactly in the analytic geometry and to within 0.4% on the grid. Three endocardia share the same crest motion:

| Case | Endocardium | End diastolic / end systolic cavity (mm²) | EF |
|---|---|---|---|
| Trabeculated | compact wall 2 mm behind the crest, 1 mm ridges at 2 mm pitch over the apical 60% | 3047 / 1404 | 0.54 |
| Smooth at crest | smooth wall on the crest surface (same apical gap, smaller cavity) | 2937 / 1314 | 0.55 |
| Smooth, volume matched | smooth wall 1 mm behind the crest (same cavity areas as trabeculated) | 3048 / 1405 | 0.54 |

**Subgrid film closure** (`ventricle/closure.py`). The grid resolves an apical crest gap of 0.75 mm (3 cells), not the physiological 0.1 to 0.5 mm. The resolved kinematics supply the opening rate of the slit, and Tier 1's Reynolds solver supplies the film suction for that gap on a matching circular patch of radius 8.75 mm. That is a groove mesh for the trabeculated wall and a smooth disk for the controls, with the volume matched wall sitting 2 mm further apart. The coupling is one way.

Each case runs three cycles (about 25,000 steps and 35 minutes each). Metrics come from the third cycle, and washout covers all three.

## Verification (19 tests for Tier 2, `pytest`)

**Solver:**
* Taylor Green decay matches the exact solution to $10^{-6}$, with a divergence of $10^{-8}$ and the correct pressure.
* Penalized Poiseuille flow matches within 1.5%, and Couette flow driven by a moving penalized wall within 4%.
* The tracer stays bounded and translates exactly.
* Cross tier check: two plates separating in the Navier Stokes solver reproduce the 1D lubrication pressure that Tier 1 is built on, within 15%.

**Geometry:**
* Myocardial area is conserved to $10^{-10}$.
* Volume matching holds within 0.2%.
* The grid cavity area tracks the analytic area within 2%.
* The prescribed wall velocity reproduces $dA/dt$ within 8%.
* The film closure recovers the Stefan solution within 3%.

**In the production runs:**
* The measured valve plane flux matches $dA/dt$ within 6 to 7%.
* The divergence in the cavity stays at $10^{-14}$.
* Between 0.5 and 0.25 mm grids, washout and most pressures change by under 5%. The trabeculated E wave gradient is the exception: it moves from 3.3 to 2.8 mmHg.

## Results

| Third cycle | Trabeculated | Smooth at crest | Smooth, volume matched |
|---|---|---|---|
| Peak E wave base to apex gradient (mmHg) | 2.8 | 2.3 | 2.0 |
| Lowest cavity pressure in early E (mmHg) | −17.6 | −7.6 | −4.3 |
| Viscous dissipation per cycle (mJ/m) | 27.2 | 22.0 | 16.6 |
| Net diastolic endocardial work, resolved (mJ/m) | −34.7 | −33.2 | −37.4 |
| Film suction peak, crest gap 0.2 mm (mmHg) | 0.21 | 6.3 | 0.016 |
| Film suction peak, crest gap 0.1 mm (mmHg) | 0.23 | 35.8 | 0.017 |
| Film energy, 0.2 mm (share of diastolic budget) | 1.3% | 6.6% | 0.2% |
| Film energy, 0.1 mm (share of diastolic budget) | 1.4% | 18.9% | 0.2% |
| Initial blood left after 3 beats | 17.2% | 17.3% | 20.7% |
| Old blood in the near wall layer at end diastole | 56% | none (no layer) | 63% |

**1. The resolved filling is physiological.** All three ventricles develop an early diastolic base to apex gradient of 2.0 to 2.8 mmHg (Figure 9). That's in the range usually reported for early diastolic intraventricular pressure gradients, and it emerges from the flow rather than being imposed. The trabeculated ventricle has the largest gradient and the highest dissipation (27 versus 17 mJ/m against the volume matched control). That matches the direction Sacco et al. found in rigid wall CFD ([doi:10.3389/fphys.2018.00458](https://doi.org/10.3389/fphys.2018.00458)): trabeculae raise the intraventricular pressure drop.

**2. Against a smooth wall emptying to the same gap, trabeculae win clearly.** The smooth at crest ventricle pays a real film penalty at the moment the mitral valve opens (Figure 11). With an end systolic gap of 0.2 mm, peak apical film suction reaches 6.3 mmHg about 6 ms after mitral opening, more than twice the resolved filling gradient. At 0.1 mm it reaches 36 mmHg and costs about 19% of the whole diastolic energy budget. The trabeculated wall caps this at 0.2 mmHg at every gap tested, because the groove network drains the film. Washout is essentially identical (17.2% versus 17.3% residual after three beats). So, compared with a smooth heart that empties to the same gap, trabeculae remove an adhesion penalty without costing washout.

**3. Against a smooth wall holding the same volumes, it is a real trade off.**
* **Energy:** the volume matched ventricle has no film problem (its walls never get closer than about 2.5 mm), the lowest dissipation, and the most favorable diastolic energy balance, about 3 mJ/m (8% of the budget) better than the trabeculated case.
* **Washout:** the trabeculated ventricle clears old blood better. After three beats 17.2% of the original blood remains, versus 20.7% for the volume matched control. Both beat perfect mixing after one beat (44% versus 46%) but fall behind it by the third beat (9.8%), the usual signature of an apex that never fully washes out. The volume matched control falls behind more. The recesses between trabeculae also hold less old blood at end diastole (56%) than the smooth control's equivalent near wall layer (63%), even though the flow inside the recesses is slower (29 versus 51 mm/s mean). So in this model the trabeculae don't create the stasis that the noncompaction literature worries about. If anything, the ridges scour the near wall layer.

**4. A 2D artifact to keep in view.** The deepest resolved suction (−17.6 mmHg) sits in the trabeculated apical slit during early E. In a 2D slice, opposing trabecular tips form dams across the slit, so blood can only enter through the 0.75 mm gaps between them. A real 3D trabecular network drains along the grooves, which is exactly what the Tier 1 closure represents. The resolved apical suction in the trabeculated case is therefore a worst case. It is reported but not used as evidence against trabeculae.

## Verdict

Tier 2 sharpens the hypothesis rather than simply confirming it. Trabeculae let the ventricle empty to near apposition without the viscous adhesion penalty a smooth wall would pay at the same gap. That penalty is not negligible: 6 to 36 mmHg and up to a fifth of the diastolic energy budget once the crest gap falls to 0.1 to 0.2 mm. If instead you hold the blood volumes fixed, a smooth ventricle fills slightly more cheaply, while the trabeculated ventricle washes out better, with about 17% less residual blood after three beats. So the most defensible statement is: **trabeculae trade a modest increase in filling dissipation for near apposition without adhesion and better washout.** That fits with Meyer et al. linking trabecular complexity to cardiac performance ([doi:10.1038/s41586-020-2635-8](https://doi.org/10.1038/s41586-020-2635-8)). These are 2D numbers, though, and the trade off is exactly the kind of result a 3D model could flip.

## Limitations

1. **2D slice.** Trabeculae are ridges cut in cross section, so the drainage network, the 3D vortex ring and out of plane washout are missing. The trabeculated apical slit is the clearest casualty (point 4 above).
2. **Prescribed kinematics.** Wall motion is imposed, so the model measures the cost of filling rather than letting filling respond to pressure. There is no atrial pressure waveform, compliance or fluid structure interaction, which is Tier 3.
3. **One way film closure.** The subgrid film suction does not feed back into the resolved flow or slow the wall.
4. **Afterload and sponge.** Outflow passes through a weak sponge in the reservoirs, so systolic pressures and systolic work are not physiological and are not reported. The sponge acts per step, so it also depends on time step. Diastolic metrics are referenced to the atrial channel below the sponge and are unaffected.
5. **Leakage and resolution.** Volume penalization leaves the walls slightly permeable (valve plane flux error 6 to 7%). The trabeculae are 4 cells wide and the resolved slit is 3 cells. Faint spectral ringing near the sharp trabecular features is visible in the vorticity fields.
6. **Single geometry.** There is one cycle shape and one trabecular pattern, with no sweep over trabecular complexity or end systolic gap in the resolved model, and no shear thinning blood model.

## Reproduce

```
pip install -r requirements.txt
pytest -m "not slow"          # fast suite; drop the filter for the cross tier plate test
for k in trab smooth_crest smooth_vol; do python -m ventricle.run --kind $k --cycles 3; done
for k in trab smooth_crest smooth_vol; do python -m ventricle.run --kind $k --cycles 3 --dx 0.5e-3 --dt-max 2e-4 --tag _coarse; done
python -m ventricle.analyze   # writes figures/fig6 to fig11 and results/tier2_results.json
```

Each production run takes about 35 minutes on one core, and the three can run in parallel.

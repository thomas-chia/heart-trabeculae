# Tier 1 results: squeeze film suction between apposed endocardial walls

## Question

Textbooks occasionally claim that trabeculae carneae keep the ventricular walls from "sticking" together by suction. Stated as physics, the most defensible version is a squeeze film (Stefan adhesion) problem: when opposing walls near the apex sit a fraction of a millimeter apart at end systole, early diastole has to pull them apart, and viscous blood has to be drawn into a very thin gap. Tier 1 asks one falsifiable question: **at physiologic separation speeds and plausible end systolic gaps, does realistic trabecular grooving meaningfully lower the pressure needed to separate the walls?**

## Model

Two rigid walls separate at a uniform speed $\dot h$. One wall carries a recess pattern $D(x,y)$, so the local gap is $h = h_0 + D$, with $h_0$ the crest gap. In the thin gap limit the pressure obeys the Reynolds lubrication equation

$$\nabla \cdot \left( \frac{h^3}{12\mu} \nabla p \right) = \dot h$$

on a circular apposed patch of radius $R$, with $p = 0$ (cavity pressure) on the rim. It is solved by finite volumes (harmonic mean face conductances, cut cell rim) in `squeeze_film/reynolds.py`. Grooves deeper than they are wide are treated as rectangular Poiseuille ducts rather than as wide slots, which otherwise overstates their drainage by up to roughly 5 fold (Figure 3, dashed line).

Baseline values are order of magnitude choices, not measurements: $\mu = 3.5$ mPa·s, $R = 10$ mm, $\dot h = 5$ cm/s (twice a 2.5 cm/s radial endocardial velocity), trabecular pitch 2 mm, groove width 1 mm, recess depth 2 mm. The fractal case nests three levels (4, 2, 1 mm pitch) with width and depth halving at each level. A volume matched smooth control places a flat wall at the mesh's mean gap, so it holds the same blood volume as the grooved film.

## Verification (12 tests, `pytest`)

The solver reproduces the exact Stefan solution for a smooth disk to under 1% in peak pressure and 2% in force, obeys the expected linearity in $\dot h$ and $\mu$ and the exact $h^{-3}$ scaling, converges with grid refinement (fractal force changes 0.5% between $n = 400$ and $600$), recovers the 1D strip limit between perfectly draining grooves within 5%, and reproduces the analytic force driven opening time for a smooth disk within 3%. Grooves never increase suction at any tested gap.

## Results

**1. A smooth wall really would stick, but only at small gaps.** Peak suction scales as $h_0^{-3}$: about 3 mmHg at 0.5 mm, 49 mmHg at 0.2 mm and 394 mmHg at 0.1 mm (Figure 1). The crossover with a typical early diastolic intraventricular pressure gradient of roughly 2.5 mmHg sits near $h_0 \approx 0.5$ mm. Above roughly 1 mm the effect is negligible for every geometry.

**2. Trabecular grooving removes almost all of it.**

| Crest gap | Smooth peak (mmHg) | Mesh peak (mmHg) | Force reduction, parallel | Force reduction, mesh | Force reduction, fractal |
|---|---|---|---|---|---|
| 0.1 mm | 394 | 1.6 | 172x | 533x | 376x |
| 0.2 mm | 49 | 0.57 | 45x | 106x | 56x |
| 0.5 mm | 3.2 | 0.34 | 5x | 9x | 5x |
| 1.0 mm | 0.39 | 0.21 | 1.5x | 1.9x | 1.5x |

With grooves, peak suction stays at or below about 1 mmHg down to 0.2 mm gaps, i.e. below the diastolic driving gradient. The reduction is two to twenty times larger than my earlier $(R/\lambda)^2 \approx 25$ back of the envelope estimate at 0.2 mm, because a groove network splits the film into small islands rather than strips (Figure 5).

**3. There is an optimal pitch, and it lands near trabecular scale.** At $h_0 = 0.2$ mm with 2 mm deep recesses, the force reduction peaks at a pitch of about 3 to 4 mm (about 165x for the mesh) and falls off on both sides (Figure 2). Coarse grooves leave large islands; fine grooves become narrow ducts that cannot carry the inflow. This is the most interesting result, since the optimum is set by the depth to width ratio, and real trabeculae (1 to 3 mm) sit on the rising shoulder of it.

**4. Depth is the control knob.** Grooves only begin to drain once the recess depth exceeds the crest gap, and they need $D/h_0 \gtrsim 5$ to give a 30x relief (Figure 3). Shallow surface roughness is useless here; a deep, interconnected recess network is what works.

**5. Opening time.** Under a constant opening stress from elastic recoil, a smooth wall needs about 49 ms at 1 mmHg (98 ms at 0.5 mmHg) to go from 0.2 mm to 2 mm, a large fraction of the roughly 100 ms early rapid filling window. Grooved walls open in 4 to 6 ms under the same stress (Figure 4). From 0.1 mm the smooth wall needs about 200 ms at 1 mmHg, which would genuinely delay filling.

**6. The critical caveat: the volume matched control.** A smooth wall holding the same blood volume as the grooved film (mean gap about 1.5 mm higher) has lower suction than any grooved surface. So trabeculae are not the most volume efficient way to avoid suction; simply not emptying as far would do better. The hypothesis survives only in a reframed form: **trabeculae let the crests come into near contact (low end systolic volume, high ejection) while keeping a drainage network open, so the walls do not pay a viscous adhesion penalty for emptying that far.** Whether that trade beats a smooth wall with a slightly larger end systolic volume is a question Tier 2 has to answer, because it depends on stroke volume and washout, not just the film.

## Verdict

The suction claim is physically real for smooth walls at end systolic gaps below about 0.5 mm, and realistic trabecular grooving eliminates it by one to two orders of magnitude, with an optimal pitch close to anatomical scale. That passes the Tier 1 gate. However, the advantage only exists relative to a smooth wall emptying to the same crest gap, so the claim should be stated as "trabeculae permit near apposition without adhesion," not "trabeculae prevent suction." This is consistent in spirit with Meyer et al., who linked trabecular fractal complexity to cardiac performance in 18,096 UK Biobank participants and biomechanical simulations ([Nature 2020, doi:10.1038/s41586-020-2635-8](https://doi.org/10.1038/s41586-020-2635-8)). It also has to be reconciled with Sacco et al., whose rigid wall CFD found trabeculae *increase* the intraventricular pressure drop and reduce wall shear stress ([Front Physiol 2018, doi:10.3389/fphys.2018.00458](https://doi.org/10.3389/fphys.2018.00458)); their models did not include near apposition of walls, which is exactly the regime where this effect lives. Both references were retrieved from PubMed.

## Limitations

1. **Inertia is not negligible.** The squeeze Reynolds number $\rho \dot h h/\mu$ is about 1.5 to 3 at the baseline 0.1 to 0.2 mm gaps and reaches 7.6 at 0.5 mm. Lubrication theory is formally valid only for values much less than 1, so the absolute pressures are order of magnitude; the smooth versus grooved ratios should be more robust because inertia scales similarly in both. Tier 2 resolves this.
2. Rigid walls with a uniform $\dot h$; real endocardium deforms, and compliance would itself relieve suction.
3. Idealized square groove networks; real trabeculae are irregular, tapered and partially bridged.
4. Grooves drain only laterally to the patch rim; real recesses also open into the cavity in the third dimension, so drainage here is conservative.
5. The end systolic crest gap, the apposed patch size and apical wall separation speed are not measured inputs. The whole conclusion hinges on whether real apical walls approach within about 0.5 mm, which needs high resolution end systolic imaging to settle.

## Reproduce

```
pip install -r requirements.txt
pytest
python -m squeeze_film.experiments   # about 4 minutes; writes figures/ and results/results.json
```

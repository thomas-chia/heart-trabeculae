# Tier 3 results: 3D resolved apposed walls with two way fluid structure coupling

## Scope, and how it differs from the original plan

My original Tier 3 was a CT derived 3D ventricle with full fluid structure interaction. That isn't feasible here. There is no imaging data at trabecular resolution in reach (the public four chamber meshes are segmented from clinical CT and smooth the trabeculae away), and the sandbox has 4 CPU cores. So I built the part of Tier 3 that answers what Tiers 1 and 2 could not, at the apex where the hypothesis actually lives:

1. **A 3D resolved apposed patch.** Two walls are pulled apart in open blood, one of them carrying a connected groove network. This is the 3D drainage topology a 2D slice cannot represent, with full Navier Stokes inertia.
2. **Two way fluid structure coupling.** Each wall is a rigid body with myocardial mass, driven by a recoil stress and held back by the fluid force the simulation computes.
3. **A reduced film model calibrated on those runs.** It is Tier 1 plus film added mass, a fitted inertia term and wall dynamics. The model is validated against the coupled runs and then carried to the physiological gaps (0.1 to 0.5 mm) and patch size (8.75 mm) that no grid here can resolve. That turns Tier 2's prescribed apical motion into a pressure driven one.

## Model

**Solver** (`apex3d/solver3d.py`). The solver is 3D pseudo spectral incompressible Navier Stokes on a triply periodic 16 by 16 by 9.6 mm box at 0.1 mm (160 by 160 by 96 cells), using the same scheme as Tier 2. Two changes were needed to make resolved squeeze films trustworthy. Both are documented in the code and pinned by tests:

* **Coupled penalization and pressure solve.** Alternating penalize and project passes, which I used in Tier 2, left rigid walls moving about 12% slower than prescribed, whatever the time step or number of passes. Solving $\nabla \cdot \left[ \frac{1}{1+\lambda} \nabla \phi \right] = \nabla \cdot b$ by conjugate gradients, with the FFT Laplacian as preconditioner and a warm start, brings the walls to 99.8 to 99.9% of their prescribed speed. Seepage through them falls to 0.1 to 0.3% of the film's volume growth, in about 15 to 30 iterations per step.
* **Viscous scaled Brinkman walls.** A penalized wall acts as no slip at the first grid node where the penalty dominates viscous diffusion, not at the geometric surface. With a one sided ramp that was a full 0.85 cells off, and since film force scales as $h^{-3}$, that alone was worth a factor of about two. Scaling the penalty with the per step viscous rate, $\lambda = (\nu \Delta t / \Delta x^2)\,\chi\,K^{2\chi-1}$, over a 3 cell ramp recentred by 0.39 cells puts the effective wall on the surface within ±0.1 cell at any sub cell position and any time step. I measured that on a penalized channel.

**Patch** (`apex3d/patch.py`). Two plates of 12 by 12 mm, with a crest gap opening from 0.6 to 1.8 mm. They are either smooth, or the top plate carries an orthogonal groove mesh (2 mm pitch, 1 mm wide, 1.5 mm deep). The film force is read from the gap pressure against the surrounding blood. That leaves out the artificial load from blood pushed around the plates' backs, which in a heart are tissue. There are two drives:

* **Prescribed motion:** the opening speed ramps smoothly to 8, 25 or 80 mm/s, giving squeeze Reynolds numbers of 2 to 33.
* **Two way coupled:** recoil stress of 0.5 or 2 mmHg, rising with a 3 ms time constant, on walls of 10.6 kg/m², about 10 mm of myocardium. The partitioned scheme carries an added mass correction, so it stays stable when the film's added mass rivals the wall's.

A residual immersed boundary force oscillation repeats every two cells of wall travel, so forces are averaged over that window.

**Reduced model** (`apex3d/reduced.py`).

$$F = -C(h)\,\dot h\,(r_0 + \kappa\,\mathrm{Re}_h) - M_a(h)\,\ddot h, \qquad \left(\tfrac{1}{2} m_w A + M_a\right)\ddot h = \sigma A - C\,\dot h\,(r_0 + \kappa\,\mathrm{Re}_h)$$

* $C(h)$ is the Tier 1 viscous resistance.
* $M_a(h)$ is the film's added mass. It comes from the inertial form of the Reynolds equation, $\nabla \cdot (h \nabla p) = \rho \ddot h$, which I added to the Tier 1 solver. It reproduces the exact disk added mass, $\pi \rho R^4 / 8h$, to 0.01%.
* $r_0$ and $\kappa$ are fitted to the prescribed motion runs, and $\kappa$ is not extrapolated past its calibrated Reynolds range.

## Verification (13 tests for Tier 3, 44 in total, `pytest`)

**Solver:**
* The ABC Beltrami flow decays exactly to $10^{-10}$.
* The Brinkman wall lands within 0.15 cells of the surface at four sub cell positions.
* The coupled solve keeps a translating slab within 1% of its speed, where the uncoupled scheme lags.

**Tier 1 additions:**
* The square patch Tier 1 force matches the exact Fourier series within 0.2%.
* The inertial kernel gives the disk added mass within 0.5%.
* Grooves cut viscous resistance more than added mass, as they should, since drainage scales as $h^3$ and inertial flow as $h$.

**Reduced model and patch:**
* The reduced model recovers the quasi static Tier 1 opening time within 1%, and ballistic opening exactly without a film.
* The patch masks place the walls at the crest gap.

**Slow cross tier checks:**
* Resolved smooth plates at low Reynolds number match Tier 1 within 12%.
* The Tier 2 plate test still passes.

**Resolution:** at 25 mm/s the windowed resolved to lubrication ratio is 0.88, 0.85 and 0.89 (smooth) and 0.67, 0.72 and 0.70 (mesh) on the 0.2, 0.15 and 0.1 mm grids.

## Results

**1. Tier 1's lubrication holds for a smooth wall at low Reynolds number. Inertia then cuts the suction.** For the smooth patch, the resolved film force (added mass removed) is $1.14 - 0.026\,\mathrm{Re}_h$ times the Tier 1 value (rms scatter 0.08, $\mathrm{Re}_h$ 2 to 33; Figure 12). Near $\mathrm{Re} = 2$ the ratio is about 1.07. By $\mathrm{Re} = 33$, convective inertia has cut the viscous suction to between a fifth and a quarter. Tier 1's caveat about inertia was justified, and its sign is now known: at these Reynolds numbers, inertia relieves suction rather than adding to it, apart from the added mass, which resists acceleration.

**2. The real 3D groove network drains better than Tier 1 assumed.** For the groove mesh the ratio is $0.79 - 0.007\,\mathrm{Re}_h$ (rms 0.12, the noisiest fit). The connected network carries about 20% more flow than Tier 1's duct corrected Reynolds model credits it with. Tier 1 was therefore conservative for trabeculae. This is also the cleanest answer to the Tier 2 artifact: in 3D the trabecular recesses drain, and the dammed 2D slit is not what happens.

**3. Wall and film inertia, not viscosity, set the opening time once the walls are pulled by a recoil stress.** Against the two way coupled runs, the calibrated reduced model predicts the time to reach 50% and 90% of the opening within 6 to 8% in all four cases (Figure 13). Quasi static Tier 1, a viscous film with no wall mass and no added mass, opens 72 to 84% too fast. Notably, Tier 1 plus wall mass plus film added mass, without the fitted corrections, does almost as well (5 to 7%). Inertia dominates these openings, so the film's viscous details matter less here than its mass.

**4. In the physiological regime, a smooth wall at the crest cannot open on time. A trabeculated wall can.** Here is the reduced model on the Tier 2 apical patch (radius 8.75 mm, 10.6 kg/m² walls, recoil rising over 20 ms). The table shows the time to open from the end systolic gap to 2 mm, in ms after mitral opening (Figure 14):

| Recoil, end systolic gap | No film (walls only) | Trabeculated | Smooth at crest | Smooth, volume matched | Tier 2 prescribed kinematics |
|---|---|---|---|---|---|
| 1 mmHg, 0.1 mm | 22.7 | 32.1 | 211.4 | 28.2 | 38.8 |
| 1 mmHg, 0.2 mm | 22.3 | 30.8 | 81.3 | 27.5 | 37.8 |
| 2 mmHg, 0.1 mm | 17.7 | 24.5 | 120.7 | 21.9 | 38.8 |
| 2 mmHg, 0.2 mm | 17.4 | 23.6 | 53.6 | 21.4 | 37.8 |
| 2 mmHg, 0.5 mm | 16.3 | 21.2 | 28.8 | 19.7 | 34.3 |
| 5 mmHg, 0.2 mm | 12.6 | 16.8 | 33.1 | 15.4 | 37.8 |

* **Trabeculated:** the film adds 3 to 9 ms of delay.
* **Volume matched:** 2 to 6 ms.
* **Smooth at crest:** 8 to 189 ms. With 1 to 2 mmHg of recoil and a 0.1 to 0.2 mm gap, its apex takes 54 to 211 ms to open. That is longer than Tier 2's prescribed opening, and at a 0.1 mm gap it can exceed the whole 170 ms E wave. At 2 mmHg and 0.1 mm, its apical gap is still only 0.2 mm at peak E.

So Tier 2's prescribed kinematics for the smooth crest ventricle were physically unreachable. Its penalty shows up as delayed apical opening, not just as the 6 to 36 mmHg film suction Tier 2 reported under forced motion. Under realistic recoil the peak film suction is lower: 0.5 to 8 mmHg viscous peak, and 0.6 to 4.5 mmHg mean including added mass. The trabeculated and volume matched walls both open faster than Tier 2 assumed, so their Tier 2 results stand.

**5. Trabeculated versus volume matched.** The two remain close. The volume matched wall opens 1 to 4 ms faster, because its walls start 2 mm further apart. That's consistent with Tier 2's finding that the volume matched ventricle fills slightly more cheaply while the trabeculated one washes out better.

## Verdict

Tier 3 confirms the mechanism in 3D and makes it pressure driven:

* **The film is real.** Resolved smooth walls reproduce Tier 1's lubrication film at low Reynolds number.
* **Grooves drain it.** The connected 3D groove network drains even better than Tier 1 assumed.
* **The penalty is delay.** When the walls are opened by a realistic recoil stress rather than forced, a smooth apex emptied to 0.1 to 0.2 mm of apposition stays shut for tens to hundreds of milliseconds into early filling. A trabeculated apex opens in about 20 to 32 ms at 1 to 2 mmHg of recoil, within 3 to 9 ms of the limit set by wall inertia alone.

That is the strongest form of the hypothesis these tiers support: **trabeculae let the apex empty to near contact and still reopen on time.** The ledger against a smooth ventricle that simply empties less is unchanged from Tier 2. That ventricle opens just as fast and fills slightly more cheaply, but washes out worse.

## Related finding for Tier 2

Tier 2's walls used a centred one cell ramp. By the same penalized channel test (2D solver, Tier 2 time step), they sit on average 0.13 cells inside the fluid, with a staircase of up to about half a cell as surfaces move between nodes. The Tier 2 averages are therefore essentially unbiased. But the resolved 0.75 mm apical slit fluctuated by roughly ±0.1 mm, part of the noise in the trabeculated pressure traces, and part of why its resolved apical suction is best read as a worst case. The Tier 2 film closure used kinematics, so it is unaffected. Tier 3 replaces that closure with the pressure driven result above.

## Limitations

1. **Patch, not ventricle.** The resolved runs are a 12 mm square patch with one grooved wall, and the reduced model carries the results to an 8.75 mm disk with 2 mm deep grooves. The inertia corrections are assumed to transfer at equal $\mathrm{Re}_h$. Both walls trabeculated, irregular ridges and tapered recesses are untested.
2. **Rigid walls and an idealized drive.** Each wall is a rigid plate driven by a uniform recoil stress with a 20 ms rise. Real recoil varies along the wall and with strain, and wall compliance would itself relieve suction. The recoil magnitude (1 to 5 mmHg) and wall mass are order of magnitude inputs.
3. **Resolvable gaps only.** Resolved gaps are 0.6 to 1.8 mm. The physiological 0.1 to 0.5 mm films rest on Tier 1 plus the calibration, and the calibrated Reynolds range (2 to 33) covers the early opening but not its fastest phase, where $\kappa$ is held at its edge value.
4. **Calibration scatter.** The fits rest on six windows per surface, with 8 to 12% rms scatter. The groove mesh fit is the noisier of the two.
5. **Not coupled back.** The pressure driven apical opening isn't fed back into the 2D ventricle of Tier 2.
6. **Not a full Tier 3.** No CT derived geometry and no full ventricle fluid structure interaction. That still needs trabecular resolution imaging and HPC.

## Reproduce

```
pip install -r requirements.txt
pytest -m "not slow"
bash results/tier3/lane1.sh & bash results/tier3/lane2.sh   # about 5 CPU hours in total
python -m apex3d.analyze   # figures/fig12 to fig14 and results/tier3_results.json
```

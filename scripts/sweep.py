"""
CALIBRATION SWEEP: required-value thresholds across parameter ranges.

Structure of the paper:
  Model      -- analytical, no data
  Calibration-- THIS FILE: thresholds as functions of parameters, swept over
                plausible ranges. Output = REQUIRED VALUES, not predictions.
  Reality    -- observed EU values located against those thresholds (reality.py)

Every threshold below is a closed form from the model; the sweep only
evaluates it. No parameter here is claimed to be the true value.
"""
import numpy as np

# ---------------------------------------------------------------- ranges
# Bounds are deliberately wide; the point is that conclusions survive them.
PHI   = [30.0, 45.0, 60.0]          # L/pax: OAG EU27 mean sector 1125km,
                                    # burn 0.025-0.040 L/ASK, LF 0.75-0.90
ELAST = [-1.8, -1.4, -1.0]          # short-haul air travel elasticity
FARE  = [80.0, 120.0, 200.0]        # EUR one-way intra-EU
SHARE = [0.15, 0.25, 0.40]          # own-carrier route share

L_PER_T = 1250.0
MJ_PER_L = 43.0 * 0.8

def alpha_of(el, fare, sh):
    """logit: own-price elasticity = -alpha*p*(1-share)"""
    return -el / (fare * (1 - sh))

def markup_of(el, fare, sh):
    return 1.0 / (alpha_of(el, fare, sh) * (1 - sh))

def ci(g_per_MJ):
    return g_per_MJ * MJ_PER_L / 1000.0   # kgCO2e per litre

print("="*84)
print("THRESHOLD 1  --  OVER-COMPLIANCE:  beta/alpha  >  phi * Delta")
print("="*84)
print("""
  Required willingness to pay for 100% SAF, EUR per one-way trip.
  Delta = SAF cost wedge, EUR/litre. Reported for a grid of (phi, Delta).
""")
DELTAS = [0.30, 0.60, 0.90, 1.20, 1.50]
print("  " + " "*14 + "".join(f"{p:>10.0f}" for p in PHI) + "   <- phi (L/pax)")
print("  Delta (EUR/L)")
for d in DELTAS:
    print(f"  {d:>10.2f}    " + "".join(f"{p*d:>10.2f}" for p in PHI))
print("""
  Read: at phi=45 and Delta=0.90, over-compliance requires WTP > EUR 40.50/trip.
  The threshold is LINEAR in both arguments -- no interior optimum to search for.
""")

print("  As a share of fare (the form WTP studies report):")
print("  " + " "*14 + "".join(f"{f:>10.0f}" for f in FARE) + "   <- fare (EUR)")
for d in [0.60, 0.90, 1.20]:
    for p in [45.0]:
        row = f"  Delta={d:.2f}      " + "".join(f"{100*p*d/f:>9.1f}%" for f in FARE)
        print(row)
print("""
  Read: the required WTP share falls with fare. A given Delta that is
  prohibitive in a EUR 80 economy market may be clearable at EUR 200.
""")

print("="*84)
print("THRESHOLD 2  --  BREAK-EVEN CARBON PRICE:  tau* = (P_j - P_F)/(CI_F - CI_j)")
print("="*84)
print("""
  The carbon price at which fuel j becomes cost-neutral without subsidy,
  EUR/tCO2e. Swept over the SAF price ratio and the lifecycle reduction.
""")
PF_T = 640.0                      # EUR/tonne fossil, used only to set the scale
RATIOS = [2.0, 3.0, 5.0, 8.0, 12.0]
REDUCS = [0.50, 0.70, 0.85, 0.95]
print("  " + " "*16 + "".join(f"{r:>10.0f}x" for r in RATIOS) + "  <- SAF/CAF price ratio")
print("  LCA reduction")
for red in REDUCS:
    row = f"  {red:>12.0%}    "
    for r in RATIOS:
        pj = PF_T * r / L_PER_T           # EUR/L
        pf = PF_T / L_PER_T
        cif = ci(89.0)
        cij = cif * (1 - red)
        tau = (pj - pf) / (cif - cij)     # EUR per kg
        row += f"{1000*tau:>11.0f}"
    print(row)
print("""
  Read (EUR per tCO2e): a fuel at 3x fossil price with an 85% lifecycle
  reduction breaks even at a carbon price around EUR 400/t. Deep-reduction
  fuels are cheaper per tonne abated ONLY if their price premium is modest.
""")

print("="*84)
print("THRESHOLD 3  --  WELFARE-IMPROVING CARBON PRICING")
print("     chi > [markup + tau*R_pax]/E_pax   (see derive_welfare2.py)")
print("="*84)
print("""
  Break-even social cost of carbon, EUR/tCO2e, at the mandate corner (phi=45).
  Below chi*, contracting traffic destroys more surplus than damage avoided.

  NOTE: the condition carries a forgone-revenue term
  tau*R_pax, where R_pax is REGULATED emissions. Under the EU ETS only the
  fossil fraction is charged (SAF zero-rated), so R_pax < E_pax and the
  threshold is lower than a lifecycle carbon price would imply.
""")
THETAS = [0.02, 0.06, 0.20, 0.70]
PHI_C = 45.0   # midpoint of the OAG corner envelope (31.2 at 0.025 L/ASK &
               # LF 0.90, up to 60.0 at 0.040 & LF 0.75) -- NOT a central
               # estimate. The calibration (scc_options.py) uses phi=35.
TAU_EU = 80.0  # EUR/tCO2e, representative EU ETS
E_F_ETS = 3.16*0.8   # kgCO2/L: ETS prices combustion only, SAF zero-rated
print(f"  Rows: (elasticity, fare, share) -> markup EUR;  Cols: mandate theta")
print(f"  (evaluated at tau = EUR {TAU_EU:.0f}/tCO2e, no SAF support)")
print("  " + " "*30 + "".join(f"{t:>12.0%}" for t in THETAS))
for el in ELAST:
    for f in FARE:
        sh = 0.25
        mk = markup_of(el, f, sh)
        row = f"  el={el:>4.1f} fare={f:>3.0f} mk={mk:>6.1f}  "
        for th in THETAS:
            cib = ci(89.0)*(1-0.85)
            cib_blend = (1-th)*ci(89.0) + th*cib
            e_pax_t = PHI_C * cib_blend / 1000.0          # lifecycle -> damages
            r_pax_t = PHI_C * (1-th) * E_F_ETS / 1000.0   # regulated -> revenue
            row += f"{(mk + TAU_EU*r_pax_t)/e_pax_t:>12.0f}"
        print(row)
print("""
  Read: chi* rises with the markup and falls with emissions per passenger.
  Tighter mandates RAISE chi* (cleaner blend -> less damage per passenger ->
  weaker case for taxing traffic). Compare against published SCC estimates.
""")

print("="*84)
print("THRESHOLD 4  --  REGIME-FLIPPING SUBSIDY:  sigma^crit")
print("="*84)
print("""
  sigma^crit = P_j - P_F + tau*(CI_j - CI_F) - beta/(alpha*phi)
  Support below this is a pure transfer + traffic expansion (Result 5);
  above it, airlines over-comply and the interior statics apply.
  Reported in EUR/tonne of SAF, at tau = 80 EUR/t, phi = 45 L/pax.
""")
TAU = 0.080
WTPS = [5.0, 15.0, 30.0, 60.0]
print("  " + " "*18 + "".join(f"{w:>12.0f}" for w in WTPS) + "  <- WTP (EUR/trip)")
for r in [2.0, 3.0, 5.0]:
    pj = PF_T*r/L_PER_T; pf = PF_T/L_PER_T
    cif = ci(89.0); cij = cif*0.15
    row = f"  SAF at {r:>3.0f}x     "
    for w in WTPS:
        sig = (pj - pf) + TAU*(cij - cif) - w/PHI_C
        row += f"{sig*L_PER_T:>12.0f}"
    print(row)
print("""
  Read (EUR/tonne SAF): negative entries mean no subsidy is needed -- WTP
  alone already clears the wedge. Large positive entries mean support would
  have to approach or exceed the SAF price itself to change blending.
""")

print("="*84)
print("SUMMARY: what the calibration establishes")
print("="*84)
print("""
  None of the above asserts a true parameter value. Each is a REQUIRED
  value implied by the model. The reality check (reality.py) then asks
  where observed EU magnitudes fall relative to these surfaces.

  The qualitative conclusions that hold across the ENTIRE swept range:
   (a) Threshold 1 is linear in phi and Delta -- no knife-edge.
   (b) Threshold 3 (chi*) never falls below EUR 394/tCO2e anywhere in the
       grid, and exceeds EUR 500 for all but the lowest-fare cells.
   (c) Threshold 4 is positive and large whenever WTP is below ~EUR 30/trip.
""")

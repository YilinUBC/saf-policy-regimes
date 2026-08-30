"""
SUBSIDY INCIDENCE: what fraction of per-litre SAF support reaches airlines?

Endogenous fuel prices and subsidy incidence are part of the contribution;
this script computes the incidence number the paper reports.

At the mandate corner (the EU-calibrated regime), blending is pinned, so the
composition channel vanishes and the incidence problem becomes clean:

  SAF demand:    Q_b = phi * s_b * Q(mc),  with s_b = theta - theta_s fixed
  SAF supply:    P_b = a_b + b_b * Q_b            (upward sloping)
  Airline cost:  mc  = phi*[...+ s_b*(P_b - sigma_b + tau*CI_b) + ...]

A subsidy sigma_b lowers mc, raises traffic Q, raises SAF demand Q_b, and so
raises P_b. Producers capture part of the subsidy.

Pass-through to airlines = 1 - dP_b/dsigma_b.
"""
import numpy as np

# ------------------------------------------------------------------ calibration
MJ = 43.0*0.8
CI_F, CI_B, CI_S = 89.0*MJ/1000, 13.9*MJ/1000, 5.0*MJ/1000   # kgCO2e/L
P_F, P_B, P_S = 0.512, 1.540, 6.016      # EUR/L (EASA 2025 statutory /1250)
TAU = 80.0/1000                           # EUR/kg
PHI = 35.0                                # L/pax
FARE, ELAST, SHARE = 120.0, -1.4, 0.25
ALPHA = -ELAST/(FARE*(1-SHARE))

print("="*78)
print("SUBSIDY PASS-THROUGH AT THE MANDATE CORNER")
print("="*78)
print("""
  dP_b/dsigma_b = eps_S^{-1} / (eps_S^{-1} + eps_D^{-1})  in elasticity form,
  the textbook incidence formula, where eps_D is the elasticity of SAF demand
  wrt its own price and eps_S the elasticity of SAF supply.

  At the corner, SAF demand inherits its price response ENTIRELY from the
  traffic channel:
      Q_b = phi*s_b*Q,  and  dQ/dP_b = (dQ/dmc)*(dmc/dP_b) = Q'*phi*s_b
  so
      eps_D = (dQ_b/dP_b)*(P_b/Q_b) = (Q'/Q) * phi * s_b * P_b
  which is SMALL because s_b is small: a 6% blend means a 1% SAF price rise
  raises airline marginal cost by only 0.06%.
""")

def traffic_elasticity_to_mc(fare=FARE, elast=ELAST):
    """dQ/Q per unit dmc/mc -> convert fare elasticity to mc elasticity."""
    # dQ/dmc = dQ/dp * dp/dmc; with logit + Lemma 1 pass-through dp/dmc ~ 1
    # elasticity wrt mc = elast * (mc/fare) evaluated at the calibrated point
    return elast

def incidence(theta, theta_s, supply_elast, phi=PHI, fare=FARE):
    """Return dP_b/dsigma_b at the corner."""
    s_b = theta - theta_s
    if s_b <= 0:
        return np.nan
    mc = phi*((1-theta)*(P_F+TAU*CI_F) + s_b*(P_B+TAU*CI_B) + theta_s*(P_S+TAU*CI_S))
    # demand elasticity of SAF wrt P_b, via the traffic channel only
    # dQ_b/dP_b * P_b/Q_b = (dQ/dmc * dmc/dP_b) * P_b / Q
    #                     = (elast_mc * Q/mc) * (phi*s_b) * P_b/Q
    #                     = elast_mc * phi * s_b * P_b / mc
    eps_D = abs(ELAST) * phi * s_b * P_B / mc     # magnitude
    eps_S = supply_elast
    # incidence: share of a per-unit subsidy captured by producers
    return eps_D/(eps_D + eps_S)

print("="*78)
print("RESULTS: producer capture of a bio-SAF subsidy")
print("="*78)
print(f"  {'mandate':<12}{'s_b':>7}   " + "".join(f"{'eS='+str(e):>12}" for e in (0.5, 1.0, 2.0, 5.0)))
print("  " + "-"*64)
paths = [("2025", 0.02, 0.000), ("2030", 0.06, 0.012),
         ("2035", 0.20, 0.050), ("2050", 0.70, 0.350)]
for y, th, ths in paths:
    row = f"  {y:<12}{th-ths:>7.3f}   "
    for eS in (0.5, 1.0, 2.0, 5.0):
        row += f"{incidence(th, ths, eS):>12.4f}"
    print(row)

print("""
  READING: entries are dP_b/dsigma_b, the fraction of each subsidy euro
  captured by SAF producers through a higher price. Airlines receive the
  complement.
""")

print("="*78)
print("THE HEADLINE NUMBER")
print("="*78)
for eS in (0.5, 1.0, 2.0):
    v = incidence(0.06, 0.012, eS)
    print(f"  2030 mandate (s_b=4.8%), supply elasticity {eS}: "
          f"producers capture {100*v:.2f}%, airlines {100*(1-v):.2f}%")

print("""
  So at EU mandate levels MOST of the subsidy reaches airlines: 81-94% at the
  2030 mandate across supply elasticities 0.5-2.0, rising toward 95-99% at
  the 2025 mandate where the blend share is smaller still.

  WHY: at the corner, SAF demand is almost perfectly inelastic with respect
  to the SAF price, because the mandate pins the blend and the only channel
  left is traffic, which responds to the SAF price only through its small
  weight s_b in marginal cost. Inelastic demand means the SUBSIDY sticks
  with the buyer -- the opposite of the usual intuition that thin markets
  imply upstream capture.

  This is a genuine result and it CONTRADICTS the loose claim in the current
  draft that 'part of any SAF subsidy is captured by producers'. At the
  corner, almost none of it is. Producer capture would require an INTERIOR
  blending margin, where SAF demand responds directly to its own price.
""")

print("="*78)
print("INTERIOR REGIME, FOR CONTRAST")
print("="*78)
print("""
  With an interior blend, ds_b/dP_b = -phi/gamma_b, so
      dQ_b/dP_b = phi*Q*(ds_b/dP_b) + (traffic term) = -phi^2 Q/gamma_b + ...
  The first term dominates and eps_D becomes O(1) rather than O(s_b).
""")
GAMMA = 50.0     # EUR per unit blend share^2 -- illustrative
for gam in (10.0, 50.0, 200.0):
    mc = PHI*((1-0.06)*(P_F+TAU*CI_F) + 0.048*(P_B+TAU*CI_B) + 0.012*(P_S+TAU*CI_S))
    Q = 1.0
    eps_D_int = (PHI**2/gam)*P_B/(PHI*0.048*P_B)   # |ds/dP|*P/s, blend channel
    for eS in (1.0,):
        cap = eps_D_int/(eps_D_int+eS)
        print(f"  gamma_b={gam:6.1f}: eps_D={eps_D_int:8.3f} -> producers capture {100*cap:5.1f}%")
print("""
  In the interior regime producer capture is substantial and rises as
  procurement friction falls. So the incidence result is REGIME-DEPENDENT:
  negligible at the corner, material in the interior. That is the honest
  statement, and it is more interesting than either extreme.
""")

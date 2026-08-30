"""
TWO-FUEL MODEL: bio-SAF and synthetic (e-SAF) as separate inputs.

Motivation from the EASA 2025 reference prices:
    CAF    EUR   640/t
    bio    EUR 1,925/t   (3.01x)
    e-SAF  EUR 7,520/t   (11.75x)
ReFuelEU imposes a total SAF mandate theta AND a synthetic sub-mandate theta_e.
Aggregating them into one theta averages over a 4x price difference -- untenable.

Fuel shares: s_b (bio), s_e (synthetic), 1-s_b-s_e (fossil).
Constraints:  s_b + s_e >= theta   (total SAF mandate)
              s_e       >= theta_e (synthetic sub-mandate)
              s_b, s_e  >= 0
"""
import sympy as sp

alpha, beta_e_b, beta_e_e, beta_f = sp.symbols(
    'alpha beta_b beta_s beta_f', positive=True)
gamma_b, gamma_e, phi, kappa = sp.symbols('gamma_b gamma_s phi kappa', positive=True)
sb, se = sp.symbols('s_b s_s', nonnegative=True)
th, th_e = sp.symbols('theta theta_s', nonnegative=True)
tau, sig_b, sig_e = sp.symbols('tau sigma_b sigma_s', nonnegative=True)
PF, PB, PE = sp.symbols('P_F P_B P_S', positive=True)
CI_F, CI_B, CI_E = sp.symbols('CI_F CI_B CI_S', positive=True)

# policy-adjusted prices
PFt = PF + tau*CI_F
PBt = PB - sig_b + tau*CI_B
PEt = PE - sig_e + tau*CI_E

D_b = sp.simplify(PBt - PFt)      # bio wedge
D_e = sp.simplify(PEt - PFt)      # synthetic wedge

print("="*76)
print("WEDGES")
print("="*76)
print("  Delta_b =", sp.expand(D_b))
print("  Delta_s =", sp.expand(D_e))
print("\n  dDelta_b/dtau =", sp.simplify(sp.diff(D_b, tau)), "  (= CI_B - CI_F < 0)")
print("  dDelta_s/dtau =", sp.simplify(sp.diff(D_e, tau)), "  (= CI_S - CI_F < 0)")
print("""
  Carbon pricing narrows BOTH wedges, but by DIFFERENT amounts:
  the reduction is tau*(CI_F - CI_j), proportional to each fuel's
  lifecycle ADVANTAGE. Since CI_S < CI_B, carbon pricing favours
  synthetic MORE than bio. A per-litre subsidy does not discriminate.
  -> the two instruments reshuffle the bio/synthetic MIX differently.
     This is Observation 1, now with a composition margin.
""")

# ------------------------------------------------------------------ costs
# per-passenger fuel cost with two SAF types
c_fuel = phi*((1 - sb - se)*PFt + sb*PBt + se*PEt)
c_fuel = sp.simplify(sp.expand(c_fuel))
# separable quadratic procurement costs (distinct supply chains)
c_proc = gamma_b/2*(sb - th)**2 + gamma_e/2*(se - th_e)**2

print("="*76)
print("PER-PASSENGER FUEL COST")
print("="*76)
print("  c_fuel =", sp.collect(sp.expand(c_fuel), [sb, se]))
print("\n  Rewritten with wedges:  c = phi*(P~_F + s_b*Delta_b + s_s*Delta_s)")
chk = sp.simplify(sp.expand(c_fuel - phi*(PFt + sb*D_b + se*D_e)))
print("  verification:", chk)

# --------------------------------------------- interior FOCs (Bertrand)
# From derive_bertrand.py: the s-margin FOC is  beta_j/alpha = dc/ds_j
print("\n" + "="*76)
print("INTERIOR SHARES (Bertrand-logit; same envelope logic as one-fuel case)")
print("="*76)
c_tot = c_fuel + c_proc
foc_b = sp.simplify(beta_e_b/alpha - sp.diff(c_tot, sb))
foc_e = sp.simplify(beta_e_e/alpha - sp.diff(c_tot, se))
print("  FOC bio:", sp.simplify(foc_b))
print("  FOC syn:", sp.simplify(foc_e))

sol = sp.solve([sp.Eq(foc_b,0), sp.Eq(foc_e,0)], [sb, se], dict=True)[0]
sb_star = sp.simplify(sol[sb]); se_star = sp.simplify(sol[se])
print("\n  s_b* =", sp.simplify(sp.expand(sb_star)))
print("  s_s* =", sp.simplify(sp.expand(se_star)))

tb = th + (beta_e_b/alpha - phi*D_b)/gamma_b
te = th_e + (beta_e_e/alpha - phi*D_e)/gamma_e
print("\n  Match to the one-fuel form theta_j + (beta_j/alpha - phi*Delta_j)/gamma_j:")
print("    bio:", sp.simplify(sp.expand(sb_star - tb)))
print("    syn:", sp.simplify(sp.expand(se_star - te)))
print("""
  -> 0,0. The two margins SEPARATE: with additively separable procurement
     costs each share has its own independent closed form. Clean result:
     the two-fuel model is no harder than the one-fuel model.
""")

# ---------------------------------------------------- comparative statics
print("="*76)
print("COMPARATIVE STATICS")
print("="*76)
for nm, var in [("tau", tau), ("sigma_b", sig_b), ("sigma_s", sig_e)]:
    print(f"  ds_b*/d{nm:8s} = {sp.simplify(sp.diff(sb_star, var))}")
for nm, var in [("tau", tau), ("sigma_b", sig_b), ("sigma_s", sig_e)]:
    print(f"  ds_s*/d{nm:8s} = {sp.simplify(sp.diff(se_star, var))}")
print("""
  KEY: ds_b/dtau = phi(CI_F-CI_B)/gamma_b  and  ds_s/dtau = phi(CI_F-CI_S)/gamma_s.
  A carbon tax shifts BOTH shares up, in proportion to each fuel's lifecycle
  advantage. A bio subsidy moves ONLY s_b; a synthetic subsidy only s_s.
  So the instruments are NOT substitutes even in direction, let alone in level.
""")

# ------------------------------------------------------- emissions & the mix
print("="*76)
print("EMISSIONS AND THE COMPOSITION MARGIN")
print("="*76)
E = phi*((1-sb-se)*CI_F + sb*CI_B + se*CI_E)
print("  E per pax =", sp.collect(sp.expand(E), [sb, se]))
print("  dE/ds_b =", sp.simplify(sp.diff(E, sb)), "  (< 0)")
print("  dE/ds_s =", sp.simplify(sp.diff(E, se)), "  (< 0, larger in magnitude)")

# abatement cost per tonne for each fuel
print("\n  IMPLICIT ABATEMENT COST (EUR per tonne CO2e) for each fuel:")
ac_b = sp.simplify(D_b/(CI_F - CI_B))
ac_e = sp.simplify(D_e/(CI_F - CI_E))
print("    bio:", ac_b)
print("    syn:", ac_e)
print("""
  A cost-effective mandate would equalise these. ReFuelEU does NOT --
  it fixes quantities (theta, theta_s) regardless of relative abatement
  cost. That inefficiency is measurable and is a core policy result.
""")

# ---------------------------------------------------- the corner condition
print("="*76)
print("CORNER CONDITIONS (what the EU calibration implies)")
print("="*76)
print("""
  s_b* > theta   iff  beta_b/alpha > phi*Delta_b
  s_s* > theta_s iff  beta_s/alpha > phi*Delta_s

  With EASA prices (calibrate_eu.py):
     phi*Delta_b = EUR  28.75/pax   (24% of a EUR 120 fare)
     phi*Delta_s = EUR 184.55/pax   (154% of fare)
  and measured WTP is EUR 5-20. BOTH bind at the corner:
     s_b* = theta,  s_s* = theta_s.

  => Under ReFuelEU the equilibrium is at the mandate floor for BOTH fuels.
     Policy operates through FARES, TRAFFIC and INCIDENCE, not through
     voluntary blending. This is the paper's central positive result.
""")

# threshold carbon price for each fuel
print("  Carbon price at which each wedge vanishes (blending becomes free):")
tau_b = sp.solve(sp.Eq(D_b, 0), tau)[0]
tau_e = sp.solve(sp.Eq(D_e, 0), tau)[0]
print("    bio:  tau* =", sp.simplify(tau_b))
print("    syn:  tau* =", sp.simplify(tau_e))
print("""
  tau*_j = (P_j - sigma_j - P_F)/(CI_F - CI_j): the carbon price that makes
  fuel j cost-neutral. Exactly the implicit abatement cost above. Report
  both numerically -- they differ by an order of magnitude for the EU.
""")

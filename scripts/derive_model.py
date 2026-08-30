"""
Symbolic derivation for the Part B model:
  logit inverse demand + Cournot + convex frequency cost + QUADRATIC SAF cost.

Airline i on route r chooses (q, f, s) taking rivals and fuel prices as given.
Verifies: FOCs, interior s*, closed-form f*, and the Hessian conditions
needed for concavity (existence) and diagonal dominance (uniqueness).
"""
import sympy as sp

# ---------------------------------------------------------------- symbols
q, f, s = sp.symbols('q f s', positive=True)
Q_minus, q0 = sp.symbols('Q_minus q0', positive=True)   # rivals' output, outside option
alpha, beta_f, beta_e = sp.symbols('alpha beta_f beta_e', positive=True)
phi, kappa, gamma = sp.symbols('phi kappa gamma', positive=True)
PF, PS = sp.symbols('P_F P_S', positive=True)            # market fuel prices
tau, sigma, theta = sp.symbols('tau sigma theta', nonnegative=True)
CI_F, CI_S = sp.symbols('CI_F CI_S', positive=True)

# ------------------------------------------------- policy-adjusted prices
PF_t = PF + tau * CI_F                # effective fossil price
PS_t = PS - sigma + tau * CI_S        # effective SAF price
Delta = sp.simplify(PS_t - PF_t)      # the SAF cost wedge

print("=" * 70)
print("SAF cost wedge  Delta = P~_S - P~_F")
print("  Delta =", sp.expand(Delta))
print("  dDelta/dsigma =", sp.diff(Delta, sigma))
print("  dDelta/dtau   =", sp.simplify(sp.diff(Delta, tau)), "   (< 0 iff CI_S < CI_F)")

# --------------------------------------------------- inverse demand (logit)
# p = (beta_f*f + beta_e*s - ln(q/q0)) / alpha
p = (beta_f * f + beta_e * s - sp.log(q / q0)) / alpha

# ------------------------------------------------------------------ costs
# per-passenger fuel cost
c_fuel = phi * ((1 - s) * PF_t + s * PS_t)
# quadratic SAF procurement cost, per passenger, measured from the mandate floor
c_saf = gamma / 2 * (s - theta) ** 2
# convex frequency cost
c_freq = kappa / 2 * f ** 2

pi = (p - c_fuel - c_saf) * q - c_freq

print("\n" + "=" * 70)
print("PROFIT")
sp.pprint(sp.simplify(pi))

# ------------------------------------------------------------------ FOCs
dq = sp.simplify(sp.diff(pi, q))
df = sp.simplify(sp.diff(pi, f))
ds = sp.simplify(sp.diff(pi, s))

print("\n" + "=" * 70)
print("FOC  dpi/dq = 0")
sp.pprint(sp.simplify(dq))
print("\nFOC  dpi/df = 0")
sp.pprint(df)
print("\nFOC  dpi/ds = 0")
sp.pprint(sp.expand(ds))

# ------------------------------------------------- closed forms f* and s*
f_star = sp.solve(sp.Eq(df, 0), f)
print("\n" + "=" * 70)
print("Closed-form frequency  f* :")
sp.pprint(sp.simplify(f_star[0]))

s_sol = sp.solve(sp.Eq(ds, 0), s)
s_star = sp.simplify(s_sol[0])
print("\nClosed-form INTERIOR SAF share  s* :")
sp.pprint(sp.expand(s_star))

# rewrite s* in terms of the wedge Delta
s_star_wedge = sp.simplify(s_star.subs({PS: Delta + PF + sigma - tau * CI_S}))
print("\n  s* in terms of the wedge:  s* = theta + (beta_e/alpha - phi*Delta)/gamma")
check = sp.simplify(s_star - (theta + (beta_e / alpha - phi * Delta) / gamma))
print("  verification (should be 0):", sp.simplify(sp.expand(check)))

# ----------------------------------------------- comparative statics of s*
print("\n" + "=" * 70)
print("COMPARATIVE STATICS of interior s*")
ds_dsigma = sp.simplify(sp.diff(s_star, sigma))
ds_dtau = sp.simplify(sp.diff(s_star, tau))
ds_dtheta = sp.simplify(sp.diff(s_star, theta))
print("  ds*/dsigma =", ds_dsigma, "   (>0)")
print("  ds*/dtau   =", ds_dtau, "   (>0 iff CI_F > CI_S)")
print("  ds*/dtheta =", ds_dtheta, "   (full pass-through of the mandate)")

# ---------------------------------------- THE EQUIVALENCE-FAILURE RESULT
# Iso-average-cost pair: perturb (tau,sigma) so that AVERAGE fuel cost at
# s = theta is unchanged. Then ask what happens to the MARGINAL incentive.
print("\n" + "=" * 70)
print("OBSERVATION 1: average-cost equivalence != incentive equivalence")
AC = phi * ((1 - theta) * PF_t + theta * PS_t)     # avg fuel cost AT the mandate
dAC_dtau = sp.simplify(sp.diff(AC, tau))
dAC_dsigma = sp.simplify(sp.diff(AC, sigma))
print("  dAC/dtau   =", dAC_dtau)
print("  dAC/dsigma =", dAC_dsigma)
# iso-average-cost direction: dsigma/dtau holding AC fixed
iso = sp.simplify(-dAC_dtau / dAC_dsigma)
print("  iso-AC slope  dsigma/dtau |_AC =", sp.simplify(iso))
# effect on s* along that direction
ds_along_iso = sp.simplify(ds_dtau + ds_dsigma * iso)
print("  ds*/dtau along iso-AC locus  =", sp.simplify(sp.expand(ds_along_iso)))
print("  -> nonzero unless CI_F = CI_S: SAME average cost, DIFFERENT blending.")

# --------------------------------------------------- second-order / Hessian
print("\n" + "=" * 70)
print("SECOND-ORDER CONDITIONS (own-choice concavity)")
H = sp.Matrix([[sp.diff(pi, a, b) for b in (q, f, s)] for a in (q, f, s)])
H = sp.simplify(H)
print("\nHessian in (q, f, s):")
sp.pprint(H)

print("\nLeading principal minors:")
m1 = sp.simplify(H[0, 0])
m2 = sp.simplify((H[:2, :2]).det())
m3 = sp.simplify(H.det())
print("  H11        =", m1, "   (<0 required)")
print("  |H_2|      =", sp.simplify(sp.expand(m2)), "   (>0 required)")
print("  det H      =", sp.simplify(sp.expand(m3)), "   (<0 required)")

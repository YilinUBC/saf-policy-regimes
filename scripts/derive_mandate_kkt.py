"""
THE MANDATE CONSTRAINT SET, DONE PROPERLY.

The issue: the paper states the
ReFuelEU constraints as

    s_b + s_s >= theta        (total SAF mandate)
    s_s       >= theta_s      (synthetic sub-mandate)

but the procurement cost is written with SEPARATE floors

    C = (gamma_b/2)(s_b - theta_b)^2 + (gamma_s/2)(s_s - theta_s)^2

with theta_b never defined. Under the true coupled constraint, synthetic SAF
can substitute for bio-SAF in meeting the total, so Proposition 1's zero
cross-effects need not hold globally.

This script:
  1. sets up the KKT system under the TRUE constraints
  2. enumerates the regimes
  3. checks in which regimes the separation result survives
  4. checks whether the EU-calibrated corner is one of them
  5. checks the feasibility constraint s_b + s_s <= 1
"""
import sympy as sp

sb, ss = sp.symbols('s_b s_s', nonnegative=True)
gb, gs = sp.symbols('gamma_b gamma_s', positive=True)
th, ths = sp.symbols('theta theta_s', nonnegative=True)
phi = sp.symbols('phi', positive=True)
Db, Ds = sp.symbols('Delta_b Delta_s')          # policy-adjusted wedges (can be any sign)
bb, bs = sp.symbols('beta_b beta_s', nonnegative=True)
alpha = sp.symbols('alpha', positive=True)
lam, mu, nu = sp.symbols('lambda mu nu', nonnegative=True)   # KKT multipliers

print("="*78)
print("PART 1: THE OBJECTIVE, WITH REFERENCE POINT AT THE *ANCHOR*")
print("="*78)
print("""
  Per-passenger blending payoff (the part of profit that depends on s):

      V(s_b,s_s) = (beta_b/alpha) s_b + (beta_s/alpha) s_s          [demand]
                   - phi*(s_b*Delta_b + s_s*Delta_s)                [fuel cost]
                   - (gamma_b/2)(s_b - a_b)^2 - (gamma_s/2)(s_s - a_s)^2

  where (a_b, a_s) is the procurement ANCHOR: the blend at which the airline
  faces no incremental procurement friction. The paper conflates this anchor
  with the mandate floor. They are conceptually distinct:
      - the mandate floor is a LEGAL constraint (an inequality)
      - the anchor is a TECHNOLOGICAL/CONTRACTUAL reference (where the
        offtake book already sits)
  Setting a_j = the floor says "contracted volumes exactly meet the mandate",
  which is the natural steady state and what the paper intends.
""")

ab, as_ = sp.symbols('a_b a_s', nonnegative=True)
V = (bb/alpha)*sb + (bs/alpha)*ss - phi*(sb*Db + ss*Ds) \
    - gb/2*(sb - ab)**2 - gs/2*(ss - as_)**2

print("="*78)
print("PART 2: KKT UNDER THE TRUE COUPLED CONSTRAINTS")
print("="*78)
print("""
  max V  s.t.  g1: s_b + s_s - theta   >= 0
               g2: s_s - theta_s       >= 0
               g3: 1 - s_b - s_s       >= 0     (feasibility -- MISSING in draft)
               s_b, s_s >= 0

  L = V + lambda*g1 + mu*g2 - nu*(s_b + s_s - 1)
""")

L = V + lam*(sb + ss - th) + mu*(ss - ths) - nu*(sb + ss - 1)
FOC_b = sp.diff(L, sb)
FOC_s = sp.diff(L, ss)
print("  dL/ds_b =", sp.simplify(FOC_b))
print("  dL/ds_s =", sp.simplify(FOC_s))

# Define the "unconstrained desired" shares
zb = sp.symbols('z_b')   # desired bio share absent constraints
zs = sp.symbols('z_s')
zb_expr = ab + (bb/alpha - phi*Db)/gb
zs_expr = as_ + (bs/alpha - phi*Ds)/gs
print("""
  Unconstrained optima (lambda=mu=nu=0):
      z_b = a_b + (beta_b/alpha - phi*Delta_b)/gamma_b
      z_s = a_s + (beta_s/alpha - phi*Delta_s)/gamma_s
  These are the interior shares of eq (8) with a_j in place of theta_j.
""")

print("="*78)
print("PART 3: REGIME ENUMERATION")
print("="*78)
print("""
  REGIME I  (both constraints slack): z_b + z_s > theta and z_s > theta_s.
      s_b* = z_b, s_s* = z_s.
      Cross-effects: ds_b/dsigma_s = 0 and ds_s/dsigma_b = 0 EXACTLY,
      because each FOC involves only its own share. Proposition 1 HOLDS.

  REGIME II (total binds, sub-mandate slack): lambda > 0, mu = 0.
      FOCs:  beta_b/alpha - phi*Delta_b - gamma_b(s_b - a_b) + lambda = 0
             beta_s/alpha - phi*Delta_s - gamma_s(s_s - a_s) + lambda = 0
             s_b + s_s = theta
      Now BOTH shares depend on BOTH subsidies through lambda.
      Proposition 1 FAILS here.  <-- the case that matters

  REGIME III (both bind): lambda > 0, mu > 0.
      s_s = theta_s and s_b = theta - theta_s. Both shares are PINNED by
      policy. Cross-effects are zero trivially (nothing responds to anything).
      Proposition 1 holds VACUOUSLY. This is the EU corner of Result 1.

  REGIME IV (sub-mandate binds, total slack): mu > 0, lambda = 0.
      s_s = theta_s, s_b = z_b > theta - theta_s.
      ds_b/dsigma_s = 0 (s_b's FOC has no sigma_s); ds_s/dsigma_b = 0
      (s_s pinned). Proposition 1 HOLDS.
""")

# Solve Regime II explicitly to show the coupling
print("="*78)
print("PART 4: REGIME II EXPLICITLY -- WHERE SEPARATION FAILS")
print("="*78)
sol2 = sp.solve([sp.Eq(bb/alpha - phi*Db - gb*(sb - ab) + lam, 0),
                 sp.Eq(bs/alpha - phi*Ds - gs*(ss - as_) + lam, 0),
                 sp.Eq(sb + ss, th)], [sb, ss, lam], dict=True)[0]
sb2 = sp.simplify(sol2[sb]); ss2 = sp.simplify(sol2[ss]); lam2 = sp.simplify(sol2[lam])
print("  s_b* =", sb2)
print("  s_s* =", ss2)
print("  lambda* =", lam2)

# sigma_b enters via Delta_b = P_b - sigma_b + tau*CI_b - P_F - tau*CI_F,
# so dDelta_b/dsigma_b = -1. Cross-derivative of s_s wrt sigma_b:
dss_dsigb = sp.simplify(sp.diff(ss2, Db)*(-1))
dsb_dsigs = sp.simplify(sp.diff(sb2, Ds)*(-1))
print("\n  ds_s*/dsigma_b =", dss_dsigb)
print("  ds_b*/dsigma_s =", dsb_dsigs)
print("""
  NONZERO. In Regime II a subsidy to one fuel crowds out the other, because
  the total mandate is a shared budget: buying more of the subsidised fuel
  relaxes the requirement to buy the other. Magnitude phi/(gamma_b+gamma_s),
  and note the SIGN is negative -- pure substitution.

  This is a genuinely new comparative static, not a defect. It should be
  stated as part of Proposition 1 rather than suppressed.
""")

print("="*78)
print("PART 5: WHICH REGIME IS THE EU?")
print("="*78)
MJ = 43.0*0.8
CI_F, CI_B, CI_S = 89.0*MJ/1000, 13.9*MJ/1000, 5.0*MJ/1000
P_F, P_B, P_S = 0.512, 1.540, 6.016      # EUR/L, EASA 2025 statutory (640,1925,7520 EUR/t / 1250)
TAU, PHI = 80.0/1000, 35.0               # EUR per kg; L/pax
D_b = (P_B + TAU*CI_B) - (P_F + TAU*CI_F)
D_s = (P_S + TAU*CI_S) - (P_F + TAU*CI_F)
print(f"  Delta_b = EUR {D_b:.4f}/L  ->  phi*Delta_b = EUR {PHI*D_b:.2f}/pax")
print(f"  Delta_s = EUR {D_s:.4f}/L  ->  phi*Delta_s = EUR {PHI*D_s:.2f}/pax")
print("""
  Retail WTP for full SAF is ~EUR 3/trip (Li et al. 2026), so
  beta_b/alpha ~ 3 << phi*Delta_b ~ 29, and beta_s/alpha ~ 3 << 185.
  Both unconstrained optima z_j lie BELOW their floors:
      z_b = a_b + (3 - 29)/gamma_b < a_b,   z_s = a_s + (3 - 185)/gamma_s < a_s
  With a_j at the floors, both constraints bind => REGIME III.

  => At EU retail parameters the economy is in Regime III, where both shares
     are pinned and the separation question is moot. The Regime II coupling
     matters for the CORPORATE segment and for counterfactuals with high
     support, which is exactly where the paper's interior analysis lives.
""")

print("="*78)
print("PART 6: FEASIBILITY  s_b + s_s <= 1")
print("="*78)
print("""
  The draft's strategy set omits this. It binds (nu > 0) only if
      z_b + z_s > 1,
  i.e. if the demand benefit exceeds the cost wedge by more than the whole
  remaining fossil share. At EU parameters z_b + z_s < theta < 1, so it is
  slack by a wide margin and nothing in the paper's results is affected.
  But it must be STATED for the strategy set to be compact -- which is what
  Proposition 2's existence argument (Weierstrass on a compact convex set)
  actually requires.
""")

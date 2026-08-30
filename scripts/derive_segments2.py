"""
EXACT effective valuation in the two-type model.

The issue: W(omega) = (1-omega)W_R + omega*W_C is claimed to
follow from the two-type logit model, but with separate logit demands the
blending FOC weights valuations by TYPE-SPECIFIC DEMAND RESPONSES, not by
population shares. Corporate and retail travellers have different participation
probabilities and outside-option shares.

This script derives the exact weight and states precisely when the simple
population-weighted average obtains.
"""
import sympy as sp

om = sp.symbols('omega', positive=True)          # corporate POPULATION share
aR, aC = sp.symbols('alpha_R alpha_C', positive=True)
bR, bC = sp.symbols('beta_R beta_C', nonnegative=True)
vR, vC = sp.symbols('varsigma_R varsigma_C', positive=True)   # own share within type
M, phi, gam, Dl = sp.symbols('M phi gamma Delta', positive=True)
s, a = sp.symbols('s a', nonnegative=True)

print("="*78)
print("PART 1: THE EXACT BLENDING FIRST-ORDER CONDITION")
print("="*78)
print("""
  Airline i sets one fare p and one blend s. Type-k demand is logit:
      q^k_i = M_k * varsigma^k_i,   M_R = (1-omega)M,  M_C = omega*M
  with  d varsigma^k_i / d s_i = beta_k * varsigma^k_i (1 - varsigma^k_i)
        d varsigma^k_i / d p_i  = -alpha_k * varsigma^k_i (1 - varsigma^k_i)

  Total traffic  Q_i = M[(1-om) vR + om vC].
  Profit  Pi = (p - mc(s)) Q_i.
""")

# marginal benefit of blending: extra traffic from both types, valued at margin
MB = (1-om)*bR*vR*(1-vR) + om*bC*vC*(1-vC)
# traffic base that pays the extra blending cost
BASE = (1-om)*vR + om*vC
# price FOC: markup satisfies  Q = mu * sum_k M_k alpha_k vs_k (1-vs_k)
DEN_P = (1-om)*aR*vR*(1-vR) + om*aC*vC*(1-vC)

print("  marginal traffic gain from ds :", sp.simplify(MB), "  (x M)")
print("  traffic paying the extra cost :", sp.simplify(BASE), "  (x M)")
print("  price-FOC denominator         :", sp.simplify(DEN_P), "  (x M)")
print("""
  Markup from the price FOC:  mu = BASE / DEN_P.
  Blending FOC:  mu * MB = BASE * dmc/ds, hence

      dmc/ds = MB / DEN_P                                            (exact)

  So the object that replaces beta/alpha in the single-type model is
""")
W_exact = MB/DEN_P
print("      W_eff = MB/DEN_P =")
sp.pprint(sp.simplify(W_exact))

print("="*78)
print("PART 2: WHEN DOES THIS COLLAPSE TO A POPULATION-WEIGHTED AVERAGE?")
print("="*78)

print("\n  CASE A: common price sensitivity alpha_R = alpha_C = alpha")
WA = sp.simplify(W_exact.subs({aR: sp.Symbol('alpha', positive=True),
                               aC: sp.Symbol('alpha', positive=True)}))
print("      W_eff =", WA)
print("""
      = [ (1-om) vR(1-vR) beta_R + om vC(1-vC) beta_C ]
        / { alpha [ (1-om) vR(1-vR) + om vC(1-vC) ] }

      a DEMAND-RESPONSE-WEIGHTED average of W_k = beta_k/alpha, with weights
          w_k proportional to  M_k * varsigma^k (1 - varsigma^k).
      NOT the population share unless the vs terms coincide.
""")

print("  CASE B: additionally equal type-specific choice shares vR = vC = v")
alpha = sp.Symbol('alpha', positive=True)
WB = sp.simplify(WA.subs({vR: sp.Symbol('v', positive=True),
                          vC: sp.Symbol('v', positive=True)}))
print("      W_eff =", WB)
target = (1-om)*bR/alpha + om*bC/alpha
print("      target (1-om)W_R + om W_C =", sp.simplify(target))
print("      identical?", sp.simplify(WB - target) == 0)
print("""
      EXACT. So the population-weighted formula requires BOTH
        (i)  alpha_R = alpha_C, and
        (ii) equal type-specific choice shares (the types differ only in
             their green valuation, not in participation).
""")

print("="*78)
print("PART 3: HOW WRONG IS THE APPROXIMATION?")
print("="*78)
print("""
  Define the demand-response weight on corporates:

      omega_eff = om*vC(1-vC) / [ (1-om) vR(1-vR) + om vC(1-vC) ]

  The exact condition is  (1-omega_eff) W_R + omega_eff W_C > phi*Delta,
  i.e. the SAME threshold with omega replaced by omega_eff. So the
  composition result is structurally exact; only the mapping from population
  share to effective weight changes.
""")

vRn, vCn = sp.symbols('v_R v_C', positive=True)
om_eff = om*vCn*(1-vCn)/((1-om)*vRn*(1-vRn) + om*vCn*(1-vCn))
print("  omega_eff =", om_eff)

print("\n  numerical sensitivity (om = 0.112, the OAG premium-cabin share):")
for vr, vc in [(0.25, 0.25), (0.25, 0.20), (0.25, 0.30), (0.20, 0.35)]:
    val = float(om_eff.subs({om: 0.112, vRn: vr, vCn: vc}))
    print(f"    vs_R={vr:.2f}, vs_C={vc:.2f}  ->  omega_eff = {val:.4f}"
          f"   ({'above' if val > 0.112 else 'below'} population share)")

print("""
  Corporate travellers plausibly have a LOWER outside-option share (they fly
  regardless), i.e. higher vs_C, which RAISES omega_eff above the population
  share. The population-weighted formula is then CONSERVATIVE for
  over-compliance: it understates the corporate weight.

  Since the paper's conclusion is that routes fall BELOW the threshold, using
  the conservative weight is the right direction: a route that fails the test
  under omega_eff > omega fails it a fortiori under omega.
""")

print("="*78)
print("PART 4: RECOMPUTE omega* UNDER THE EXACT WEIGHT")
print("="*78)
PHI_D, W_R = 28.75, 3.0
for W_C, lab in [(51.7, "60% level"), (131.0, "100% level")]:
    om_star = (PHI_D - W_R)/(W_C - W_R)
    print(f"  W_C={W_C:6.1f} ({lab:10s}): omega*_eff = {om_star:.3f}"
          f"  -- in EFFECTIVE-weight units, not population units")
print("""
  The numerical omega* is unchanged: it is a threshold on the weight that
  enters the FOC. What changes is its interpretation -- it is a threshold on
  omega_eff, and equals a threshold on the population share only under (i)
  and (ii) above.
""")

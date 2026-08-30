"""
FUEL-MARKET FIXED POINT -- corrected.

Fix vs derive_fuelmarket.py: mc evaluated at the PROFIT-maximizing s* is not
a cost function, so Shephard's lemma does not apply and dmc/dP_F != phi*(1-s*).
The correct total derivative must include the s* response:

    dmc/dP_F = dmc/dP_F|_s  +  (dmc/ds)*(ds*/dP_F)

and at the profit optimum dmc/ds = beta_e/alpha (NOT zero), because the
s-FOC equates marginal cost of blending to the marginal DEMAND benefit.
That is the source of the beta_e/(alpha*gamma) term.
"""
import sympy as sp

alpha, beta_e = sp.symbols('alpha beta_e', positive=True)
gamma, phi = sp.symbols('gamma phi', positive=True)
theta = sp.symbols('theta', nonnegative=True)
tau, sig = sp.symbols('tau sigma', nonnegative=True)
CI_F, CI_S = sp.symbols('CI_F CI_S', positive=True)
bF, bS = sp.symbols('b_F b_S', positive=True)
PF, PS = sp.symbols('P_F P_S', positive=True)

PFt = PF + tau*CI_F
PSt = PS - sig + tau*CI_S
D   = sp.simplify(PSt - PFt)
s_star = theta + (beta_e/alpha - phi*D)/gamma

print("="*74)
print("PART 1: THE CORRECT PRICE DERIVATIVES OF mc AT s*")
print("="*74)

sv = sp.symbols('s_v')
mc_gen = phi*((1-sv)*PFt + sv*PSt) + gamma/2*(sv-theta)**2

# partial derivatives holding s fixed
dmc_dPF_partial = sp.simplify(sp.diff(mc_gen, PF))
dmc_dPS_partial = sp.simplify(sp.diff(mc_gen, PS))
dmc_ds          = sp.simplify(sp.diff(mc_gen, sv))
print("  holding s fixed:  dmc/dP_F =", dmc_dPF_partial, "  dmc/dP_S =", dmc_dPS_partial)
print("  dmc/ds =", sp.simplify(dmc_ds))
print("  at s = s*, dmc/ds =", sp.simplify(sp.expand(dmc_ds.subs(sv, s_star))),
      "  <-- equals beta_e/alpha, NOT zero")
print("""
  This is the s-FOC: marginal cost of blending = marginal demand benefit.
  Cost-minimizing s would set dmc/ds = 0; profit-maximizing s sets it to
  beta_e/alpha. Airlines OVER-BLEND relative to cost minimization because
  SAF raises willingness to pay. Worth stating as a remark in the paper.
""")

# total derivatives
ds_dPF = sp.diff(s_star, PF)
ds_dPS = sp.diff(s_star, PS)
mc_star = sp.simplify(mc_gen.subs(sv, s_star))
dmc_dPF_total = sp.simplify(sp.expand(sp.diff(mc_star, PF)))
dmc_dPS_total = sp.simplify(sp.expand(sp.diff(mc_star, PS)))
print("  ds*/dP_F =", ds_dPF, "   ds*/dP_S =", ds_dPS)
print("  TOTAL dmc/dP_F =", dmc_dPF_total)
print("  TOTAL dmc/dP_S =", dmc_dPS_total)

# decomposition check
chk_F = sp.simplify(sp.expand(dmc_dPF_total - (dmc_dPF_partial.subs(sv,s_star) + (beta_e/alpha)*ds_dPF)))
chk_S = sp.simplify(sp.expand(dmc_dPS_total - (dmc_dPS_partial.subs(sv,s_star) + (beta_e/alpha)*ds_dPS)))
print("\n  DECOMPOSITION CHECK  total - [partial + (beta_e/alpha)*ds/dP]:")
print("    P_F:", chk_F, "   P_S:", chk_S)
print("    -> 0,0 confirms the envelope decomposition with the beta_e/alpha term.")

print("\n  Simplified totals in terms of s*:")
print("    dmc/dP_F = phi*(1-s*) + (beta_e/alpha)*(phi/gamma)")
print("    dmc/dP_S = phi*s*     - (beta_e/alpha)*(phi/gamma)")
v1 = sp.simplify(sp.expand(dmc_dPF_total - (phi*(1-s_star) + (beta_e/alpha)*(phi/gamma))))
v2 = sp.simplify(sp.expand(dmc_dPS_total - (phi*s_star   - (beta_e/alpha)*(phi/gamma))))
print("    verify:", v1, ",", v2)

print("\n" + "="*74)
print("PART 2: CORRECTED JACOBIAN OF THE COMPOSITE MAP")
print("="*74)

Qv, Qp = sp.symbols('Qbar Qprime')      # Q>0, Q'<0
s = sp.symbols('s', positive=True)      # equilibrium s*
be = sp.symbols('B', positive=True)     # shorthand for beta_e/(alpha*gamma)*phi

dc_dPF = phi*(1-s) + be
dc_dPS = phi*s     - be
ds_F   =  phi/gamma
ds_S   = -phi/gamma

dQF_dPF = phi*(Qp*dc_dPF*(1-s) - Qv*ds_F)
dQF_dPS = phi*(Qp*dc_dPS*(1-s) - Qv*ds_S)
dQS_dPF = phi*(Qp*dc_dPF*s     + Qv*ds_F)
dQS_dPS = phi*(Qp*dc_dPS*s     + Qv*ds_S)

DT = sp.Matrix([[bF*dQF_dPF, bF*dQF_dPS],
                [bS*dQS_dPF, bS*dQS_dPS]])
print("  DT =")
sp.pprint(sp.simplify(DT))
print("\n  trace =", sp.simplify(sp.expand(DT.trace())))
print("  det   =", sp.simplify(sp.factor(sp.expand(DT.det()))))

print("""
  SIGN CHECK on own-price entries (need < 0 for well-behaved demand):
    dQ_F/dP_F = phi[ Q'(phi(1-s)+B)(1-s) - Q*phi/gamma ]
      Q'<0 and (phi(1-s)+B)>0 => first term < 0; second term < 0. => NEGATIVE. OK
    dQ_S/dP_S = phi[ Q'(phi*s-B)*s + Q*phi/gamma ]
      SECOND TERM IS NOW POSITIVE (+Q*phi/gamma). And if phi*s < B the first
      term is POSITIVE too. So dQ_S/dP_S can be POSITIVE.
""")

cond = sp.simplify(phi*s - be)
print("  dQ_S/dP_S > 0  possible when  phi*s* < B = beta_e*phi/(alpha*gamma), i.e.")
print("      s* < beta_e/(alpha*gamma)")
print("""
  ECONOMICS: if the green-preference term is large relative to the SAF
  procurement cost slope, a higher SAF price can RAISE SAF demand, because
  the induced fall in blending is outweighed by ... no: rather, the
  composition channel (-Q*ds/dP_S = +Q*phi/gamma) dominates. This is an
  upward-sloping demand region and it THREATENS uniqueness of the fixed point.
""")

print("="*74)
print("PART 3: WHEN IS THE FIXED POINT WELL-BEHAVED?")
print("="*74)
print("""
  Contraction (sup-norm) requires max row sum of |DT| < 1. Bounding entries:
      |DT_1j| <= b_F * phi * ( |Q'|*max(phi+B, phi) + Q*phi/gamma )
      |DT_2j| <= b_S * phi * ( |Q'|*max(phi+B, phi) + Q*phi/gamma )

  SUFFICIENT CONDITION (C'):
      2 * max(b_F,b_S) * phi * [ |Q'|*(phi + B) + Q*phi/gamma ]  <  1
  with B = beta_e*phi/(alpha*gamma).

  Reading:
   - b small  (elastic fuel supply)      -> easier
   - gamma large (steep SAF cost)        -> easier  (B falls AND Q*phi/gamma falls)
   - beta_e large (strong green premium) -> HARDER  (B rises)
  So a strong green preference is precisely what destabilizes the fuel-market
  fixed point. That is a genuine and reportable economic finding.
""")

print("="*74)
print("PART 4: GE POLICY EFFECT WITH THE CORRECTED JACOBIAN")
print("="*74)
dsstar_dsig = phi/gamma
dT_dsig = sp.Matrix([bF*phi*(-Qv*dsstar_dsig), bS*phi*(Qv*dsstar_dsig)])
I2 = sp.eye(2)
dP_dsig = sp.simplify((I2-DT).inv()*dT_dsig)
ds_total = sp.simplify(sp.together(dsstar_dsig + ds_F*dP_dsig[0] + ds_S*dP_dsig[1]))
print("  TOTAL ds*/dsigma (GE) =")
sp.pprint(ds_total)
print("\n  partial-equilibrium value = phi/gamma =", phi/gamma)
ratio = sp.simplify(sp.together(ds_total/(phi/gamma)))
print("\n  GE / PE ratio =")
sp.pprint(sp.simplify(ratio))
print("""
  If the ratio is in (0,1) the subsidy is partly captured by SAF producers
  (incidence). Evaluate numerically in the calibration to report pass-through.
""")

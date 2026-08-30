"""
DOES INSTRUMENT NON-EQUIVALENCE SURVIVE THE ACTUAL EU ETS BASE?

Motivation: the paper prices lifecycle emissions, tau*CI_j,
but the EU ETS zero-rates qualifying SAF (emission factor 0) and charges
kerosene at 3.16 tCO2/t. So the model's carbon price is a textbook lifecycle
tax, not the instrument used in the European calibration.

Generalise: \tilde P_j = P_j - sigma_j + tau*e_j, with e_j the REGULATORY
pricing factor and CI_j the lifecycle intensity used for damages.

  lifecycle benchmark : e_F = CI_F, e_b = CI_b, e_s = CI_s
  EU ETS as written   : e_F = CI_F, e_b = e_s = 0

Question: do Propositions 1 and 2 survive under the ETS row, and how do the
magnitudes change?
"""
import sympy as sp

tau, sigb, sigs = sp.symbols('tau sigma_b sigma_s', nonnegative=True)
phi, gb, gs, alpha = sp.symbols('phi gamma_b gamma_s alpha', positive=True)
eF, eb, es = sp.symbols('e_F e_b e_s', nonnegative=True)
CIF, CIb, CIs = sp.symbols('CI_F CI_b CI_s', positive=True)
PF, Pb, Ps = sp.symbols('P_F P_b P_s', positive=True)
ab, as_, bb, bs = sp.symbols('a_b a_s beta_b beta_s', nonnegative=True)
th, ths = sp.symbols('theta theta_s', positive=True)

# interior shares, Regime I, general regulatory base
Db = (Pb - sigb + tau*eb) - (PF + tau*eF)
Ds = (Ps - sigs + tau*es) - (PF + tau*eF)
sb = ab + (bb/alpha - phi*Db)/gb
ss = as_ + (bs/alpha - phi*Ds)/gs

print("="*78)
print("PART 1: BLENDING RESPONSES UNDER A GENERAL BASE")
print("="*78)
print("  ds_b/dtau     =", sp.simplify(sp.diff(sb, tau)), "   [= phi(e_F-e_b)/gamma_b]")
print("  ds_s/dtau     =", sp.simplify(sp.diff(ss, tau)))
print("  ds_b/dsigma_b =", sp.simplify(sp.diff(sb, sigb)))
print("""
  Under the LIFECYCLE base these are phi(CI_F-CI_j)/gamma_j: the carbon price
  discriminates by lifecycle advantage.
  Under the ETS base (e_b=e_s=0) they are phi*e_F/gamma_j for BOTH fuels: the
  carbon price still rewards SAF over fossil, but no longer discriminates
  BETWEEN SAF types. Support still rewards only its own fuel.
  => the fossil-vs-SAF asymmetry survives; the bio-vs-synthetic one does not.
""")

print("="*78)
print("PART 2: PROPOSITION 2 UNDER EACH BASE")
print("="*78)

def iso_cost_response(e_F, e_b, single=True):
    """ds*/dtau along the iso-average-cost locus, single-fuel version."""
    D = (Pb - sigb + tau*e_b) - (PF + tau*e_F)
    s_star = ab + (bb/alpha - phi*D)/gb
    AC = phi*((1-th)*(PF + tau*e_F) + th*(Pb - sigb + tau*e_b))
    w = sp.Symbol('w')
    direction = sp.solve(sp.Eq(sp.diff(AC, tau) + sp.diff(AC, sigb)*w, 0), w)[0]
    return sp.simplify(sp.diff(s_star, tau) + sp.diff(s_star, sigb)*direction)

r_life = iso_cost_response(CIF, CIb)
r_ets = iso_cost_response(eF, 0)
print("  lifecycle base : ds*/dtau|AC =", sp.factor(r_life))
print("  ETS base       : ds*/dtau|AC =", sp.factor(r_ets))
print("  ETS zero?      :", sp.simplify(r_ets) == 0)
print("""
  Under the lifecycle base this is CI_F*phi/(gamma*theta) -- the paper's eq.
  Under the ETS base it is e_F*phi/(gamma*theta), i.e. the SAME expression with
  the fossil emission factor in place of the fossil lifecycle intensity. Since
  the EU ETS charges kerosene at its full combustion factor, e_F is close to
  CI_F and the magnitude is essentially unchanged.

  => PROPOSITION 2 SURVIVES. It is not an artifact of lifecycle pricing. The
     driver is that the carbon price has ANY positive base on fossil fuel while
     support is paid per litre of SAF -- the bases differ regardless of whether
     SAF is zero-rated or charged its residual intensity.
""")

print("="*78)
print("PART 3: THE BLEND-NEUTRAL DIRECTION (COROLLARY 2.1)")
print("="*78)
x_b, x_s = sp.symbols('x_b x_s')
tb = sp.diff(sb, tau) + sp.diff(sb, sigb)*x_b
ts = sp.diff(ss, tau) + sp.diff(ss, sigs)*x_s
sol = sp.solve([sp.Eq(tb, 0), sp.Eq(ts, 0)], [x_b, x_s], dict=True)[0]
print("  general base: dsigma_b/dtau =", sp.simplify(sol[x_b]))
print("               dsigma_s/dtau =", sp.simplify(sol[x_s]))
print("""
  = -(e_F - e_j): support indexed to the REGULATORY advantage, not necessarily
  the lifecycle one. Under a lifecycle carbon price these coincide. Under the
  ETS they do not: the blend-neutral direction is a FLAT -e_F per unit tau for
  both fuels, because the ETS itself does not discriminate between them.

  POLICY READING, sharper than before: under zero-rating, the instrument that
  would restore blend-neutrality is a flat volumetric subsidy -- so ETS-style
  carbon pricing and volumetric support are CLOSER to equivalent on the
  composition margin than a lifecycle carbon price would be, while remaining
  non-equivalent on the fossil-vs-SAF margin and in cost.
""")

sb_f, ss_f = th - ths, ths
AC2 = phi*((1-sb_f-ss_f)*(PF+tau*eF) + sb_f*(Pb-sigb+tau*eb) + ths*(Ps-sigs+tau*es))
drift = sp.diff(AC2, tau) + sp.diff(AC2, sigb)*sol[x_b] + sp.diff(AC2, sigs)*sol[x_s]
print("  dAC/dtau along blend-neutral direction =", sp.simplify(sp.expand(drift)))
print("  (= phi*e_F: the carbon charge on the fossil share, unreachable by any")
print("   SAF-volume instrument. Same conclusion as before, general base.)")

print("="*78)
print("PART 4: NUMBERS")
print("="*78)
MJ = 43.0*0.8
CI_F_n = 89.0*MJ/1000        # kgCO2e/L lifecycle
E_F_n = 3.16*0.8             # kgCO2/L, ETS factor 3.16 t/t x 0.8 kg/L
print(f"  lifecycle CI_F = {CI_F_n:.4f} kgCO2e/L")
print(f"  ETS e_F        = {E_F_n:.4f} kgCO2/L   (3.16 tCO2/t fuel, 0.8 kg/L)")
ratio = E_F_n/CI_F_n
print(f"  ratio e_F/CI_F = {ratio:.3f}   ({100*(1-ratio):.0f}% lower)")
print(f"""
  The ETS factor is {100*(1-ratio):.0f}% BELOW the lifecycle intensity, because the ETS
  prices combustion CO2 (3.16 t/t) while the lifecycle figure (89 gCO2e/MJ)
  includes upstream extraction, refining and distribution.

  Consequences, all in the same direction:
    - the wedge derivative d(Delta)/dtau falls by {100*(1-ratio):.0f}%, so a given carbon
      price moves blending LESS than the lifecycle specification implies;
    - break-even carbon prices tau* rise by 1/{ratio:.3f} = {1/ratio:.2f}x, since a
      smaller base needs a higher price to close the same wedge;
    - Proposition 2's divergence e_F*phi/(gamma*theta) falls by {100*(1-ratio):.0f}%.

  Sign and ordering are unchanged everywhere. But the {100*(1-ratio):.0f}% gap is NOT
  negligible for the reported break-even levels: EUR 398/tCO2e for bio-SAF
  becomes about EUR {398/ratio:.0f}/tCO2e when the ETS base is priced rather than
  the lifecycle base. Report both, or state which base each number uses.
""")

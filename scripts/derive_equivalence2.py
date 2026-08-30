"""
PROPOSITION 2 IN THE TWO-FUEL MODEL.

Single-fuel benchmark underlying Proposition 2. The proposition is stated for
one SAF share, one subsidy and one mandate; the paper's model has two of each.
With two subsidies and one carbon price, holding average cost constant does not
pin down a unique direction: the answer depends on which support instrument
absorbs the compensation.

This script derives the iso-average-cost comparative static for four
economically distinct compensation rules, in Regime I (both constraints slack,
so both margins are interior and the question is well posed).

Notation: shares s_b, s_s; wedges Delta_j = P_j - sigma_j + tau*CI_j - P_F
- tau*CI_F; interior optima s_j* = a_j + (beta_j/alpha - phi*Delta_j)/gamma_j.
"""
import sympy as sp

tau, sigb, sigs = sp.symbols('tau sigma_b sigma_s', nonnegative=True)
phi, gb, gs, alpha = sp.symbols('phi gamma_b gamma_s alpha', positive=True)
CIF, CIb, CIs = sp.symbols('CI_F CI_b CI_s', positive=True)
PF, Pb, Ps = sp.symbols('P_F P_b P_s', positive=True)
ab, as_ = sp.symbols('a_b a_s', nonnegative=True)
bb, bs = sp.symbols('beta_b beta_s', nonnegative=True)
th, ths = sp.symbols('theta theta_s', positive=True)

# interior shares (Regime I)
Db = (Pb - sigb + tau*CIb) - (PF + tau*CIF)
Ds = (Ps - sigs + tau*CIs) - (PF + tau*CIF)
sb = ab + (bb/alpha - phi*Db)/gb
ss = as_ + (bs/alpha - phi*Ds)/gs

print("="*78)
print("PART 1: THE AVERAGE-COST OBJECT IN THE TWO-FUEL MODEL")
print("="*78)
# Average fuel cost per passenger, evaluated at the mandate floor shares
# (the natural reference: what a compliant airline pays)
sb_f, ss_f = th - ths, ths
AC = phi*((1 - sb_f - ss_f)*(PF + tau*CIF)
          + sb_f*(Pb - sigb + tau*CIb)
          + ss_f*(Ps - sigs + tau*CIs))
AC = sp.expand(AC)
dAC_dtau = sp.simplify(sp.diff(AC, tau))
dAC_dsigb = sp.simplify(sp.diff(AC, sigb))
dAC_dsigs = sp.simplify(sp.diff(AC, sigs))
print("  dAC/dtau     =", dAC_dtau)
print("  dAC/dsigma_b =", dAC_dsigb)
print("  dAC/dsigma_s =", dAC_dsigs)
print("""
  So the iso-average-cost SURFACE is 2-dimensional in (sigma_b, sigma_s) for
  each dtau: one equation, two unknowns. A DIRECTION must be chosen.
""")

# blending responses
dsb_dtau = sp.diff(sb, tau); dss_dtau = sp.diff(ss, tau)
dsb_dsigb = sp.diff(sb, sigb); dss_dsigs = sp.diff(ss, sigs)
print("  ds_b/dtau     =", sp.simplify(dsb_dtau), "   (= phi(CI_F-CI_b)/gamma_b)")
print("  ds_s/dtau     =", sp.simplify(dss_dtau))
print("  ds_b/dsigma_b =", sp.simplify(dsb_dsigb))
print("  ds_s/dsigma_s =", sp.simplify(dss_dsigs))
print("  ds_b/dsigma_s =", sp.simplify(sp.diff(sb, sigs)), "  (Regime I: separation)")

def compensated(rule_name, dsigb_dtau, dsigs_dtau, describe):
    """total derivative of each share along the stated direction"""
    tot_b = sp.simplify(dsb_dtau + dsb_dsigb*dsigb_dtau + sp.diff(sb, sigs)*dsigs_dtau)
    tot_s = sp.simplify(dss_dtau + dss_dsigs*dsigs_dtau + sp.diff(ss, sigb)*dsigb_dtau)
    tot_total = sp.simplify(tot_b + tot_s)
    print("-"*78)
    print(f"  RULE: {rule_name}")
    print(f"        {describe}")
    print("    ds_b/dtau|AC =", sp.factor(sp.simplify(tot_b)))
    print("    ds_s/dtau|AC =", sp.factor(sp.simplify(tot_s)))
    print("    d(s_b+s_s)/dtau|AC =", sp.factor(sp.simplify(tot_total)))
    return tot_b, tot_s, tot_total

print("="*78)
print("PART 2: FOUR COMPENSATION RULES")
print("="*78)

# Rule A: only bio support adjusts (sigma_s fixed)
A = sp.solve(sp.Eq(dAC_dtau + dAC_dsigb*sp.Symbol('x'), 0), sp.Symbol('x'))[0]
compensated("A. bio-only compensation",
            A, 0,
            "sigma_s held fixed; sigma_b absorbs the whole compensation")

# Rule B: only synthetic support adjusts
B = sp.solve(sp.Eq(dAC_dtau + dAC_dsigs*sp.Symbol('x'), 0), sp.Symbol('x'))[0]
compensated("B. synthetic-only compensation",
            0, B,
            "sigma_b held fixed; sigma_s absorbs the whole compensation")

# Rule C: equal per-litre compensation, dsigma_b = dsigma_s = z
z = sp.Symbol('z')
C = sp.solve(sp.Eq(dAC_dtau + dAC_dsigb*z + dAC_dsigs*z, 0), z)[0]
compensated("C. uniform per-litre compensation",
            C, C,
            "both support rates rise equally (the natural 'volumetric' rule)")

# Rule D: compensation indexed to lifecycle savings, dsigma_j = rho*(CI_F - CI_j)
rho = sp.Symbol('rho')
D_rho = sp.solve(sp.Eq(dAC_dtau + dAC_dsigb*rho*(CIF-CIb) + dAC_dsigs*rho*(CIF-CIs), 0), rho)[0]
tb, ts, tt = compensated("D. intensity-indexed compensation (CfD-style)",
            D_rho*(CIF-CIb), D_rho*(CIF-CIs),
            "dsigma_j = rho*(CI_F - CI_j): support paid per tonne avoided")

print("="*78)
print("PART 3: THE HEADLINE -- DOES RULE D RESTORE EQUIVALENCE?")
print("="*78)
print("  Under rule D, d(s_b+s_s)/dtau|AC =", sp.simplify(tt))
print("  Is it zero?", sp.simplify(tt) == 0)
print("""
  If nonzero, intensity indexing does NOT fully restore equivalence in the
  two-fuel model, because the carbon price and the indexed subsidy still
  reach the two fuels through different curvatures gamma_b, gamma_s.
  Equivalence on the TOTAL margin requires gamma_b = gamma_s as well.
  That is a sharper and more honest statement than the draft's current claim.
""")
print("  Under gamma_b = gamma_s = gamma:")
tt_eq = sp.simplify(tt.subs(gs, gb))
print("    d(s_b+s_s)/dtau|AC =", sp.simplify(tt_eq))
print("    zero?", sp.simplify(tt_eq) == 0)

print("="*78)
print("PART 4: SINGLE-FUEL COLLAPSE -- RECOVER THE PAPER'S EQ (12)")
print("="*78)
# one fuel: set theta_s = 0, drop synthetic entirely
sb_only = ab + (bb/alpha - phi*((Pb - sigb + tau*CIb) - (PF + tau*CIF)))/gb
AC1 = phi*((1-th)*(PF + tau*CIF) + th*(Pb - sigb + tau*CIb))
dir1 = sp.solve(sp.Eq(sp.diff(AC1, tau) + sp.diff(AC1, sigb)*sp.Symbol('w'), 0), sp.Symbol('w'))[0]
tot1 = sp.simplify(sp.diff(sb_only, tau) + sp.diff(sb_only, sigb)*dir1)
print("  ds*/dtau|AC (single fuel) =", sp.simplify(sp.factor(tot1)))
print("  paper's eq (12): CI_F*phi/(gamma*theta)")
print("  match?", sp.simplify(tot1 - CIF*phi/(gb*th)) == 0)

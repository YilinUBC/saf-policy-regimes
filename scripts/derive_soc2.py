"""
RECONCILE THE TWO SECOND-ORDER CONDITIONS.

The issue: eq (7) and eq (9) in the draft are not
algebraically equivalent, yet Appendix A says one is a rearrangement of the
other. Which is right?

  eq (7):  kappa > beta_f^2 * q * (M - N q)^2 / (alpha * (M - N q + q)^2)
  eq (9):  |H2| = kappa/(alpha q) - beta_f^2/alpha^2 > 0
           <=>  kappa > beta_f^2 q / alpha

They differ by the factor R = (M - N q)^2/(M - N q + q)^2 < 1, so (7) is
strictly WEAKER than (9).

Resolution: they come from DIFFERENT models of the outside option.
  derive_soc.py        : q0 held FIXED       -> gives (9)
  derive_uniqueness2.py: q0 = M - N q ENDOG. -> gives (7)
The paper's model has an endogenous outside option (Section 3.2), so (7) is
the correct condition and the |H2| expression printed as (9) is the
fixed-q0 special case. This script proves the nesting.
"""
import sympy as sp

M, N = sp.symbols('M N', positive=True)
alpha, beta_f, kappa, gamma = sp.symbols('alpha beta_f kappa gamma', positive=True)
q, f, s = sp.symbols('q f s', positive=True)
phi = sp.symbols('phi', positive=True)
beta_e, theta = sp.symbols('beta_e theta', nonnegative=True)
PF, PS_ = sp.symbols('P_F P_S', positive=True)

print("="*78)
print("PART 1: FIXED OUTSIDE OPTION  ->  the eq (9) form")
print("="*78)

q0 = sp.symbols('q0', positive=True)
p_fix = (beta_f*f + beta_e*s - sp.log(q/q0))/alpha
c = phi*((1-s)*PF + s*PS_) + gamma/2*(s-theta)**2
pi_fix = (p_fix - c)*q - kappa/2*f**2

H_fix = sp.hessian(pi_fix, (q, f, s))
H2_fix = sp.simplify(H_fix[:2, :2].det())
print("  H11  =", sp.simplify(H_fix[0, 0]))
print("  |H2| =", sp.simplify(H2_fix))
crit_fix = sp.solve(sp.Eq(H2_fix, 0), kappa)[0]
print("  |H2| > 0  <=>  kappa >", sp.simplify(crit_fix))

print("""
  With q0 FIXED, dp/dq = -1/(alpha q), so H11 = -1/(alpha q) and
  |H2| = kappa/(alpha q) - beta_f^2/alpha^2. This is exactly eq (9).
""")

print("="*78)
print("PART 2: ENDOGENOUS OUTSIDE OPTION  ->  the eq (7) form")
print("="*78)

# q0 = M - N q : every passenger not flying on one of the N carriers
p_end = (beta_f*f + beta_e*s - sp.log(q/(M - N*q)))/alpha
pi_end = (p_end - c)*q - kappa/2*f**2

H_end = sp.hessian(pi_end, (q, f, s))
H11_end = sp.simplify(H_end[0, 0])
H2_end = sp.simplify(H_end[:2, :2].det())
print("  H11  =", H11_end)
print("  |H2| =", sp.simplify(sp.factor(H2_end)))
crit_end = sp.solve(sp.Eq(sp.numer(sp.together(H2_end)), 0), kappa)[0]
crit_end = sp.simplify(crit_end)
print("\n  |H2| > 0  <=>  kappa >", crit_end)

draft = beta_f**2*q*(M - N*q)**2/(alpha*(M - N*q + q)**2)
print("  eq (7) as PRINTED in the draft:", draft)
print("  difference from truth         :", sp.simplify(crit_end - draft))
print("  identical?                    :", sp.simplify(crit_end - draft) == 0)
print("""
  => eq (7) AS PRINTED IS ALSO WRONG. Its denominator is (M - N q + q)^2;
     the correct denominator is M^2. They coincide only if N = 1.
""")

print("="*78)
print("PART 3: THE NESTING, AND THE CLEAN FORM")
print("="*78)
R = sp.simplify(crit_end/(beta_f**2*q/alpha))
print("  ratio (endogenous critical kappa)/(fixed-q0 critical kappa) =")
sp.pprint(sp.factor(R))
print("""
  R = (M - N q)^2 / M^2 = varsigma_0^2, the OUTSIDE SHARE squared.

  So the correct condition has a clean reading:

      kappa  >  (beta_f^2 * q / alpha) * varsigma_0^2                (7')

  i.e. the fixed-q0 threshold of eq (9) DISCOUNTED by the square of the
  outside share. Since varsigma_0 < 1, condition (7') is strictly weaker
  than (9): endogenous substitution to the outside option flattens residual
  demand and makes concavity EASIER, not harder.

  Consequences for the draft:
    - eq (7) must be replaced by (7').
    - Appendix A's claim that (9) is a rearrangement of (7) is false.
      (9) is the fixed-q0 special case; (7') is the operative condition in
      the paper's model, and (9) => (7') but not conversely.
""")

# Verify the sign claim numerically at a representative point
sub = {M: 1000.0, N: 3.0, q: 80.0, alpha: 0.0156, beta_f: 0.05}
print("  numerical check at M=1000, N=3, q=80, alpha=0.0156, beta_f=0.05:")
print("    fixed-q0 critical kappa      =", float((beta_f**2*q/alpha).subs(sub)))
print("    endogenous critical kappa    =", float(crit_end.subs(sub)))
print("    ratio R                      =", float(R.subs(sub)))

print("="*78)
print("PART 4: DOES det H = -gamma*q*|H2| STILL HOLD?")
print("="*78)
detH_end = sp.simplify(H_end.det())
print("  det H + gamma*q*|H2| =", sp.simplify(detH_end + gamma*q*H2_end))
print("  (0 confirms the block-diagonal factorisation survives)")

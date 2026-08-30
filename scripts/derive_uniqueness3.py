"""
PROPOSITION 2: does the reduction to a standard logit price game actually work?

The issue: block-diagonality of the Hessian is a LOCAL
property at an optimum. It does not by itself show that the joint problem
reduces globally to a fixed-characteristics logit price game, because

    f* = beta_f * q / (alpha * kappa)

depends on equilibrium traffic and therefore feeds back into utility and
market shares. Substituting f*(q) changes the effective own-price derivative,
so Caplin-Nalebuff / Mizuno cannot simply be invoked on the reduced game
without checking what the substitution does.

This script does the check.
"""
import sympy as sp

alpha, bf, kap, gam, phi = sp.symbols('alpha beta_f kappa gamma phi', positive=True)
be = sp.symbols('beta_e', nonnegative=True)
M, N = sp.symbols('M N', positive=True)
p, mc = sp.symbols('p mc', positive=True)
vs, vs0 = sp.symbols('varsigma varsigma_0', positive=True)

print("="*78)
print("PART 1: WHAT f* SUBSTITUTION DOES TO MEAN UTILITY")
print("="*78)
print("""
  Mean utility  d = beta_f f + beta_e s - alpha p  (+ xi).
  Optimal frequency  f* = beta_f q /(alpha kappa)  with q = M*varsigma.
  Substituting,
      d(p, varsigma) = beta_f^2 M varsigma /(alpha kappa) + beta_e s - alpha p

  So mean utility depends on the airline's OWN share -- the model becomes a
  fixed point in varsigma even holding rivals fixed. This is exactly the
  feedback described above. Define
      G(varsigma) = beta_f^2 M varsigma /(alpha kappa)
  the 'frequency feedback' term, with slope
""")
G_slope = bf**2*M/(alpha*kap)
print("      dG/dvarsigma =", G_slope)

print("="*78)
print("PART 2: THE OWN-SHARE FIXED POINT")
print("="*78)
print("""
  For given rival utilities, the logit share solves
      varsigma = exp(G(varsigma) + beta_e s - alpha p) / D
  where D = 1 + sum_j exp(d_j) is taken as given by the firm.

  Write varsigma = A * exp(G(varsigma)) with A = exp(beta_e s - alpha p)/D.
  A solution exists and is UNIQUE iff the map T(varsigma) = A exp(G(varsigma))
  is a contraction on (0,1), i.e. iff

      |T'(varsigma)| = A exp(G) * dG/dvarsigma = varsigma * dG/dvarsigma < 1

  Since varsigma < 1, a SUFFICIENT condition is dG/dvarsigma < 1:
""")
cond = sp.Eq(bf**2*M/(alpha*kap), 1)
kappa_crit = sp.solve(cond, kap)[0]
print("      beta_f^2 M /(alpha kappa) < 1   <=>   kappa > beta_f^2 M / alpha")
print("      critical kappa =", kappa_crit)

print("""
  COMPARE with the paper's condition (11):
      kappa > (beta_f^2 q / alpha) * varsigma_0^2
  Since q = M*varsigma and varsigma*varsigma_0^2 < 1, the paper's condition is
  WEAKER than what the share fixed point needs in the worst case. They coincide
  only as varsigma -> 1, varsigma_0 -> 1, which cannot both happen.
""")
print("  ratio (share-FP requirement)/(SOC requirement) = 1/(varsigma*varsigma_0^2)")
print("  at varsigma=0.25, varsigma_0=0.5 :",
      float(1/(0.25*0.5**2)), " -- share FP needs a 16x larger kappa")

print("="*78)
print("PART 3: SO WHAT IS THE HONEST STATEMENT?")
print("="*78)
print("""
  Two distinct conditions are in play:

   (A) SOC / own-concavity, condition (11):
           kappa > (beta_f^2 q/alpha) varsigma_0^2
       -- guarantees each firm's problem is well behaved at an interior point.

   (B) Global uniqueness of the own-share fixed point induced by the
       frequency feedback:
           kappa > beta_f^2 M / alpha
       -- guarantees the substitution f*(q) does not create multiple
          own-share solutions, so the reduced price game is well defined.

  (B) implies (A) since q*varsigma_0^2 < M. So imposing (B) is enough for both,
  and ONLY under (B) is the reduction to a standard logit price game
  legitimate. Under (A) alone, the correct claim is existence plus LOCAL
  uniqueness.

  RECOMMENDATION for the paper: state (B) as the uniqueness condition, note
  that it implies (A), and keep (A) as the weaker requirement for concavity
  alone. Then the appeal to Caplin-Nalebuff and Mizuno is legitimate, because
  under (B) mean utility is a well-defined function of (p, rivals) alone.
""")

print("="*78)
print("PART 4: IS (B) RESTRICTIVE AT CALIBRATED VALUES?")
print("="*78)
ALPHA = 1.4/(120.0*0.75)
print(f"  alpha = {ALPHA:.5f}/EUR")
for Mv in (1000.0, 5000.0):
    for bfv in (0.02, 0.05, 0.10):
        kmin = bfv**2*Mv/ALPHA
        print(f"    M={Mv:6.0f}, beta_f={bfv:.2f}  ->  kappa > {kmin:10.1f}")
print("""
  kappa is the coefficient on (kappa/2) f^2 in EUR per flight-squared, i.e. the
  curvature of schedule cost. These bounds are modest relative to observed
  short-haul operating costs, and kappa is swept rather than calibrated
  (Section 7.3), so no reported result depends on where in the admissible
  region it sits. The condition is transparent and checkable, which is what
  the paper should claim -- no more.
""")

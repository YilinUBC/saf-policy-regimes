"""
Sign the dominance condition properly, and re-check the SOC now that the
outside option is endogenous (q0 = M - N q), which changes the own-effect
relative to derive_soc.py where q0 was held fixed.

Substitute q = x*M/N  with x in (0,1) = industry share of the market
(so q0 = M(1-x)). This is the natural normalization and makes signs readable.
"""
import sympy as sp

M, N = sp.symbols('M N', positive=True)
alpha, beta_f, kappa, gamma, q = sp.symbols('alpha beta_f kappa gamma q', positive=True)
x = sp.symbols('x', positive=True)          # aggregate market penetration = N q / M

print("=" * 72)
print("PART 1: SIGN THE DOMINANCE CONDITION")
print("=" * 72)

dom = (M**2 - M*N**2*q + M*q + N**3*q**2 - 2*N**2*q**2 + N*q**2) / (alpha*q*(M**2 - 2*M*N*q + N**2*q**2))
dom = sp.simplify(dom)
print("\n  raw dominance expression:")
sp.pprint(dom)

# denominator: alpha*q*(M - N q)^2 > 0 always. So sign = sign(numerator).
num = sp.numer(sp.together(dom))
den = sp.denom(sp.together(dom))
print("\n  denominator factored:", sp.factor(den), "  -> positive (since q<M/N)")
print("  numerator factored  :")
sp.pprint(sp.factor(num))

# substitute q = x M / N
num_x = sp.simplify(sp.expand(num.subs(q, x*M/N)))
print("\n  numerator with q = x*M/N :")
sp.pprint(sp.simplify(sp.factor(num_x)))

# collect: divide by M^2 (positive) to normalize
num_norm = sp.simplify(sp.expand(num_x / M**2))
print("\n  normalized (divided by M^2):")
sp.pprint(sp.simplify(sp.factor(num_norm)))

print("\n  Test at representative values:")
for Nv in [2, 3, 5]:
    for xv in [sp.Rational(1,10), sp.Rational(1,2), sp.Rational(8,10), sp.Rational(95,100)]:
        val = num_norm.subs({N: Nv, x: xv, M: 1})
        print(f"    N={Nv}, x={float(xv):.2f}  ->  {sp.nsimplify(sp.simplify(val))} = {float(sp.simplify(val)):+.4f}")

print("""
  INTERPRETATION: if this is positive for all x in (0,1) and N>=2, the
  q-block is strictly diagonally dominant => unique equilibrium (Rosen 1965).
""")

# ---------------------------------------------------------------------------
print("=" * 72)
print("PART 2: RE-DERIVE THE SOC WITH ENDOGENOUS OUTSIDE OPTION")
print("=" * 72)
print("""
  In derive_soc.py the own-effect was d2pi/dq2 = -1/(alpha q), computed
  holding q0 FIXED. With q0 = M - N q endogenous the own-effect becomes
  the expression below. Condition (A) must be restated.
""")

qi = sp.symbols('q_i', positive=True)
fi, si = sp.symbols('f_i s_i', positive=True)
beta_e, phi, theta, Delta, PFt = sp.symbols('beta_e phi theta Delta Ptilde_F', positive=True)

q0_gen = M - qi - (N-1)*q
p_gen = (beta_f*fi + beta_e*si - sp.log(qi/q0_gen))/alpha
c_fuel = phi*(PFt + si*Delta)
c_saf = gamma/2*(si-theta)**2
pi_gen = (p_gen - c_fuel - c_saf)*qi - kappa/2*fi**2

# full Hessian in (q_i, f_i, s_i)
H = sp.Matrix([[sp.diff(pi_gen, a, b) for b in (qi, fi, si)] for a in (qi, fi, si)])
H_sym = sp.simplify(H.subs(qi, q))

print("  Hessian (symmetric point), entries:")
nm = ['q','f','s']
for i in range(3):
    for j in range(i,3):
        print(f"    d2pi/d{nm[i]}d{nm[j]} =", sp.simplify(H_sym[i,j]))

# the (q,s) cross-partial should STILL vanish at the interior s-optimum
sFOC = beta_e/alpha - phi*Delta - gamma*(si-theta)
s_star = sp.solve(sp.Eq(sFOC,0), si)[0]
H_opt = sp.simplify(H_sym.subs(si, s_star))
print("\n  Hessian at interior s*:")
sp.pprint(sp.simplify(H_opt))

m1 = sp.simplify(H_opt[0,0])
m2 = sp.simplify(sp.expand(H_opt[:2,:2].det()))
m3 = sp.simplify(sp.expand(H_opt.det()))
print("\n  H11  =", sp.simplify(m1))
print("  |H2| =", sp.simplify(sp.factor(m2)))
print("  detH =", sp.simplify(sp.factor(m3)))

# express own-effect with q = x M / N
own_x = sp.simplify(m1.subs(q, x*M/N))
print("\n  H11 with q = x*M/N :", sp.simplify(sp.factor(own_x)))

m2_x = sp.simplify(sp.factor(m2.subs(q, x*M/N)))
print("  |H2| with q = x*M/N :")
sp.pprint(m2_x)

print("""
  RESTATED CONDITION (A'):  |H2| > 0  <=>
      kappa * (own-price-effect magnitude) > beta_f^2/alpha^2
  where the own-price effect now depends on market penetration x.
""")

# solve |H2|>0 for kappa
kappa_min = sp.solve(sp.Eq(m2, 0), kappa)
print("  critical kappa (|H2|=0):", sp.simplify(kappa_min[0]) if kappa_min else "none")
print("  => need kappa >", sp.simplify(sp.factor(kappa_min[0])) if kappa_min else "")

# check detH = -gamma*q*|H2| still holds
print("\n  detH + gamma*q*|H2| =", sp.simplify(sp.expand(m3 + gamma*q*m2)))
print("  (0 confirms detH = -gamma*q*|H2| survives the endogenous outside option)")

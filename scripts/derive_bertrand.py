"""
BERTRAND-LOGIT: verify the markup identity and the SAF margin properly.

Fix vs derive_blp.py: do NOT ask sympy to solve the FOC for p_i (p_i sits
inside the share -> LambertW). Instead substitute the share as a symbol,
verify the FOC reduces to the standard identity, then handle the s-margin
via the envelope/implicit-function route rather than closed-form solve.
"""
import sympy as sp

M, N = sp.symbols('M N', positive=True)
alpha, beta_f, beta_e = sp.symbols('alpha beta_f beta_e', positive=True)
kappa, gamma, phi = sp.symbols('kappa gamma phi', positive=True)
pi_, pj_ = sp.symbols('p_i p_j', positive=True)
fi, si = sp.symbols('f_i s_i', positive=True)
theta, Delta, PFt = sp.symbols('theta Delta Ptilde_F', positive=True)
sh = sp.symbols('sigma_i', positive=True)     # own share as a symbol

print("=" * 74)
print("PART 1: BERTRAND-LOGIT MARKUP IDENTITY")
print("=" * 74)

d_i = beta_f*fi + beta_e*si - alpha*pi_
d_j = beta_f*fi + beta_e*si - alpha*pj_
D = 1 + sp.exp(d_i) + (N-1)*sp.exp(d_j)
share_i = sp.exp(d_i)/D

mc = phi*(PFt + si*Delta) + gamma/2*(si-theta)**2
profit = (pi_ - mc)*M*share_i - kappa/2*fi**2

FOC = sp.diff(profit, pi_)

# key logit fact: d(share_i)/d(p_i) = -alpha * share_i * (1 - share_i)
dshare = sp.simplify(sp.diff(share_i, pi_))
claim  = -alpha*share_i*(1-share_i)
print("  d(share_i)/dp_i  +  alpha*share_i*(1-share_i)  =",
      sp.simplify(sp.expand(dshare - claim)))
print("  -> 0 confirms the standard logit derivative.\n")

# FOC = M*[share + (p-mc)*dshare] = 0  =>  1 - alpha(p-mc)(1-share) = 0
FOC_norm = sp.simplify(FOC/(M*share_i))
FOC_sub  = sp.simplify(FOC_norm.subs(share_i, sh))
print("  FOC / (M*share_i) =")
sp.pprint(sp.simplify(FOC_norm))

resid = sp.simplify(sp.expand(FOC_norm - (1 - alpha*(pi_-mc)*(1-share_i))))
print("\n  FOC/(M*s_i) - [1 - alpha*(p-mc)*(1-share_i)] =", sp.simplify(resid))
print("""
  => p_i - mc_i = 1/(alpha*(1 - share_i))            [STANDARD LOGIT MARKUP]

  Bounded, depends only on own share, and the Bertrand-logit best-response
  map is a contraction (Caplin-Nalebuff 1991; Mizuno 2003). Existence AND
  uniqueness are citable off-the-shelf results -- no Gale-Nikaido needed.
""")

print("=" * 74)
print("PART 2: THE SAF MARGIN UNDER BERTRAND")
print("=" * 74)
print("""
  Under Bertrand with the markup identity, at the optimum
      p_i = mc_i + 1/(alpha(1-share_i)).
  The s-margin then trades off (i) demand shift beta_e and (ii) mc shift.
  Use the ENVELOPE argument: dprofit/ds at the optimal price.
""")

# envelope: profit at optimum = (markup)*M*share; ds affects mc and d_i
mu = 1/(alpha*(1-sh))                       # markup, treating share as sigma_i
dmc_ds = sp.simplify(sp.diff(mc, si))
print("  dmc/ds       =", dmc_ds)
print("  d(mean utility)/ds = beta_e")
print("""
  Marginal profit in s (envelope, holding p at its optimum):
     dPi/ds = M*[ (dshare/ds)*mu  -  share*(dmc/ds) ]
  With logit,  dshare/ds = share*(1-share)*beta_e.
  Substituting mu = 1/(alpha(1-share)):
     dPi/ds = M*share*[ beta_e/alpha  -  dmc/ds ]
  Setting = 0:
""")
foc_s = beta_e/alpha - dmc_ds
s_star = sp.solve(sp.Eq(foc_s, 0), si)[0]
print("  s* =", sp.simplify(sp.expand(s_star)))

target = theta + (beta_e/alpha - phi*Delta)/gamma
print("  s* - [theta + (beta_e/alpha - phi*Delta)/gamma] =",
      sp.simplify(sp.expand(s_star - target)))
print("""
  => 0. THE INTERIOR SAF SHARE IS IDENTICAL UNDER COURNOT AND BERTRAND.

  This is a strong robustness result: the share-weighting (1-share) in the
  markup exactly cancels the (1-share) in the logit share derivative, so the
  SAF margin is governed purely by  beta_e/alpha  vs  dmc/ds. Observation 1
  therefore does NOT depend on the mode of competition.
""")

print("=" * 74)
print("PART 3: OBSERVATION 1 UNDER BERTRAND")
print("=" * 74)
tau, sigma_pol = sp.symbols('tau sigma', nonnegative=True)
CI_F, CI_S = sp.symbols('CI_F CI_S', positive=True)
PF, PS = sp.symbols('P_F P_S', positive=True)
Delta_expl = (PS - sigma_pol + tau*CI_S) - (PF + tau*CI_F)
s_expl = theta + (beta_e/alpha - phi*Delta_expl)/gamma

ds_dsig = sp.simplify(sp.diff(s_expl, sigma_pol))
ds_dtau = sp.simplify(sp.diff(s_expl, tau))
print("  ds*/dsigma =", ds_dsig)
print("  ds*/dtau   =", ds_dtau)

AC = phi*((1-theta)*(PF+tau*CI_F) + theta*(PS-sigma_pol+tau*CI_S))
iso = sp.simplify(-sp.diff(AC,tau)/sp.diff(AC,sigma_pol))
along = sp.simplify(sp.expand(ds_dtau + ds_dsig*iso))
print("  iso-AC slope dsigma/dtau =", sp.simplify(iso))
print("  ds*/dtau along iso-AC    =", sp.simplify(along))
print("""
  Same expression as under Cournot:  CI_F*phi/(gamma*theta) > 0.
  Observation 1 is robust to the competition mode.
""")

print("=" * 74)
print("PART 4: FREQUENCY MARGIN UNDER BERTRAND")
print("=" * 74)
print("""
  dPi/df = M*[ (dshare/df)*mu ] - kappa*f,  dshare/df = share*(1-share)*beta_f
         = M*share*beta_f/alpha - kappa*f
  => f* = M*share*beta_f/(alpha*kappa)
  Interior, positive, and increasing in own share. Concave in f since
  d2Pi/df2 = -kappa < 0 holding share fixed.
""")
f_star = M*sh*beta_f/(alpha*kappa)
print("  f* =", f_star)
print("""
  NOTE: under Cournot we had f* = beta_f*q/(alpha*kappa) with q = M*share,
  i.e. THE SAME expression. The frequency margin is also mode-invariant.
""")

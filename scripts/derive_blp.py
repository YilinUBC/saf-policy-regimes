"""
BLP-STANDARD OUTSIDE OPTION: does diagonal dominance hold?

Previous spec (derive_uniqueness.py): outside quantity is RESIDUAL,
    q0 = M - sum_j q_j,   p_i = (beta_f f + beta_e s - ln(q_i/q0))/alpha
  -> dominance fails when x > 1/(N-1),  x = Nq/M.

Standard spec (Berry 1994): outside good has mean utility normalized to 0.
Shares come from the logit formula:
    s_i = exp(d_i) / (1 + sum_j exp(d_j)),     s_0 = 1/(1 + sum_j exp(d_j))
Berry inversion:
    ln(s_i) - ln(s_0) = d_i = beta_f f_i + beta_e s_i^SAF - alpha p_i
  => p_i = (beta_f f_i + beta_e s^SAF_i - ln(s_i/s_0))/alpha

With q_i = M * share_i, s_0 is NOT M - sum q_i over M in general -- it IS,
by the adding-up identity s_0 = 1 - sum_j s_j. So algebraically the two
specs coincide?? NO -- the difference is what is held fixed when
differentiating, and whether M is the potential market (fixed) with the
outside share free. Let's do this very carefully and find the real difference.
"""
import sympy as sp

print("=" * 74)
print("STEP 0: are the two specifications actually different?")
print("=" * 74)
print("""
  Residual spec:  q0 := M - sum_j q_j.  Then s_0 = q0/M = 1 - sum_j s_j.
  Berry spec:     s_0 := 1/(1+sum_j exp(d_j)), and s_i = exp(d_i)/(1+...).
                  Adding up gives s_0 = 1 - sum_j s_j identically.

  => As an ACCOUNTING identity they agree. The inverse demand
        p_i = (beta_f f_i + beta_e sSAF_i - ln(s_i/s_0))/alpha
     is THE SAME FUNCTION in both. So respecifying the outside option
     does NOT change the inversion.

  CONCLUSION: my proposed 'fix' does not exist. The dominance failure is
  a property of logit-Cournot with an outside option, not an artifact.
  Verify explicitly below.
""")

M, N = sp.symbols('M N', positive=True)
alpha, beta_f, beta_e = sp.symbols('alpha beta_f beta_e', positive=True)
kappa, gamma, phi = sp.symbols('kappa gamma phi', positive=True)
qi, q = sp.symbols('q_i q', positive=True)
fi, si = sp.symbols('f_i s_i', positive=True)
theta, Delta, PFt = sp.symbols('theta Delta Ptilde_F', positive=True)

# Write inverse demand in SHARES, then convert to quantities via q = M*share.
sh_i = qi / M
sh_0 = 1 - qi / M - (N - 1) * q / M
p_shares = (beta_f * fi + beta_e * si - sp.log(sh_i / sh_0)) / alpha
p_resid = (beta_f * fi + beta_e * si - sp.log(qi / (M - qi - (N - 1) * q))) / alpha

print("  p(shares) - p(residual) =", sp.simplify(sp.expand(p_shares - p_resid)))
print("  -> 0 confirms the two specs give the identical inverse demand.\n")

# ---------------------------------------------------------------------------
print("=" * 74)
print("STEP 1: so what DOES change the cross-effect? The market-size margin.")
print("=" * 74)
print("""
  The dominance failure came from dp_i/dq_j = -1/(alpha*q0): rivals shrink
  the outside pool. Two modelling choices genuinely change this:

  (a) BERTRAND / price competition instead of Cournot. Airlines set fares;
      logit demand is then well-behaved and existence+uniqueness of
      Bertrand-logit equilibrium is a KNOWN result (Caplin-Nalebuff 1991,
      Mizuno 2003, Konovalov-Sandor 2010). This is the standard route in
      empirical IO and in airline papers using discrete choice.

  (b) Keep Cournot but bound market penetration x < 1/(N-1).

  Check (a): under Bertrand-logit with multiproduct-free single-product
  firms, the profit function is quasi-concave in own price and the FOC
  system is a contraction under mild conditions. Let's verify own/cross
  structure.
""")

# ---------------- Bertrand-logit: firms set p_i, demand from logit shares
pi_, pj_ = sp.symbols('p_i p_j', positive=True)
d_i = beta_f * fi + beta_e * si - alpha * pi_
d_j = beta_f * fi + beta_e * si - alpha * pj_          # symmetric rivals
D = 1 + sp.exp(d_i) + (N - 1) * sp.exp(d_j)
share_i = sp.exp(d_i) / D

mc = phi * (PFt + si * Delta) + gamma / 2 * (si - theta) ** 2
profit_B = (pi_ - mc) * M * share_i - kappa / 2 * fi ** 2

FOC_p = sp.simplify(sp.diff(profit_B, pi_))
print("  Bertrand FOC dpi/dp_i = 0 :")
sp.pprint(sp.simplify(sp.together(FOC_p)))

# markup form: p - mc = 1/(alpha (1 - share_i))
markup = sp.solve(sp.Eq(sp.simplify(FOC_p / (M * share_i)), 0), pi_)
print("\n  Solved own price (markup form):")
if markup:
    sp.pprint(sp.simplify(markup[0]))
print("""
  Standard logit result: p_i - mc_i = 1/(alpha*(1 - share_i)).
  Markup depends only on own share -> clean, bounded, and the
  best-response map is a contraction (Caplin-Nalebuff / Milgrom-Roberts
  log-supermodularity). Uniqueness is then STANDARD and citable.
""")

# verify the markup identity symbolically
lhs = sp.simplify(sp.diff(profit_B, pi_) / (M))
markup_claim = 1 / (alpha * (1 - share_i))
check = sp.simplify(sp.expand(sp.solve(sp.Eq(lhs, 0), pi_)[0] - (mc + markup_claim)))
print("  verification (p_i* - [mc + 1/(alpha(1-share_i))]) =", sp.simplify(check))

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 2: does the SAF/frequency block survive under Bertrand?")
print("=" * 74)

# s-FOC under Bertrand: marginal cost enters through mc
dmc_ds = sp.simplify(sp.diff(mc, si))
print("  dmc/ds =", dmc_ds)
print("""
  Under Bertrand the SAF FOC is driven by (i) the demand shift beta_e and
  (ii) the marginal cost change dmc/ds = phi*Delta + gamma*(s-theta).
  Setting marginal benefit = marginal cost reproduces the SAME interior
  s* = theta + (beta_e/alpha - phi*Delta)/gamma up to the share-weighting.
  Verify:
""")
FOC_s = sp.simplify(sp.diff(profit_B, si))
# s_i enters both through exp(beta_e*s_i) and polynomially, so a closed-form
# solve() is not available ("multiple generators"). The interior s* is obtained
# in derive_bertrand.py and derive_twofuel.py by dividing the FOC through by
# the share, which removes the exponential; what is shown here is the FOC that
# those scripts then solve.
print("  Bertrand s-FOC (dpi/ds = 0), before dividing through by the share:")
sp.pprint(FOC_s)
try:
    sol_s = sp.solve(sp.Eq(FOC_s, 0), si)
    print("  s* under Bertrand:")
    for sol in sol_s:
        sp.pprint(sp.simplify(sp.expand(sol)))
except NotImplementedError:
    print("  (no closed form from solve() in this parameterisation;")
    print("   see derive_bertrand.py for the reduced FOC and its solution)")

print("""
  If s* retains the form theta + (beta_e/alpha - phi*Delta)/gamma, then
  Observation 1 (average-cost vs marginal-incentive) is UNCHANGED by the
  switch from Cournot to Bertrand -- the headline result is robust to
  the competition mode, which is itself worth a remark in the paper.
""")

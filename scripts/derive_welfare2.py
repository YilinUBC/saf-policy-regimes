"""
WELFARE AT THE MANDATE CORNER -- FULL DERIVATIVE, NOT A SKETCH.

The issue: the Result 4 condition chi*E > markup may be
missing a tau*E term, because marginal cost already contains the carbon
payment while government revenue collects it. Claim: the correct condition is

    chi*E  >  markup + tau*R - phi*(s_b*sigma_b + s_s*sigma_s)

This script derives dW/dtau from an explicitly stated W with NO shortcuts,
so the tau term either appears or it does not.

Setup at the corner (s_b = theta - theta_s, s_s = theta_s fixed by mandate):
  W(tau) = CS(tau) + PS(tau) + G(tau) - D(tau)
with
  CS = M * (-ln(varsigma_0))/alpha        Small-Rosen log-sum
  PS = N * [ (p - mc) * M*varsigma - kappa/2 f^2 ]
  G  = Q * [ tau*R_pax - phi*(s_b*sigma_b + s_s*sigma_s) ]   <- REGULATED
  D  = chi * Q * E_pax,      Q = M * N * varsigma

Everything depends on tau ONLY through mc (blending is pinned at the corner),
so we can write d/dtau = (dmc/dtau) * d/dmc and work with a single scalar.
"""
import sympy as sp

alpha = sp.symbols('alpha', positive=True)
M, N = sp.symbols('M N', positive=True)
phi = sp.symbols('phi', positive=True)
tau = sp.symbols('tau', nonnegative=True)
chi = sp.symbols('chi', positive=True)
E = sp.symbols('E_pax', positive=True)        # LIFECYCLE emissions per pax (damages)
R = sp.symbols('R_pax', positive=True)        # REGULATED emissions per pax (revenue)
S = sp.symbols('Sigma', nonnegative=True)     # phi*(s_b*sigma_b + s_s*sigma_s) >= 0
mc = sp.symbols('mc', positive=True)

# ---------------------------------------------------------------- primitives
# Logit: own share varsigma(mc), outside share varsigma_0(mc).
# Keep them as unspecified functions -- the result must not depend on their form.
vs = sp.Function('varsigma')(mc)     # own share of ONE airline
vs0 = sp.Function('varsigma_0')(mc)  # outside share

Q = M*N*vs                           # total route traffic
markup = 1/(alpha*(1-vs))            # Lemma 1

print("="*78)
print("WELFARE DERIVATIVE AT THE MANDATE CORNER")
print("="*78)

# ---------------------------------------------------------------- components
CS = M*(-sp.log(vs0))/alpha
PS = N*markup*M*vs                   # variable profit; freq cost is tau-invariant
G  = Q*(tau*R - S)
D  = chi*Q*E
W  = CS + PS + G - D

# ------------------------------------------------- the two Small-Rosen facts
# For logit with the outside option normalised to zero utility:
#   d varsigma_0 / d mc = alpha * varsigma_0 * varsigma_total   (cost up -> outside up)
# where varsigma_total = N*vs is the inside share. Hence
#   dCS/dmc = -M * varsigma_total.   (Roy's identity: envelope on the log-sum)
# We impose this rather than assume a functional form.
vs_tot = N*vs
dCS_dmc = -M*vs_tot

dvs_dmc = sp.Derivative(vs, mc)

print("\nInputs (imposed, not assumed functional forms):")
print("  dCS/dmc     = -M*N*varsigma          [Small-Rosen envelope / Roy]")
print("  dQ/dmc      =  M*N*dvarsigma/dmc  < 0")
print("  markup      =  1/(alpha*(1-varsigma))  [Lemma 1]")

# ---------------------------------------------------------------- dW/dmc
dPS_dmc = sp.diff(PS, mc)
dG_dmc  = sp.diff(G, mc)
dD_dmc  = sp.diff(D, mc)
dW_dmc  = sp.simplify(dCS_dmc + dPS_dmc + dG_dmc - dD_dmc)

print("\ndW/dmc, term by term:")
print("  dCS/dmc =", dCS_dmc)
print("  dPS/dmc =", sp.simplify(dPS_dmc))
print("  dG/dmc  =", sp.simplify(dG_dmc))
print("  dD/dmc  =", sp.simplify(dD_dmc))

# Substitute dQ/dmc = M*N*vs' and collect on vs'
dQ_dmc = M*N*dvs_dmc

# Rewrite: the CS term and the PS "quantity" term partially cancel.
# Collect everything as A + B*dvarsigma/dmc.
expr = sp.expand(dW_dmc)
B = sp.simplify(sp.diff(expr, dvs_dmc))          # coefficient on dvarsigma/dmc
A = sp.simplify(expr - B*dvs_dmc)                # the rest

print("\nCollecting dW/dmc = A + B * (dvarsigma/dmc):")
print("  A =", sp.simplify(A))
print("  B =", sp.simplify(B))

print("""
INTERPRETATION
  A collects the terms that do NOT run through the traffic response: the
  consumer-surplus loss from a higher fare and the mechanical fiscal/damage
  terms evaluated at fixed Q.
  B collects the marginal social value of a passenger: markup + tax revenue
  - subsidy outlay - damage.
""")

# ---------------------------------------------------------------- the condition
# dW/dtau = (dmc/dtau) * dW/dmc, and dmc/dtau = phi*(regulated blend) = R_pax > 0.
# Carbon pricing raises welfare iff dW/dtau > 0 iff dW/dmc > 0 (since E>0).
print("="*78)
print("THE CONDITION")
print("="*78)
print("""
  dmc/dtau = phi * (regulated blend) = R_pax > 0, so sign(dW/dtau) = sign(dW/dmc).
""")

B_clean = sp.simplify(B/(M*N))
print("  B/(M*N) =", sp.expand(B_clean))
print("""
  Since dvarsigma/dmc < 0, dW/dmc > 0 requires B < 0, i.e.

      markup + tau*R_pax - Sigma - chi*E_pax  <  0
  <=> chi*E_pax  >  markup + tau*R_pax - Sigma

  where Sigma = phi*(s_b*sigma_b + s_s*sigma_s) is the per-passenger subsidy
  outlay. CONCLUSION: the tau*E term belongs in the condition.
""")

# ---------------------------------------------------------------- numbers
print("="*78)
print("CALIBRATED NUMBERS")
print("="*78)
MJ_PER_L = 43.0*0.8
CI_F_n = 89.0*MJ_PER_L/1000
CI_B_n = 13.9*MJ_PER_L/1000
CI_S_n = 5.0*MJ_PER_L/1000
FARE, ELAST, SHARE = 120.0, -1.4, 0.25
ALPHA = -ELAST/(FARE*(1-SHARE))
MARKUP = 1.0/(ALPHA*(1-SHARE))
TAU = 80.0

E_F_n = 3.16*0.8            # EU ETS factor, kgCO2/L; SAF zero-rated

def epax_t(phi_, th, ths):
    """LIFECYCLE emissions per pax, tonnes -- enters DAMAGES."""
    ci = (1-th)*CI_F_n + (th-ths)*CI_B_n + ths*CI_S_n
    return phi_*ci/1000.0

def rpax_t(phi_, th, ths, lifecycle_base=False):
    """REGULATED emissions per pax, tonnes -- enters REVENUE and dmc/dtau."""
    if lifecycle_base:
        return epax_t(phi_, th, ths)
    return phi_*((1-th)*E_F_n)/1000.0     # only the fossil fraction is charged

print(f"  markup = EUR {MARKUP:.2f}   (elasticity {ELAST}, fare EUR {FARE}, share {SHARE})")
print(f"  tau    = EUR {TAU}/tCO2e,  no SAF support in the EU baseline (Sigma = 0)")
print()
print("  chi* = [ markup + tau*R_pax ] / E_pax        (Sigma = 0)")
print()
print(f"  {'year':<7}{'phi':>4}{'E_pax':>10}{'R_pax':>10}"
      f"{'chi* ETS':>10}{'chi* life':>11}{'naive':>9}")
print("  " + "-"*61)
for phi_ in (35.0, 45.0):
    for y, th, ths in [("2025",0.02,0.0), ("2030",0.06,0.012),
                       ("2035",0.20,0.05), ("2050",0.70,0.35)]:
        Ep = epax_t(phi_, th, ths)
        Rp = rpax_t(phi_, th, ths)
        Rl = rpax_t(phi_, th, ths, lifecycle_base=True)
        ets  = (MARKUP + TAU*Rp)/Ep
        life = (MARKUP + TAU*Rl)/Ep         # = tau + markup/E when R=E
        naive = MARKUP/Ep                   # the pre-correction figure
        print(f"  {y:<7}{phi_:>4.0f}{Ep:>10.5f}{Rp:>10.5f}"
              f"{ets:>10.0f}{life:>11.0f}{naive:>9.0f}")

print("""
  Three columns, three specifications:
    naive      chi* = markup/E_pax            -- omits forgone carbon revenue
    chi* life  chi* = tau + markup/E_pax      -- correct IF the carbon price
                                                 were levied on lifecycle CO2e
    chi* ETS   chi* = [markup + tau*R]/E_pax  -- correct under the actual ETS,
                                                 where SAF is zero-rated

  Because R_pax < E_pax (only the fossil fraction is charged, and only its
  combustion CO2), the ETS threshold sits BELOW the lifecycle one. The gap
  widens as the mandate tightens, since a cleaner blend shrinks R_pax faster
  than E_pax.
""")

print("="*78)
print("CHECK: the old sketch in derive_welfare.py")
print("="*78)
print("""
  The old script wrote
      dW/dtau = -(dQ/dtau)*[chi*E_pax - markup]
  which silently dropped BOTH the fiscal term (+tau*E per lost passenger,
  revenue the government no longer collects) and the subsidy saving (-Sigma).
  It is a partial-equilibrium sketch, not the derivative of the W stated at
  the top of that file. This script supersedes it.
""")

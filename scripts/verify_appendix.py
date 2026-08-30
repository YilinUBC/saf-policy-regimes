"""
Verification of every claim to appear in Appendix A.
Each block prints an expression that must be 0 (or a stated sign).
"""
import sympy as sp

alpha, be, bb, bs, bf = sp.symbols('alpha beta_e beta_b beta_s beta_f', positive=True)
gam, gb, gs, phi, kap = sp.symbols('gamma gamma_b gamma_s phi kappa', positive=True)
th, ths = sp.symbols('theta theta_s', nonnegative=True)
tau, sig, sgb, sgs = sp.symbols('tau sigma sigma_b sigma_s', nonnegative=True)
PF, PB, PS = sp.symbols('P_F P_B P_S', positive=True)
CIF, CIB, CIS = sp.symbols('CI_F CI_B CI_S', positive=True)   # lifecycle: damages
eF, eB, eS = sp.symbols('e_F e_B e_S', nonnegative=True)      # regulatory: carbon-price base
M, N, q, f, s = sp.symbols('M N q f s', positive=True)
q0 = sp.symbols('q_0', positive=True)
ok = lambda lbl, e: print(f'  {lbl:<52} {sp.simplify(e)}')

print('='*78); print('LEMMA 2  --  envelope: dmc/ds at the profit optimum'); print('='*78)
D = (PB - sgb + tau*eB) - (PF + tau*eF)
sv = sp.symbols('s_v')
mc_gen = phi*((1-sv)*(PF+tau*eF) + sv*(PB-sgb+tau*eB)) + gam/2*(sv-th)**2
s_star = th + (bb/alpha - phi*D)/gam
ok('dmc/ds at s*  minus  beta_b/alpha  (=0)',
   sp.diff(mc_gen, sv).subs(sv, s_star) - bb/alpha)
ok('cost-min s  minus  (theta - phi*Delta/gamma)  (=0)',
   sp.solve(sp.Eq(sp.diff(mc_gen, sv), 0), sv)[0] - (th - phi*D/gam))
ok('gap (profit-max minus cost-min) minus beta_b/(alpha*gamma)',
   s_star - (th - phi*D/gam) - bb/(alpha*gam))

print('='*78); print('LEMMA 3  --  block-diagonal Hessian at interior s*'); print('='*78)
p = (bf*f + be*s - sp.log(q/q0))/alpha
c_fuel = phi*((1-s)*(PF+tau*eF) + s*(PS-sig+tau*eS))
c_saf = gam/2*(s-th)**2
pi = (p - c_fuel - c_saf)*q - kap/2*f**2
H = sp.Matrix([[sp.diff(pi,a,b) for b in (q,f,s)] for a in (q,f,s)])
Dw = (PS - sig + tau*eS) - (PF + tau*eF)
sst = th + (be/alpha - phi*Dw)/gam
ok('d2pi/dq ds at s*  (=0, block-diagonality)', H[0,2].subs(s, sst))
Ho = sp.simplify(H.subs(s, sst))
m1 = Ho[0,0]; m2 = sp.expand(Ho[:2,:2].det()); m3 = sp.expand(Ho.det())
ok('H11 + 1/(alpha*q)   (=0)', m1 + 1/(alpha*q))
ok('|H2| - [kappa/(alpha q) - beta_f^2/alpha^2]', m2 - (kap/(alpha*q) - bf**2/alpha**2))
ok('detH + gamma*q*|H2|   (=0)', m3 + gam*q*m2)

print('='*78); print('LEMMA 4  --  Cournot dominance factorisation'); print('='*78)
x = sp.symbols('x', positive=True)
num = (M**2 - M*N**2*q + M*q + N**3*q**2 - 2*N**2*q**2 + N*q**2)
ok('numerator - (M-Nq+q)(M... ) factorisation check',
   sp.factor(num) - sp.factor((-M + N*q - q)*(-M + N**2*q - N*q)))
numx = sp.simplify(sp.expand(num.subs(q, x*M/N)/M**2))
ok('normalised numerator - (N-Nx+x)(1+x-Nx)/N',
   sp.simplify(numx - (N - N*x + x)*(1 + x - N*x)/N))

print('='*78); print('LEMMA 6  --  mode invariance (Bertrand s* = Cournot s*)'); print('='*78)
shr = sp.symbols('varsigma', positive=True)
dmc_ds = sp.diff(mc_gen, sv)
s_bert = sp.solve(sp.Eq(bb/alpha - dmc_ds, 0), sv)[0]
ok('s*_Bertrand - s*_Cournot  (=0)', s_bert - s_star)

print('='*78); print('PROPOSITION 1  --  two-fuel separation'); print('='*78)
sb, ss = sp.symbols('s_b s_s', nonnegative=True)
Db = (PB - sgb + tau*eB) - (PF + tau*eF)
Ds = (PS - sgs + tau*eS) - (PF + tau*eF)
c2 = phi*((1-sb-ss)*(PF+tau*eF) + sb*(PB-sgb+tau*eB) + ss*(PS-sgs+tau*eS)) \
     + gb/2*(sb-th)**2 + gs/2*(ss-ths)**2
sol = sp.solve([sp.Eq(bb/alpha - sp.diff(c2,sb),0),
                sp.Eq(bs/alpha - sp.diff(c2,ss),0)], [sb,ss], dict=True)[0]
ok('s_b* - [theta + (beta_b/alpha - phi*Delta_b)/gamma_b]',
   sol[sb] - (th + (bb/alpha - phi*Db)/gb))
ok('s_s* - [theta_s + (beta_s/alpha - phi*Delta_s)/gamma_s]',
   sol[ss] - (ths + (bs/alpha - phi*Ds)/gs))
ok('cross-effect ds_b/dsigma_s  (=0)', sp.diff(sol[sb], sgs))
ok('cross-effect ds_s/dsigma_b  (=0)', sp.diff(sol[ss], sgb))
ok('ds_b/dtau - phi(e_F-e_B)/gamma_b', sp.diff(sol[sb],tau) - phi*(eF-eB)/gb)

print('='*78); print('PROPOSITION 2  --  equivalence failure along iso-average-cost'); print('='*78)
AC = phi*((1-th)*(PF+tau*eF) + th*(PB-sgb+tau*eB))
iso = -sp.diff(AC,tau)/sp.diff(AC,sgb)
tot = sp.diff(s_star,tau) + sp.diff(s_star,sgb)*iso
ok('ds*/dtau|_isoAC  -  e_F*phi/(gamma*theta)', sp.simplify(tot - eF*phi/(gam*th)))

print('='*78); print('COROLLARY 2.1  --  blend-neutral support direction'); print('='*78)
# the unique direction holding BOTH shares fixed
x_b, x_s = sp.symbols('x_b x_s')
tb = sp.diff(sol[sb], tau) + sp.diff(sol[sb], sgb)*x_b
ts = sp.diff(sol[ss], tau) + sp.diff(sol[ss], sgs)*x_s
dirs = sp.solve([sp.Eq(tb,0), sp.Eq(ts,0)], [x_b,x_s], dict=True)[0]
ok('dsigma_b/dtau + (e_F - e_b)  (=0)', sp.simplify(dirs[x_b] + (eF-eB)))
ok('dsigma_s/dtau + (e_F - e_s)  (=0)', sp.simplify(dirs[x_s] + (eF-eS)))
# residual average-cost drift along that direction
AC2 = phi*((1-(th-ths)-ths)*(PF+tau*eF) + (th-ths)*(PB-sgb+tau*eB)
           + ths*(PS-sgs+tau*eS))
drift = sp.diff(AC2,tau) + sp.diff(AC2,sgb)*dirs[x_b] + sp.diff(AC2,sgs)*dirs[x_s]
ok('dAC/dtau|blend-neutral  -  phi*e_F', sp.simplify(drift - phi*eF))

print('='*78); print('PROPOSITION 1 (Regime II)  --  crowding-out under a binding total mandate'); print('='*78)
lam = sp.symbols('lambda')
ab, as_ = sp.symbols('a_b a_s', nonnegative=True)
r2 = sp.solve([sp.Eq(bb/alpha - phi*Db - gb*(sb-ab) + lam, 0),
               sp.Eq(bs/alpha - phi*Ds - gs*(ss-as_) + lam, 0),
               sp.Eq(sb+ss, th)], [sb, ss, lam], dict=True)[0]
ok('ds_s/dsigma_b + phi/(gamma_b+gamma_s)',
   sp.simplify(sp.diff(r2[ss], sgb) + phi/(gb+gs)))
ok('ds_b/dsigma_s + phi/(gamma_b+gamma_s)',
   sp.simplify(sp.diff(r2[sb], sgs) + phi/(gb+gs)))
ok('d(s_b+s_s)/dsigma_b at binding total mandate  (=0)',
   sp.simplify(sp.diff(r2[sb]+r2[ss], sgb)))

print('='*78); print('LEMMA 7  --  SOC with ENDOGENOUS outside option'); print('='*78)
Mv, Nv, qv, fv, kap, bf = sp.symbols('M N q f kappa beta_f', positive=True)
p_end = (bf*fv + bb*sv - sp.log(qv/(Mv - Nv*qv)))/alpha
pi_end = (p_end - (phi*((1-sv)*(PF+tau*eF) + sv*(PB-sgb+tau*eB))
                   + gam/2*(sv-th)**2))*qv - kap/2*fv**2
H = sp.hessian(pi_end, (qv, fv, sv))
vs0 = 1 - Nv*qv/Mv
ok('H11 + 1/(alpha*q*varsigma_0^2)', sp.simplify(H[0,0] + 1/(alpha*qv*vs0**2)))
ok('|H2| - [kappa/(alpha q vs0^2) - beta_f^2/alpha^2]',
   sp.simplify(H[:2,:2].det() - (kap/(alpha*qv*vs0**2) - bf**2/alpha**2)))
# The factorisation detH = -gamma*q*|H2| holds AT the interior optimum in s,
# where the (q,s) cross-partial vanishes (Lemma 3). Evaluate there.
s_opt = sp.solve(sp.Eq(sp.diff(pi_end, sv), 0), sv)[0]
H_opt = H.subs(sv, s_opt)
ok('cross-partial d2pi/dq ds at s*  (=0)', sp.simplify(H_opt[0,2]))
ok('detH + gamma*q*|H2|  at s*',
   sp.simplify(H_opt.det() + gam*qv*H_opt[:2,:2].det()))

print('='*78); print('RESULT 3/5  --  corner comparative statics'); print('='*78)
sb_c = th - ths; ss_c = ths
mc_c = phi*((1-sb_c-ss_c)*(PF+tau*eF) + sb_c*(PB-sgb+tau*eB) + ss_c*(PS-sgs+tau*eS))
E = phi*((1-sb_c-ss_c)*CIF + sb_c*CIB + ss_c*CIS)
ok('dE/dtau at corner  (=0)', sp.diff(E,tau))
ok('dE/dsigma_b at corner  (=0)', sp.diff(E,sgb))
ok('dmc/dsigma_b + phi*(theta-theta_s)  (=0)', sp.diff(mc_c,sgb) + phi*(th-ths))
R = phi*((1-sb_c-ss_c)*eF + sb_c*eB + ss_c*eS)          # regulated, revenue base
ok('dmc/dtau - R_pax  (regulated base, not lifecycle)', sp.diff(mc_c,tau) - R)
ok('E_pax - R_pax under ETS zero-rating  (SAF terms + fossil CI_F-e_F gap)',
   sp.simplify((E - R).subs({eB: 0, eS: 0})
               - phi*((1-sb_c-ss_c)*(CIF-eF) + sb_c*CIB + ss_c*CIS)))

print('='*78); print('RESULT 4  --  forgone revenue uses REGULATED emissions R_pax'); print('='*78)
chi, Sig, vsi = sp.symbols('chi Sigma varsigma_i', positive=True)
Ev, Rv, mcv = sp.symbols('E_pax R_pax mc', positive=True)
vs = sp.Function('vs')(mcv)
Qv = Mv*Nv*vs
W = Mv*(-sp.log(sp.Function('vs0')(mcv)))/alpha \
    + Nv*(1/(alpha*(1-vs)))*Mv*vs + Qv*(tau*Rv - Sig) - chi*Qv*Ev
dW = sp.diff(W, mcv)
# impose Roy: dCS/dmc = -M*N*vs  (replace the log-sum derivative)
dW = dW.subs(sp.Derivative(sp.Function('vs0')(mcv), mcv),
             alpha*sp.Function('vs0')(mcv)*Nv*vs)
coef = sp.simplify(sp.diff(dW, sp.Derivative(vs, mcv)))
target = Mv*Nv*(1/(alpha*(1-vs)**2) + tau*Rv - Sig - chi*Ev)
ok('coeff on dvarsigma/dmc - M*N*[markup_deriv + tau*R - Sigma - chi*E]',
   sp.simplify(coef - target))
print()
print('all expressions above must be 0')

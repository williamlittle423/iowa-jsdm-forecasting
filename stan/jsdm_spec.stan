// Bayesian dynamic JSDM implemented to the project specification ("HAB Prediction from Metabarcoding").
//   reads   (Eq. 11):  n_j ~ DirichletMultinomial(kappa * softmax((z_{l,t[j]}, 0)))
//   process (Eq. 15/20): z_{l,t+1} = mu_s + A (z_{l,t} - mu_s) + B x_{l,t} + eps,  eps ~ N_K(0, Sigma)
//   initial (Eq. 26):  z_{l,1} ~ N_K(mu_s, I);  sites: mu_s ~ N_K(mu0, diag(tau^2))
//   Sigma   (Eq. 24-25): diag(sigma) R diag(sigma); sigma_g ~ HalfNormal(0.5); R ~ LKJ(2)
//   priors  (Eq. 21-22, 26): mu0_g ~ N(mu0_g, 1.5^2); B ~ N(0, 0.5^2); log kappa ~ N(log 50, 1);
//            A_gg ~ N(0.5, 0.2^2), A_gl ~ N(0, sd_off^2) (sd_off = 0.15), jointly RESTRICTED to rho(A) < 1 (Eq. 23).
//   toxin   (Eq. 41-42): log V ~ N(aV + cV' z, sV^2); below detection: Pr(log V < log d) (censored, never substituted).
//   Stability is built into the fitted prior or a suitable stable parameterization
//   full A:  A = L Q L^{-1},  Q = Bm (I + Bm Bm')^{-1/2},  L = chol(Sigma)  (Ansley & Kohn 1986; Heaps 2022). For
//            fixed Sigma this maps R^{KxK} one-to-one ONTO {A : rho(A) < 1}. The prior density is placed on A itself
//            (the restricted normal above) and the change of variables Bm -> A is corrected with log|dA/dBm|
//            (= log|dQ/dBm|, since similarity by fixed L has unit Jacobian), computed by central differences.
//   diagonal A: A = diag(a), a_g in (-1, 1) with the N(0.5, 0.2^2) prior = the same prior restricted to rho(A) < 1.
functions {
  matrix qmap(vector b, int K) {
    matrix[K, K] Bm = to_matrix(b, K, K);
    matrix[K, K] S = add_diag(tcrossprod(Bm), 1.0);
    vector[K] lam = eigenvalues_sym(S);
    matrix[K, K] V = eigenvectors_sym(S);
    return Bm * (V * diag_matrix(inv_sqrt(lam)) * V');
  }
  real log_jac_qmap(vector b, int K, real h) {
    int D = K * K;
    matrix[D, D] Jm;
    for (i in 1:D) {
      vector[D] e = rep_vector(0, D);
      e[i] = h;
      Jm[, i] = (to_vector(qmap(b + e, K)) - to_vector(qmap(b - e, K))) / (2 * h);
    }
    return log_determinant(Jm);
  }
}
data {
  int<lower=2> G;                              // groups; group G is the reference ("Other")
  int<lower=1> S;                              // sites (beaches)
  int<lower=1> L;                              // series (beach-seasons)
  int<lower=2> TT;                             // weekly grid
  array[L] int<lower=1, upper=S> site;
  int<lower=0> P;
  array[L] matrix[TT, P] X;                    // origin-available, training-standardised environment
  int<lower=0> J;                              // libraries (replicates share their parent state)
  array[J, G] int<lower=0> n;
  array[J] int<lower=1, upper=L> jl;
  array[J] int<lower=1, upper=TT> jt;
  int<lower=0> NV;                             // toxin measurements
  array[NV] int<lower=1, upper=L> vl;
  array[NV] int<lower=1, upper=TT> vt;
  vector[NV] lv;                               // natural log toxin; log(detection limit) when censored
  array[NV] int<lower=0, upper=1> cens;
  vector[G - 1] mu00;
  real aV0;
  int<lower=0, upper=1> full_A;
  real<lower=0> sd_off;
  int<lower=0, upper=1> use_lik;               // 0 = prior only (prior predictive checks)
  int<lower=0> NO;                             // series whose terminal state is saved for forecasting
  array[NO] int<lower=1, upper=L> ol;
  array[NO] int<lower=1, upper=TT> ot;
}
transformed data {
  int K = G - 1;
  real h_fd = 1e-5;
}

parameters {
  vector[K] mu0;
  vector<lower=0>[K] tau_mu;
  array[S] vector[K] mu_raw;
  vector[full_A == 1 ? K * K : 0] b;
  vector<lower=-1, upper=1>[full_A == 1 ? 0 : K] a_diag;
  matrix[K, P] B;
  vector<lower=0>[K] sigma;
  cholesky_factor_corr[K] L_R;
  real log_kappa;
  array[L] matrix[K, TT] eps;
  real aV;
  vector[K] cV;
  real<lower=0> sV;
}

transformed parameters {
  matrix[K, K] L_S = diag_pre_multiply(sigma, L_R);
  matrix[K, K] A;
  real log_jac = 0;
  array[S] vector[K] mu;
  if (full_A == 1) {
    A = mdivide_right(L_S * qmap(b, K), L_S);
    log_jac = log_jac_qmap(b, K, h_fd);
  } else {
    A = diag_matrix(a_diag);
  }
  for (s in 1:S) mu[s] = mu0 + tau_mu .* mu_raw[s];
}

model {
  if (full_A == 1) {
    for (r in 1:K) for (c in 1:K) A[r, c] ~ normal(r == c ? 0.5 : 0.0, r == c ? 0.2 : sd_off);
    target += log_jac;
  } else {
    a_diag ~ normal(0.5, 0.2);
  }
  mu0 ~ normal(mu00, 1.5);
  tau_mu ~ normal(0, 1);
  for (s in 1:S) mu_raw[s] ~ std_normal();
  to_vector(B) ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
  L_R ~ lkj_corr_cholesky(2);
  log_kappa ~ normal(log(50), 1);
  for (l in 1:L) to_vector(eps[l]) ~ std_normal();
  aV ~ normal(aV0, 2);
  cV ~ normal(0, 0.5);
  sV ~ normal(0, 1);
  if (use_lik == 1) {
    real kap = exp(log_kappa);
    array[L] matrix[K, TT] z;
    for (l in 1:L) {
      vector[K] m = mu[site[l]];
      z[l][, 1] = m + eps[l][, 1];
      for (t in 1:(TT - 1)) z[l][, t + 1] = m + A * (z[l][, t] - m) + B * X[l][t]' + L_S * eps[l][, t + 1];
    }
    for (j in 1:J) n[j] ~ dirichlet_multinomial(kap * softmax(append_row(z[jl[j]][, jt[j]], 0)));
    for (i in 1:NV) {
      real eta = aV + dot_product(cV, z[vl[i]][, vt[i]]);
      if (cens[i] == 1) target += normal_lcdf(lv[i] | eta, sV);
      else target += normal_lpdf(lv[i] | eta, sV);
    }
  }
}

generated quantities {
  matrix[K, K] Sigma = multiply_lower_tri_self_transpose(L_S);
  real rhoA = max(abs(get_real(eigenvalues(A))));   // spectral radius (eigenvalues may be complex)
  array[NO] vector[K] zT;
  {
    for (o in 1:NO) {
      int l = ol[o];
      vector[K] m = mu[site[l]];
      vector[K] zz = m + eps[l][, 1];
      for (t in 1:(ot[o] - 1)) zz = m + A * (zz - m) + B * X[l][t]' + L_S * eps[l][, t + 1];
      zT[o] = zz;
    }
  }
}

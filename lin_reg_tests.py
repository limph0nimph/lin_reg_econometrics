import scipy as sp
import numpy as np
class LinearRegression():
  def __init__(self, regularize = 0):
    self.regularize = 0

  def fit(self, X, y):
    '''
    X vector of shape(n_observations, n_features)
    y vector of shape  (n_observations, 1)
    returns beta of shape n_features
    '''
    self.n_observations = X.shape[0]
    self.n_features =  X.shape[1]
    assert X.shape[0] == len(y)

    self.X = X
    self.y = y
    beta  = np.linalg.inv(X.T@X + self.regularize*np.eye(X.shape[1]))@X.T@y

    self.beta = beta

    return beta

  def F_test(self, A, c, alpha_test = 0.05 , r_A=0):

    '''
      test restrictions Ab=c
      A matrix (r_A,  n_features) of retrictions on beta
      r_A rank of matrix A
      c vector right part of restrictions
      X vector of shape(n_observations, n_features)
      y vector of shape  (n_observations, 1)
      returns beta of shape n_features
    '''
    n = self.n_observations
    p = self.n_features
    X = self.X
    y = self.y
    inv_XT_X = np.linalg.inv(X.T@X)
    self.beta_restricted = self.beta - inv_XT_X@A.T@(np.linalg.inv(A@inv_XT_X@A.T))@(A@self.beta -c)
    if r_A == 0:
      r_A = np.linalg.matrix_rank(A)
    RSS_UR = np.sum((y - X@self.beta)**2)
    RSS_R =  np.sum((y - X@self.beta_restricted)**2)
    F_st = ((RSS_R - RSS_UR ) / r_A) /((RSS_UR)/(n-p))
    p_value = 1- sp.stats.f.cdf(F_st, dfn = r_A , dfd = n-p, loc=0, scale=1)
    print(self.beta_restricted)
    if p_value > alpha_test:
      print ("fail to reject H0 snd restrictions can be valid")
    else:
      print("reject  H 0, restriction model is not valid")

    return {"Fst": F_st,"p_value": p_value}



  def RSS(self):
    return np.sum((self.y - self.X@self.beta)**2)

  def predict(self, X):

    return X@self.beta

  def standard_deviation_beta(self):
      sigma_hat = self.RSS()/(self.n_observations - self.n_features)
      return sigma_hat * np.linalg.inv(self.X.T@self.X)

  def t_test_siginificance(self, alpha_critical):
    '''
    alpha_critical is prob to reject H_o
    '''
    betas_info ={}
    for k in range(self.n_features):
      t_st_beta_k = self.beta[k]/(np.sqrt(self.standard_deviation_beta()[k, k]))
      t_cdf = t.cdf(t_st_beta_k, df = self.n_observations - self.n_features )

      if alpha_critical / 2 < t_cdf < 1 - alpha_critical / 2:
        is_significant = False
        #print(f"fail to reject H_0 that coeffiecint of {k} feature is eqaul to 0 at level {alpha_critical}","non significant")

      else:
        is_significant = True
        #print(f"reject H_0 that coeffiecint of {k} feature is eqaul to 0 at level {alpha_critical}", "significant")
      betas_info[k] = {"t statistics":round(t_st_beta_k, 3),
                       "p_value": round(2*min(1-t_cdf, t_cdf), 3),
                       "is_significant":is_significant}
    return betas_info

  def R_squared(self):
    return 1 - self.RSS() / np.sum((self.y-np.mean(self.y))**2)


 def GoldfeildQuant_test(self, suspicion_feature_ind, critical_level =0.05 ):

    """Computes the White_test test to test null hypothesis if residuals are homoscedastic.

    Parameters
    ----------
    suspicion_feature_ind: int  index of a feature column in self.X according to which we sort
    critical_level: float  уровень значимости


    Returns
    -------
    test_statistic: formula  $F_st$
    p_value: float
      the probability calculated  using  distributions.kstwo.sf(test_statistic, N)
      where N = len(residuals)
    is_homoscedastic: bool
        whether F_st>F_critical then reject H_0 so, is_homoscedastic=0
        if F_st > F_critical then fail to  reject H_0 so, is_homoscedastic=1

    """
    X_data = self.X
    
    part1_index = self.X[:,suspicion_feature_ind]<=np.quantile(self.X[:,suspicion_feature_ind],q=0.34)
    part2_index = self.X[:,suspicion_feature_ind]>=np.quantile(self.X[:,suspicion_feature_ind],q=0.66)

    
    X_part1 = self.X[part1_index]
    y_part1 = self.y[part1_index]
    beta_part1 = np.linalg.inv(X_part1.T@X_part1)@X_part1.T@y_part1
    RSS_part1 = np.sum((y_part1- X_part1@beta_part1)**2)

    
    X_part2 = self.X[part2_index]
    y_part2 = self.y[part2_index]
    beta_part2 = np.linalg.inv(X_part2.T@X_part2)@X_part2.T@y_part2
    RSS_part2 = np.sum((y_part2- X_part2@beta_part2)**2)

    n1 = self.X[part1_index].shape[0]
    n2 =self.X[part2_index].shape[0]
    k =  self.X[part2_index].shape[1]
    
    F_statistics = RSS_part2*(n1-k)/((n2-k)*RSS_part1)
    
    p_value = 1 - scipy.stats.f.cdf(F_statistics, dfn =n2-k , dfd =n1 -k, loc=0, scale=1)
   
    if p_value < critical_level:
        is_homoscedastic = 0
    else:
        is_homoscedastic = 1
    return F_statistics, p_value, is_homoscedastic

  def White_test(self, critical_level = 0.05, constant_included = True):
     """Computes the White_test test to test null hypothesis if residuals are homoscedastic.

    Parameters
    ----------
    critical_level: float
    constant_included: bool
          if constant_included==True then it means that self.X[0] is vector of ones
           else then it means thatconstant is not included in a original regression


    Returns
    -------
    test_statistic: formula $nR^2$

    p_value: float

    is_homoscedastic: bool
        whether p-value> critica_value then reject H_0 so, is_homoscedastic=0
        if p-value<= critica_value then fail to  reject H_0 so, is_homoscedastic=1

    """
    #calculate residuals terms 
     residuals = self.y - self.predict(self.X)
     if constant_included:
      stacked_data = []
      #calculate cross terms
      for i in range(1, self.X.shape[1]):
        for j in range(1,i+1):
          cross_term = self.X[:,i]*self.X[:,j]
          stacked_data.append(cross_term)

      stacked_data = np.asarray(stacked_data).transpose()
      #calculate  regressors for residuals terms (2 point)
      stacked_data = np.concatenate([self.X, stacked_data],axis=1)
      residuals2 = residuals**2
      
      beta_res = np.linalg.inv(stacked_data.T@stacked_data)@stacked_data.T@residuals2
      RSS_res = np.sum((residuals2 - stacked_data@beta_res)**2)
      TSS_res = np.sum((residuals2 - np.mean(residuals))**2)

      #calculate statistics AND P VALUE  and is_homoscedastic
      n = self.X.shape[0]
      test_statistic = n*(1- (RSS_res/TSS_res))
      df = stacked_data.shape[1]-1
      p_value = 1-chi2.cdf(test_statistic, df)

      if p_value < critical_level:
        is_homoscedastic = 0
      else:
        is_homoscedastic = 1
      return test_statistic, p_value, is_homoscedastic


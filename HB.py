import numpy as np
from scipy.special import gamma
import torch

def load_tree_data()
    return

class hbn_tree():
    def __init__(self):
        self.data = None #TODO
        #global tree variables or constants
        self.a0 = 1
        self.b0 = 1
        self.t = 3
        self.nu = 1
        self.gamma = self.sample_gamma(self, [1], [1], (1,1))

        #data defined variables
        self.num_features = -1 #implement data loader for this
        self.num_classes = -1 #implement data loader for this
            #could predine in data... but do not know yet
        self.z_b = np.array() #indices are images and values are classes
        self.z_s = np.array() #indices are classes and values are categories

        #level three parameters
        self.alpha_3 = self.sample_gamma(1,1)
        self.tau_3 = self.sample_gamma(1,1)
        #level two parameters
        init_size = (self.z_s.max(), self.num_features)
        self.mu_2 = self.sample_norm(0, 1/self.tau_3, init_size)
        self.alpha_2 = np.random.exponential(self.alpha_3, init_size)
        self.tau_2 = 1/self.sample_gamma(self.a0, self.b0, init_size)
        #level one parameters
        self.mu_1 = np.zeros((self.num_classes, self.num_features))
        self.tau_1 = np.zeros((self.num_classes, self.num_features))
        for class_ind, category in enumerate(self.z_s):
            mu = self.mu_2[category]
            nu = self.nu
            alpha = self.alpha_2[category]
            beta = self.alpha_2[category]/self.tau_2[category]

            size = (1, self.num_features
            self.tau_1[class_ind] = self.sample_gamma(alpha, beta, size)
            self.mu_1[class_ind] = self.sample_norm(mu, 1/(nu*self.tau_1[class_ind]), size)


    def sample_gamma(self, alpha, beta, size:tuple = None):
        return np.random.gamma(alpa, 1/beta, size)

    def sample_norm(self, mu, tau, size:tuple = None):
        sigma = np.sqrt(tau)
        return np.random.normal(mu, sigma, size)

    def sample_tree(self, gibbs_steps=1000):
        for _ in gibbs_steps:
            self.update_step()

    def level_two_helper(alpha_k, alpha_0, children_tau, tau_k):
        S_k = np.sum(tau_children)
        T_k = np.sum(np.log(tau_children))
        n_k = tau_children.shape[0]
        exp_mult = (alpha_0+S_k/tau_k-T_k)

        exp_overall = np.exponential(-alpha_k*exp_mult)
        top_term = (alpha_k/tau_k)**(alpha_k*n_k);
        bottom_term = gamma(alpha_k)**n_k;
        return exp_overall*top_term/bottom_term;

    def update_step(self):
        size = (1, self.num_features) #1byD
        #level one update step
        for class_ind, category in enumerate(self.z_s):
            X_chunk = data.X[:,np.argwhere(data.z_b==class_ind)[:,0]]
            X_mean = X_chunk.mean(0)
            n = X_chunk.shape[1]

            alpha = self.alpha_2[category]+n/2
            beta = self.alpha_2[category]/self.tau_2[category]+ \
                    .5*(n*self.nu)/(self.nu+n)*self.mu_2[category]**2
            mu = (self.nu*self.mu_2+n*X_mean)/(self.nu+n)
            nu = self.nu+n

            self.tau_1[class_ind] = self.sample_gamma(alpha, beta, size)
            self.mu_1[class_ind] = self.sample_norm(mu, 1/(nu*self.tau_1[class_ind]), size)

        #level two update step
        for category in range(self.z_s.max()):
            nodes_children = np.argwhere(data.z_s==category)[:,0]
            children_tau = self.tau_1[nodes_children,:]
            children_mu = self.mu_1[nodes_children,:]

            norm_prec = (1/self.tau_3)+np.sum(1/(self.nu*children_tau))
            norm_mean = np.sum(children_mu/(self.nu*children_tau))/norm_prec

            self.mu_2 = self.sample_norm(norm_mean, norm_prec, size)

            alpha_inv = self.a0+nodes_children.shape[0]*self.alpha_2[category]
            beta_inv self.b0+self.alpha_2[category]*np.sum(children_tau)

            self.tau_2 = 1/self.sample_gamma(alpha_inv, beta_inv, size)

            alpha_2_prop = self.sample_gamma(self.t, self.t/self.alpha_2[category], size)
            prob_prop = self.level_two_helper(alpha_2_prop, self.alpha_3, children_tau, self.tau_2)
            prob_prev = self.level_two_helper(self.alpha_2[category], self.alpha_3, children_tau, self.tau_2)
            accept = np.min(prob_prop/prob_prev,0)
            assign_new_values = np.random.random((1, accept.shape[0]))
            ind_to_change = np.argwhere(assign_new_values > 1-accept)

            self.alpha_2[category, ind_to_change] = alpha_2_prop[ind_to_change]

        #level three update step
        KD = self.num_classes*self.num_features
        self.alpha_3 = self.sample_gamma(1+KD, 1+KD*np.mean(self.alpha_2))
        self.tau_3 = self.sample_gamma(1+KD, 1+np.sum(self.mu_2**2))
